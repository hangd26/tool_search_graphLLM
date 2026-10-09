import os
import json
import re
from io import BytesIO, StringIO
from datetime import datetime

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from openai import OpenAI

import pdfplumber
from pypdf import PdfReader

# ====================== 1. Load environment ======================
load_dotenv()

# ====================== 2. Cấu hình trang ======================
st.set_page_config(
    page_title="Ứng dụng tìm kiếm giao dịch & tài liệu",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Ứng dụng tìm kiếm giao dịch & tài liệu")
st.markdown("Tải file **Excel** hoặc **PDF** từ máy tính → tìm kiếm theo tên → tải báo cáo Excel về máy.")

# ====================== 3. Sidebar ======================
with st.sidebar:
    st.header("⚙️ Cấu hình")
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Hoặc đặt OPENAI_API_KEY trong file .env"
    )
    # Ưu tiên model mạnh cho structured extraction
    model_name = st.selectbox(
        "Model",
        options=[
            "gpt-4o",          
            "gpt-4.1",
            "gpt-4.1-mini",
            "o4-mini",
            "gpt-4o-mini",
            "gpt-4-turbo",
        ],
        index=0
    )
    temperature = st.slider("Temperature", 0.0, 1.0, 0.0, 0.1)


# ====================== Helper functions ======================
def extract_customer_name(prompt: str) -> str:
    """Lấy tên khách hàng từ câu lệnh."""
    prompt_clean = prompt.strip()
    patterns = [
        r"tìm\s*kiếm\s*tất\s*cả\s*giao\s*dịch\s*liên\s*quan\s*đến\s*",
        r"tìm\s*kiếm\s*giao\s*dịch\s*khách\s*hàng\s*",
        r"tìm\s*kiếm\s*giao\s*dịch\s*",
        r"tìm\s*tất\s*cả\s*giao\s*dịch\s*",
        r"giao\s*dịch\s*liên\s*quan\s*đến\s*",
        r"khách\s*hàng\s*",
        r"tên\s*",
    ]
    name = prompt_clean
    for p in patterns:
        name = re.sub(p, "", name, flags=re.IGNORECASE)
    name = name.strip(" :.-{}\"'")
    return name if name else prompt_clean


def create_excel_report(df_result: pd.DataFrame, customer_name: str, source_file: str = "") -> bytes:
    """Tạo file Excel báo cáo."""
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_result.to_excel(writer, sheet_name="Ket_qua", index=False)

        total_amount = "N/A"
        for col in df_result.columns:
            col_lower = str(col).lower()
            if any(k in col_lower for k in ["tiền", "tien", "amount", "phát sinh nợ", "phat_sinh_no", "so_tien"]):
                try:
                    total_amount = f"{pd.to_numeric(df_result[col], errors='coerce').sum():,.0f}"
                except Exception:
                    pass
                break

        summary = pd.DataFrame({
            "Thông tin": [
                "Tên khách hàng",
                "File nguồn",
                "Số giao dịch tìm thấy",
                "Tổng số tiền (nếu có)",
                "Thời gian xuất báo cáo"
            ],
            "Giá trị": [
                customer_name,
                source_file or "N/A",
                len(df_result),
                total_amount,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ]
        })
        summary.to_excel(writer, sheet_name="Tom_tat", index=False)

    output.seek(0)
    return output.getvalue()


def parse_llm_json_to_df(text: str) -> pd.DataFrame:
    """Parse JSON từ output LLM thành DataFrame (xử lý nhiều trường hợp)."""
    if not text or not str(text).strip():
        return pd.DataFrame()

    text = str(text).strip()

    # Bỏ markdown code fence nếu có
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)

    # Trường hợp object bọc array: {"transactions": [...]} hoặc {"data": [...]}
    try:
        obj = json.loads(text)
        if isinstance(obj, list):
            return pd.DataFrame(obj) if obj else pd.DataFrame()
        if isinstance(obj, dict):
            for key in ("transactions", "data", "results", "giao_dich", "items", "records"):
                if key in obj and isinstance(obj[key], list):
                    return pd.DataFrame(obj[key]) if obj[key] else pd.DataFrame()
            # Object đơn
            return pd.DataFrame([obj])
    except json.JSONDecodeError:
        pass

    # Tìm JSON array trong text
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            data = json.loads(match.group())
            if isinstance(data, list) and len(data) > 0:
                return pd.DataFrame(data)
        except json.JSONDecodeError:
            pass

    return pd.DataFrame()


def chunk_text(text: str, max_chars: int = 50000, overlap: int = 1000):
    """Chia text dài thành các chunk có overlap."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start = end - overlap
        if start < 0:
            start = 0
        if end >= len(text):
            break
    return chunks


def extract_transactions_from_pdf(
    client: OpenAI,
    model: str,
    pdf_text: str,
    customer_name: str,
    temperature: float = 0.0
) -> tuple[pd.DataFrame, str]:
    """
    Trích xuất giao dịch từ PDF bằng OpenAI (JSON mode).
    Trả về (DataFrame, raw_response_text).
    """
    system_msg = (
        "Bạn là chuyên gia trích xuất dữ liệu giao dịch ngân hàng từ văn bản PDF. "
        "Luôn trả về JSON hợp lệ theo đúng schema được yêu cầu. "
        "Không thêm giải thích, không thêm markdown."
    )

    user_template = """Từ nội dung PDF dưới đây, hãy tìm TẤT CẢ giao dịch / dòng liên quan đến tên: "{customer_name}"
(so khớp không phân biệt hoa thường; tên có thể nằm ở người chuyển, người nhận, nội dung, hoặc bất kỳ chỗ nào).

YÊU CẦU BẮT BUỘC:
1. Nếu TÌM THẤY: trả về JSON object dạng:
{{
  "found": true,
  "transactions": [
    {{
      "Ngay": "YYYY-MM-DD hoặc nguyên văn trong PDF",
      "So_but_toan": "mã bút toán / mã GD nếu có, null nếu không",
      "Phat_sinh_no": số hoặc null,
      "Phat_sinh_co": số hoặc null,
      "So_tien": số tiền chính nếu có, null nếu không,
      "Nguoi_chuyen": "tên nếu có, null nếu không",
      "Nguoi_nhan": "tên nếu có, null nếu không",
      "Noi_dung": "TOÀN BỘ nội dung/diễn giải giao dịch gốc trong PDF — KHÔNG rút gọn, KHÔNG bỏ chi tiết",
      "Tai_khoan": "số tài khoản nếu có, null nếu không",
      "Ngan_hang": "tên ngân hàng nếu có, null nếu không",
      "Don_vi_thu_huong_don_vi_chuyen": "đơn vị nếu có, null nếu không",
      
      
    }}
  ]
}}

2. Nếu KHÔNG tìm thấy bất kỳ thông tin nào liên quan đến "{customer_name}":
{{
  "found": false,
  "transactions": []
}}

LƯU Ý QUAN TRỌNG về cột Noi_dung:
- Phải lấy ĐẦY ĐỦ nội dung chuyển khoản / diễn giải như trong PDF (ví dụ: "chuyen tien cho Nguyen Van A thanh toan hop dong...", số hóa đơn, mã tham chiếu...).
- Không được để trống hoặc chỉ ghi "chuyen tien" nếu PDF có thêm chi tiết.

========== NỘI DUNG PDF (đoạn) ==========
{chunk}
========== HẾT ==========
"""

    all_rows = []
    raw_parts = []
    chunks = chunk_text(pdf_text, max_chars=50000, overlap=1500)

    for i, chunk in enumerate(chunks):
        user_msg = user_template.format(customer_name=customer_name, chunk=chunk)
        try:
            # JSON mode để ép output hợp lệ
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            )
            raw = resp.choices[0].message.content or ""
            raw_parts.append(raw)

            data = json.loads(raw)
            txs = data.get("transactions") or data.get("data") or []
            if isinstance(txs, list):
                all_rows.extend(txs)
            elif data.get("found") is False:
                continue
        except Exception as e:
            raw_parts.append(f"[Chunk {i+1} error: {e}]")
            continue

    # Gộp & loại trùng gần đúng theo So_but_toan + Ngay + So_tien
    if not all_rows:
        return pd.DataFrame(), "\n---\n".join(raw_parts)

    df = pd.DataFrame(all_rows)
    # Deduplicate
    subset_cols = [c for c in ["So_but_toan", "Ngay", "Phat_sinh_no", "Phat_sinh_co", "So_tien", "Noi_dung"] if c in df.columns]
    if subset_cols:
        df = df.drop_duplicates(subset=subset_cols, keep="first")
    else:
        df = df.drop_duplicates(keep="first")

    return df.reset_index(drop=True), "\n---\n".join(raw_parts)


# ====================== 4. Upload file ======================
st.subheader("1. Tải file từ máy tính")

file_type = st.radio(
    "Loại file:",
    options=["Excel (.xlsx, .xls)", "PDF (.pdf)"],
    horizontal=True
)

df = None
pdf_text = None
file_name = None

if file_type == "Excel (.xlsx, .xls)":
    uploaded_file = st.file_uploader(
        "Chọn file Excel từ máy tính",
        type=["xlsx", "xls"]
    )
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file)
            file_name = uploaded_file.name
            st.success(f"✅ Đã tải file Excel: **{file_name}**")
        except Exception as e:
            st.error(f"Lỗi đọc file Excel: {e}")

else:
    uploaded_file = st.file_uploader(
        "Chọn file PDF từ máy tính",
        type=["pdf"]
    )
    if uploaded_file is not None:
        try:
            file_name = uploaded_file.name
            text_parts = []

            with pdfplumber.open(uploaded_file) as pdf:
                for i, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Trang {i + 1} ---\n{page_text}")

            if not text_parts:
                uploaded_file.seek(0)
                reader = PdfReader(uploaded_file)
                for i, page in enumerate(reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(f"--- Trang {i + 1} ---\n{page_text}")

            pdf_text = "\n\n".join(text_parts) if text_parts else ""

            if pdf_text.strip():
                st.success(f"✅ Đã tải và đọc file PDF: **{file_name}** ({len(text_parts)} trang)")
            else:
                st.warning("⚠️ File PDF không chứa text có thể trích xuất (có thể là file scan/ảnh). Cần OCR.")
        except Exception as e:
            st.error(f"Lỗi đọc file PDF: {e}")


# ====================== 5. Xử lý theo loại file ======================
if df is not None or (pdf_text is not None and pdf_text.strip()):

    # -------------------- EXCEL --------------------
    if df is not None:
        st.subheader("2. Thông tin dữ liệu Excel")
        c1, c2, c3 = st.columns(3)
        c1.metric("Số dòng", df.shape[0])
        c2.metric("Số cột", df.shape[1])
        c3.metric("Tên file", file_name or "N/A")

        with st.expander("Xem trước dữ liệu (5 dòng đầu)"):
            st.dataframe(df.head(), use_container_width=True)

        st.subheader("3. Tìm kiếm giao dịch khách hàng (Excel)")

        if not openai_api_key:
            st.warning("Vui lòng nhập OpenAI API Key ở sidebar.")
            st.stop()

        llm = ChatOpenAI(model=model_name, temperature=temperature, api_key=openai_api_key)
        agent = create_pandas_dataframe_agent(
            llm, df, verbose=True, allow_dangerous_code=True,
            agent_type="openai-tools", handle_parsing_errors=True
        )

        user_prompt = st.text_input(
            "Nhập câu lệnh tìm kiếm:",
            value="tìm kiếm tất cả giao dịch liên quan đến Nguyễn Văn A",
            key="search_excel"
        )

        if st.button("🔍 Tìm kiếm Excel", type="primary", use_container_width=True, key="btn_excel"):
            if not user_prompt.strip():
                st.warning("Vui lòng nhập câu lệnh.")
            else:
                customer_name = extract_customer_name(user_prompt)
                system_instruction = f"""
Bạn là trợ lý tìm kiếm giao dịch.
Tìm TẤT CẢ giao dịch liên quan đến tên khách hàng (cột Người chuyển HOẶC Người nhận, không phân biệt hoa thường).
Nếu tìm thấy: trả về DataFrame.
Nếu không: trả lời "Không tìm thấy tên khách hàng [Tên] trong dữ liệu."
Câu lệnh: {user_prompt}
"""
                with st.spinner("Đang tìm kiếm..."):
                    try:
                        response = agent.invoke({"input": system_instruction})
                        result = response.get("output", response) if isinstance(response, dict) else response
                        st.success("✅ Kết quả từ Agent:")
                        st.write(result)

                        name_cols = [c for c in df.columns if any(
                            k in str(c).lower() for k in ["người", "nguoi", "chuyển", "chuyen", "nhận", "nhan", "tên", "ten", "name"]
                        )]
                        if not name_cols:
                            name_cols = df.columns.tolist()

                        mask = pd.Series([False] * len(df))
                        for col in name_cols:
                            mask = mask | df[col].astype(str).str.contains(customer_name, case=False, na=False)
                        df_result = df[mask].copy()

                        if len(df_result) > 0:
                            st.dataframe(df_result, use_container_width=True)
                            st.subheader("4. Tải báo cáo Excel về máy")
                            excel_bytes = create_excel_report(df_result, customer_name, file_name)
                            st.download_button(
                                label="📥 Tải báo cáo Excel (.xlsx)",
                                data=excel_bytes,
                                file_name=f"Bao_cao_GD_{customer_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key="dl_excel"
                            )
                        else:
                            st.warning(f"Không tìm thấy tên khách hàng **{customer_name}** trong dữ liệu.")
                    except Exception as e:
                        st.error(f"Lỗi: {e}")
                        st.exception(e)

    # -------------------- PDF --------------------
    if pdf_text is not None and pdf_text.strip():
        st.subheader("2. Nội dung file PDF")
        st.caption(f"File: **{file_name}** | Độ dài text: {len(pdf_text):,} ký tự")

        with st.expander("Xem trước nội dung PDF (1500 ký tự đầu)"):
            st.text(pdf_text[:1500] + ("..." if len(pdf_text) > 1500 else ""))

        st.subheader("3. Tìm kiếm giao dịch trong PDF")

        if not openai_api_key:
            st.warning("Vui lòng nhập OpenAI API Key ở sidebar.")
            st.stop()

        pdf_prompt = st.text_input(
            "Nhập câu lệnh tìm kiếm:",
            value="tìm kiếm tất cả giao dịch liên quan đến Nguyễn Văn A",
            placeholder="Ví dụ: tìm kiếm tất cả giao dịch liên quan đến Nguyễn Văn A",
            key="search_pdf"
        )
        st.caption(
            "💡 Dùng **gpt-4o** (mặc định) + JSON mode để trích xuất đủ cột, đặc biệt **Noi_dung** chi tiết. "
            "File dài sẽ được chia chunk tự động."
        )

        if st.button("🔍 Tìm kiếm PDF", type="primary", use_container_width=True, key="btn_pdf"):
            if not pdf_prompt.strip():
                st.warning("Vui lòng nhập câu lệnh tìm kiếm.")
            else:
                customer_name = extract_customer_name(pdf_prompt)
                client = OpenAI(api_key=openai_api_key)

                with st.spinner(f"Đang trích xuất giao dịch từ PDF bằng **{model_name}** (JSON mode)..."):
                    try:
                        df_result, raw_debug = extract_transactions_from_pdf(
                            client=client,
                            model=model_name,
                            pdf_text=pdf_text,
                            customer_name=customer_name,
                            temperature=temperature,
                        )

                        if df_result.empty:
                            st.warning(f"❌ Không tìm thấy tên khách hàng **{customer_name}** trong file PDF.")
                            with st.expander("Xem raw response từ AI (debug)"):
                                st.code(raw_debug[:5000] if raw_debug else "(empty)", language="json")
                        else:
                            st.success(f"✅ Đã tìm thấy **{len(df_result)}** giao dịch/thông tin liên quan đến **{customer_name}**:")
                            st.dataframe(df_result, use_container_width=True)

                            # Cảnh báo nếu Noi_dung quá ngắn / generic
                            if "Noi_dung" in df_result.columns:
                                short_nd = df_result["Noi_dung"].astype(str).str.len() < 15
                                if short_nd.any():
                                    st.info(
                                        "⚠️ Một số dòng có **Noi_dung** ngắn. "
                                        "Nếu PDF có nội dung chi tiết hơn, thử model **gpt-4.1** hoặc **o4-mini**, "
                                        "hoặc kiểm tra phần text đã trích từ PDF ở mục 2."
                                    )

                            st.subheader("4. Tải báo cáo Excel về máy")
                            st.info("Báo cáo gồm sheet **Ket_qua** (chi tiết) và sheet **Tom_tat** (tóm tắt).")

                            excel_bytes = create_excel_report(df_result, customer_name, file_name)
                            st.download_button(
                                label="📥 Tải báo cáo Excel (.xlsx)",
                                data=excel_bytes,
                                file_name=f"Bao_cao_GD_{customer_name.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                use_container_width=True,
                                key="dl_pdf_excel"
                            )

                            with st.expander("Raw AI response (debug)"):
                                st.code(raw_debug[:8000] if raw_debug else "", language="json")

                    except Exception as e:
                        st.error(f"Lỗi khi gọi LLM: {e}")
                        st.exception(e)

else:
    st.info("👆 Vui lòng chọn loại file và tải file Excel hoặc PDF từ máy tính để bắt đầu.")
