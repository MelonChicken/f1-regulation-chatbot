import os
from processors.text_processor import (
    load_pdf,
    split_by_article,
    split_into_sections,
    chunk_optimize,
    save_vectorstore, fallback_chunking
)

from processors.table_processor import (
    extract_tables,
    save_table_vectorstore, convert_tables_to_documents
)


# -----------------------------
# 문서 타입 자동 감지
# -----------------------------
def detect_doc_type(filename: str):
    name = filename.lower()

    if "sporting" in name or "section_b" in name:
        return "sporting"

    if "technical" in name or "Technical" in name or "section_c" in name:
        return "technical"

    if "operational" in name or "Operational" in name:
        return "operational"

    return "misc"  # fallback


# -----------------------------
# PDF → text vectorstore
# -----------------------------
def build_text_store(pdf_path, out_dir):
    pages = load_pdf(pdf_path)

    # 1) Sporting 규정 전용 파싱 시도
    article_chunks = split_by_article(pages)

    if len(article_chunks) == 0:
        print("⚠ ARTICLE 패턴이 없어 fallback chunking 사용")
        chunks = fallback_chunking(pages)
    else:
        # Sporting 구조 정상 처리
        sections = []
        for title, body in article_chunks:
            sections.extend(split_into_sections(title, body))

        chunks = chunk_optimize(sections)

    save_vectorstore(chunks, out_dir)



# -----------------------------
# PDF → table vectorstore
# -----------------------------

def build_table_store(pdf_path, out_dir):
    tables = extract_tables(pdf_path)

    # ✔ Table → Document 변환
    docs = convert_tables_to_documents(tables)

    # ✔ Chroma 저장
    save_table_vectorstore(docs, out_dir)


def build_vectorstore_for_single_file(pdf_path):
    doc_type = detect_doc_type(pdf_path)

    text_dir = f"output/chroma/{doc_type}_text"
    table_dir = f"output/chroma/{doc_type}_tables"

    build_text_store(pdf_path, text_dir)
    build_table_store(pdf_path, table_dir)


# -----------------------------
# data 폴더 전체 자동 처리
# -----------------------------
def build_all_vectorstores_from_data():
    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]

    if not pdf_files:
        print("❌ No PDF files found.")
        return

    for pdf in pdf_files:
        pdf_path = os.path.join(data_dir, pdf)
        build_vectorstore_for_single_file(pdf_path)


# 실행용
if __name__ == "__main__":
    build_all_vectorstores_from_data()
