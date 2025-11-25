import os
import json
import camelot

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# -----------------------------------------------------------
# 1. PDF에서 표 추출 (Camelot)
# -----------------------------------------------------------
def extract_tables(pdf_path):
    """
    Camelot으로 PDF에서 표 추출.
    표가 0개인 경우에도 안전하게 처리.
    """
    print("Extracting tables from PDF...")

    try:
        tables = camelot.read_pdf(
            pdf_path,
            pages="all",
            flavor="lattice"
        )
    except Exception as e:
        print(f"Camelot extraction failed: {e}")
        return []

    print(f"Total tables extracted by Camelot: {len(tables)}")
    return tables


# -----------------------------------------------------------
# 2. 표 → JSON Document 변환
# -----------------------------------------------------------
def convert_tables_to_documents(tables):
    """
    Camelot Table 객체 리스트 → JSON String → Document 변환
    빈 table(데이터 거의 없음)은 자동 필터링.
    """

    documents = []

    for idx, tbl in enumerate(tables):
        df = tbl.df

        # 🔥 빈 테이블 필터링
        if df is None or df.empty:
            print(f"Skipping empty table #{idx}")
            continue

        # 🔥 내용이 너무 짧거나 의미 없는 경우 스킵
        flat_text = " ".join(df.astype(str).values.flatten())
        if len(flat_text.strip()) < 10:
            print(f"Skipping meaningless table #{idx}")
            continue

        json_table = df.to_dict(orient="records")

        doc = Document(
            page_content=json.dumps(json_table, ensure_ascii=False, indent=2),
            metadata={
                "type": "table",
                "table_index": idx,
                "page": tbl.page
            }
        )

        documents.append(doc)

    print(f"Valid tables converted to Documents: {len(documents)}")
    return documents


# -----------------------------------------------------------
# 3. Chroma VectorStore 저장
# -----------------------------------------------------------
def save_table_vectorstore(docs, persist_dir="output/chroma/f1_tables"):
    print("Saving table vectorstore...")

    # 🔥 문서가 0개면 Chroma 생성하면 안됨
    if len(docs) == 0:
        print("⚠ No table docs found. Skipping table vectorstore creation.")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    print("✓ Table vectorstore created.")
    return vectorstore


# -----------------------------------------------------------
# 4. 전체 실행 파이프라인
# -----------------------------------------------------------
def build_table_vectorstore(pdf_path, persist_dir="output/chroma/f1_tables"):
    """
    1. PDF에서 테이블 추출
    2. Document 변환
    3. Chroma 저장
    """
    tables = extract_tables(pdf_path)
    docs = convert_tables_to_documents(tables)
    save_table_vectorstore(docs, persist_dir)