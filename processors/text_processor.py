import re
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma


# -----------------------------------------------------------
# 1. PDF 로드
# -----------------------------------------------------------
def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()


# -----------------------------------------------------------
# 2. ARTICLE split
# -----------------------------------------------------------
def split_by_article(pages):
    combined = "".join(p.page_content + "\n" for p in pages)

    pattern = r"(ARTICLE\s+B\d+(?::)?[^\n]*)"
    parts = re.split(pattern, combined)

    article_chunks = []

    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip()

        # 🔥 body가 충분히 길지 않으면 skip
        if len(body) < 15:
            continue

        article_chunks.append((title, body))

    return article_chunks


# -----------------------------------------------------------
# 3. SECTION split
# -----------------------------------------------------------
def split_into_sections(article_title, body_text):
    pattern = r"\b(B\d+(?:\.\d+)+)\b"
    parts = re.split(pattern, body_text)

    sections = []

    intro = parts[0].strip()
    if len(intro) > 5:
        sections.append(
            Document(
                page_content=intro,
                metadata={"article": article_title, "section": "intro"}
            )
        )

    for i in range(1, len(parts), 2):
        section_id = parts[i].strip()
        text = parts[i + 1].strip()

        # 🔥 내용이 너무 짧으면 skip
        if len(text) < 5:
            continue

        sections.append(
            Document(
                page_content=text,
                metadata={"article": article_title, "section": section_id}
            )
        )

    return sections


# -----------------------------------------------------------
# 4. 최적화 Chunking
# -----------------------------------------------------------

def chunk_optimize(sections, max_chars: int = 1000, overlap: int = 200):
    """
    Section 단위 chunking 전략:

    - 기본 단위는 Section 하나 (B1.7.3 전체를 하나로 유지)
    - Section 텍스트 길이가 max_chars 이하이면 그대로 한 개 chunk로 사용
    - 너무 긴 Section만 RecursiveCharacterTextSplitter로 나눔
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    optimized_chunks = []

    for sec in sections:
        text = (sec.page_content or "").strip()
        if not text:
            continue

        # 섹션 전체 길이가 짧으면 그냥 한 덩어리로 사용
        if len(text) <= max_chars:
            optimized_chunks.append(
                Document(
                    page_content=text,
                    metadata={
                        "article": sec.metadata.get("article"),
                        "section": sec.metadata.get("section"),
                        "subchunk_index": 0,
                    },
                )
            )
            continue

        # 너무 긴 섹션만 splitter로 재분할
        split_texts = splitter.split_text(text)

        for i, chunk in enumerate(split_texts):
            chunk = chunk.strip()
            if not chunk:
                continue

            optimized_chunks.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "article": sec.metadata.get("article"),
                        "section": sec.metadata.get("section"),
                        "subchunk_index": i,
                    },
                )
            )

    return optimized_chunks



# -----------------------------------------------------------
# 5. Chroma 저장
# -----------------------------------------------------------
def save_vectorstore(chunks, persist_dir):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    os.makedirs(persist_dir, exist_ok=True)

    # ⭐ 최종 필터링 (100% 보호)
    clean_chunks = [
        c for c in chunks
        if c.page_content and c.page_content.strip()
    ]

    if len(clean_chunks) == 0:
        raise ValueError(f"No valid chunks found to embed for {persist_dir}")

    return Chroma.from_documents(
        documents=clean_chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
def fallback_chunking(pages):
    """
    ARTICLE 패턴이 전혀 없는 규정 문서를 위한 fallback chunking
    - 전체 문서를 그대로 chunking
    - Technical Regulations, Appendix, Annex 등 처리 가능
    """
    raw_text = "".join(p.page_content + "\n" for p in pages)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = splitter.split_text(raw_text)

    documents = []
    for idx, c in enumerate(chunks):
        if c.strip():
            documents.append(
                Document(
                    page_content=c,
                    metadata={
                        "article": "unknown",
                        "section": "fallback",
                        "subchunk_index": idx
                    }
                )
            )
    return documents
