import os
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_core.retrievers import BaseRetriever

# ---------------------------------------
# 0. Embeddings (전역 1개만 사용)
# ---------------------------------------
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# ---------------------------------------
# 1. VECTORSTORE 관리 (자동 로드)
# ---------------------------------------
VECTORSTORES = {}

def load_vectorstores():
    """
    output/chroma 폴더의 모든 vectorstore를 자동 로드하여
    VECTORSTORES dict에 저장.
    """
    base_dir = "output/chroma"

    if not os.path.exists(base_dir):
        print("No vectorstores directory found.")
        return

    for folder in os.listdir(base_dir):
        vs_path = os.path.join(base_dir, folder)

        if os.path.isdir(vs_path):
            try:
                vs = Chroma(
                    persist_directory=vs_path,
                    embedding_function=embeddings
                )
                VECTORSTORES[folder] = {
                    "name": folder,
                    "path": vs_path,
                    "vs": vs,
                }
                print(f"Loaded VectorStore: {folder}")
            except Exception as e:
                print(f"Failed to load VectorStore {folder}: {e}")


# 앱 실행 시 자동 로드
load_vectorstores()


# ---------------------------------------
# 2. Query Routing
# ---------------------------------------
def route_query(query: str) -> str:
    q = query.lower()
    table_keywords = [
        "table", "표", "points", "score", "alloc", "타이어", "할당", "schedule"
    ]
    if any(k in q for k in table_keywords):
        return "table"
    return "text"


# ---------------------------------------
# 3. 모든 VectorStore 검색 (Cross-store search)
# ---------------------------------------
def retrieve_across_all(query: str, k: int = 6, target_type: str = None):
    """
    모든 VectorStore에 대해 검색 결과를 합쳐서 반환.

    target_type = "table" 또는 "text" 또는 None
    """
    results = []

    for name, item in VECTORSTORES.items():
        # route 기반 필터링 수행
        if target_type == "table" and "tables" not in name:
            continue
        if target_type == "text" and "text" not in name:
            continue

        retriever = item["vs"].as_retriever(search_kwargs={"k": k})
        docs = retriever.invoke(query)
        results.extend(docs)

    return results


# ---------------------------------------
# 4. ListRetriever — 결과 리스트를 retriever처럼 래핑
# ---------------------------------------

class ClosureRetriever(BaseRetriever):

    def __init__(self, docs_fn):
        super().__init__()
        self._docs_fn = docs_fn  # 문서 제공 함수

    def _get_relevant_documents(self, query: str) -> List[Document]:
        return self._docs_fn(query)

    async def _aget_relevant_documents(self, query: str) -> List[Document]:
        return self._docs_fn(query)


# ---------------------------------------
# 5. get_retriever — 통합 retriever 생성
# ---------------------------------------
def get_retriever(k=12, query=""):
    route = route_query(query)

    # 1) route 기반으로 해당 스토어만 선택
    target = [
        item["vs"]
        for name, item in VECTORSTORES.items()
        if (route == "table" and "tables" in name)
        or (route == "text" and "text" in name)
    ]

    # fallback – 아무 것도 없으면 전체 사용
    if not target:
        target = [item["vs"] for _, item in VECTORSTORES.items()]

    # 2) 여러 vectorstore → 하나의 docs_fn 으로 감싸기
    def docs_fn(q):
        docs = []
        for vs in target:
            retr = vs.as_retriever(search_kwargs={"k": k})
            docs.extend(retr.invoke(q))
        return docs

    return ClosureRetriever(docs_fn)

