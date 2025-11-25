import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from retriever import get_retriever
from rag_answer import ask_question
from processors.build_vectorstores import build_all_vectorstores_from_data
from streamlit_lottie import st_lottie


# ----------------------------------------------------------
# Lottie 파일 로드
# ----------------------------------------------------------
def load_lottie_from_file(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


LOADING_ANIMATION = load_lottie_from_file("assets/loading.json")


# ----------------------------------------------------------
# Streamlit 기본 설정 + F1 스타일 전역 CSS
# ----------------------------------------------------------
st.set_page_config(
    page_title="F1 Sporting Regulations Q&A",
    layout="wide"
)

# F1 다크 테마 + 채팅 버블 + Evidence 카드 스타일
st.markdown("""
<style>
/* 전체 배경 & 글꼴 */
[data-testid="stAppViewContainer"] {
    background-color: #FFDFB9;
    color: #3B0714;
}

/* 사이드바 스타일 */
[data-testid="stSidebar"] {
    background-color: #F4C9A1;
}

/* 기본 제목 스타일 (F1 레드) */
h1 {
    color: #A4193D;
    font-weight: 800;
    letter-spacing: -1px;
}

/* 섹션 헤더 */
h2, h3 {
    color: #A4193D;
}

/* 버튼 스타일 */
.stButton>button {
    background: linear-gradient(90deg, #C82452, #7C132E);
    color: white;
    border-radius: 999px;
    border: none;
    padding: 0.35rem 1.2rem;
    font-weight: 600;
}
.stButton>button:hover {
    background: linear-gradient(90deg, #D8345F, #8F1737);
    color: white;
}

/* 텍스트 입력창 라벨 */
label {
    color: #E2E2E2 !important;
}

/* 채팅 버블 스타일 */
[data-testid="stChatMessage"] {
    margin-bottom: 0.4rem;
}
[data-testid="stChatMessage"] div[data-testid="stMarkdown"] {
    border-radius: 12px;
    padding: 0.6rem 0.8rem;
    background-color: #C7D3D4;
}
[data-testid="stChatMessage"][data-testid="stChatMessage-user"] div[data-testid="stMarkdown"] {
    border-left: 3px solid #A4193D;
}
[data-testid="stChatMessage"][data-testid="stChatMessage-assistant"] div[data-testid="stMarkdown"] {
    border-left: 3px solid #F4A8C0;
}

/* Evidence 카드 */
.evidence-card {
    background-color: #C7D3D4;
    padding: 10px 12px;
    border-radius: 8px;
    margin-bottom: 10px;
    border-left: 3px solid #A4193D;
}
.evidence-title {
    color: #A4193D;
    font-weight: 600;
    margin-bottom: 4px;
}
.evidence-meta {
    color: #C4C4C4;
    font-size: 0.85em;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# 상단 F1 배너 + 타이틀
# ----------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(90deg, #C82452, #7C132E);
    padding: 10px 18px;
    border-radius: 8px;
    margin-bottom: 16px;
    display: flex;
    align-items: center;
    justify-content: space-between;
">
    <div style="color:white; font-weight:700; font-size:20px;">
        🏁 F1 Sporting Regulations Expert Chatbot
    </div>
    <div style="color:#FFD7D1; font-size:12px;">
        FIA Sporting & Technical Docs · RAG 기반 질의응답
    </div>
</div>
""", unsafe_allow_html=True)


# ----------------------------------------------------------
# 초기 Vectorstore 체크
# ----------------------------------------------------------
def initialize_vectorstores():
    data_dir = "data"
    pdf_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        st.warning("⚠ data 폴더에 PDF가 없습니다. Sidebar에서 업로드해주세요.")
        return

    # 이미 chroma 폴더가 있으면 그대로 사용
    if os.path.exists("output/chroma"):
        st.info("✔ 기존 vectorstore가 감지되었습니다. 바로 질문 가능합니다.")
        return

    st.info("📚 처음 실행: data 폴더의 문서로 벡터스토어 생성 중...")

    # Lottie 로더 표시
    loader_placeholder = st.empty()
    with loader_placeholder:
        st_lottie(LOADING_ANIMATION, height=140, key="init-lottie")

    # 실제 벡터스토어 생성
    build_all_vectorstores_from_data()

    # 로딩 애니메이션 제거
    loader_placeholder.empty()

    st.success("🎉 벡터스토어 생성 완료!")


initialize_vectorstores()


# ----------------------------------------------------------
# 세션 상태 초기화
# ----------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []  # Chat history

if "last_docs" not in st.session_state:
    st.session_state["last_docs"] = []  # Evidence 패널


# ----------------------------------------------------------
# SIDEBAR - 문서 관리
# ----------------------------------------------------------
with st.sidebar:
    st.header("📄 문서 관리 & 벡터스토어 생성")

    data_dir = "data"
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    st.subheader("📚 현재 등록된 PDF 문서")
    if len(pdf_files) == 0:
        st.info("data 폴더에 PDF 문서가 없습니다.")
    else:
        for f in pdf_files:
            st.markdown(f"- **{f}**")

    st.markdown("---")
    st.subheader("🔄 전체 문서 재처리")

    if st.button("📦 data 폴더 문서로 벡터스토어 재생성", use_container_width=True):
        loader_placeholder = st.empty()
        with loader_placeholder:
            st_lottie(LOADING_ANIMATION, height=140, key="rebuild-all")

        build_all_vectorstores_from_data()
        loader_placeholder.empty()

        st.success("🎉 전체 벡터스토어 재생성 완료!")

    st.markdown("---")
    st.subheader("📤 PDF 업로드")

    uploaded_pdf = st.file_uploader("규정 PDF 업로드", type=["pdf"])
    if uploaded_pdf is not None:
        save_path = os.path.join("data", uploaded_pdf.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_pdf.getbuffer())

        st.success(f"저장됨 → `{save_path}`")

        if st.button("📄 업로드한 PDF만 벡터스토어 생성", use_container_width=True):
            from processors.build_vectorstores import build_vectorstore_for_single_file

            loader_placeholder = st.empty()
            with loader_placeholder:
                st_lottie(LOADING_ANIMATION, height=140, key="rebuild-single")

            build_vectorstore_for_single_file(save_path)
            loader_placeholder.empty()

            st.success("🎉 업로드한 문서 기반 벡터스토어 생성 완료!")


# ----------------------------------------------------------
# 두 개의 레이아웃 (좌: Evidence / 우: Chat)
# ----------------------------------------------------------
left_panel, right_panel = st.columns([1.0, 2.0])


# ==========================================================
# RIGHT PANEL: CHAT UI
# ==========================================================
with right_panel:
    st.header("💬 질의응답")

    # 로딩 애니메이션 placeholder
    loading_area = st.empty()

    # ---------------------------
    # 입력창 (항상 상단 고정)
    # ---------------------------
    user_query = st.text_input(
        "질문을 입력하세요...",
        key="top_input",
        placeholder="예: 피트 레인 속도 제한은 얼마인가요?"
    )

    # 전송 버튼
    send = st.button("전송", use_container_width=True)

    # ---------------------------
    # 질문 처리
    # ---------------------------
    if send and user_query.strip():
        # 1) 사용자 메시지 추가
        st.session_state["messages"].append(
            {"role": "user", "content": user_query}
        )

        # 2) 문서 검색 (Evidence Panel용)
        retriever = get_retriever(k=5, query=user_query)
        docs = retriever.invoke(user_query)[:4]
        st.session_state["last_docs"] = docs

        # 3) Lottie 로딩 + RAG 답변 생성
        with loading_area:
            st_lottie(LOADING_ANIMATION, height=120, key="qa-loading")

        answer = ask_question(user_query, 12)

        # 로딩 제거
        loading_area.empty()

        # 4) Assistant 메시지 저장
        st.session_state["messages"].append(
            {"role": "assistant", "content": answer}
        )

        # 입력창 초기화 후 rerun → top_input 값 리셋
        st.session_state.pop("top_input", None)
        st.rerun()

    # ---------------------------
    # 메시지 출력 (최신순)
    # ---------------------------
    for msg in reversed(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            # rag_answer는 마크다운 기반 출력이므로 unsafe_allow_html는 상황에 따라 조정 가능
            st.markdown(msg["content"], unsafe_allow_html=True)


# ==========================================================
# LEFT PANEL: 근거(Evidence) 패널
# ==========================================================
with left_panel:
    st.header("📘 답변에 사용된 규정 원문")

    if len(st.session_state["last_docs"]) == 0:
        st.info("아직 질문이 없습니다. 질문을 입력하면 관련된 규정 원문이 여기에 표시됩니다.")
    else:
        for i, d in enumerate(st.session_state["last_docs"]):
            st.markdown(
                f"<div class='evidence-card'>"
                f"<div class='evidence-title'>📄 문단 {i+1}</div>",
                unsafe_allow_html=True
            )

            # 텍스트 문서
            if d.metadata.get("type") != "table":
                meta_html = (
                    f"<div class='evidence-meta'>"
                    f"Article: <b>{d.metadata.get('article')}</b> · "
                    f"Section: <b>{d.metadata.get('section')}</b>"
                    f"</div>"
                )
                st.markdown(meta_html, unsafe_allow_html=True)
                st.text(d.page_content)

            # 표 문서
            else:
                meta_html = (
                    f"<div class='evidence-meta'>"
                    f"Table Index: <b>{d.metadata.get('table_index')}</b> · "
                    f"Page: <b>{d.metadata.get('page')}</b>"
                    f"</div>"
                )
                st.markdown(meta_html, unsafe_allow_html=True)

                try:
                    df = pd.DataFrame(json.loads(d.page_content))
                    st.table(df)
                except Exception:
                    st.text(d.page_content)

            st.markdown("</div>", unsafe_allow_html=True)
