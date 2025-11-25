import os
import json
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from retriever import get_retriever, route_query
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
# Streamlit 기본 설정
# ----------------------------------------------------------
st.set_page_config(
    page_title="F1 Sporting Regulations Q&A",
    layout="wide"
)


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
        st_lottie(LOADING_ANIMATION, height=150)

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
# 메인 제목
# ----------------------------------------------------------
st.title("🏎️ F1 Sporting Regulations 규정 챗봇")


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

    if st.button("📦 data 폴더 문서로 벡터스토어 재생성"):
        # 전체 재생성에도 Lottie 사용
        loader_placeholder = st.empty()
        with loader_placeholder:
            st_lottie(LOADING_ANIMATION, height=150)

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

        st.success(f"저장됨 → {save_path}")

        if st.button("📄 업로드한 PDF만 벡터스토어 생성"):
            from processors.build_vectorstores import build_vectorstore_for_single_file

            loader_placeholder = st.empty()
            with loader_placeholder:
                st_lottie(LOADING_ANIMATION, height=150)

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
    st.header("💬 Chat")

    # 로딩 애니메이션을 넣을 자리(비어있는 컨테이너)
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

        # 2) 문서 검색
        retriever = get_retriever(k=5, query=user_query)
        docs = retriever.invoke(user_query)[:4]
        st.session_state["last_docs"] = docs

        # 3) Lottie 로딩 + RAG 답변 생성
        with loading_area:
            st_lottie(LOADING_ANIMATION, height=150)

        answer = ask_question(user_query, 12)
        # 로딩 제거
        loading_area.empty()

        # 4) Assistant 메시지 저장
        st.session_state.messages.append({"role": "assistant", "content": answer})

        # 입력창 초기화 후 rerun (입력창 비우기 & 최신 메시지 표시)
        st.session_state.pop("top_input", None)
        st.rerun()

    # ---------------------------
    # 메시지 출력 (최신순)
    # ---------------------------
    for msg in reversed(st.session_state["messages"]):
        with st.chat_message(msg["role"]):
            # rag_answer는 마크다운/텍스트 기반이므로 unsafe_allow_html는 상황에 따라 조정
            st.markdown(msg["content"], unsafe_allow_html=True)


# ==========================================================
# LEFT PANEL: 근거(Evidence) 패널
# ==========================================================
with left_panel:
    st.header("📘 답변에 사용된 규정 원문")

    if len(st.session_state["last_docs"]) == 0:
        st.info("아직 질문이 없습니다.")
    else:
        for i, d in enumerate(st.session_state["last_docs"]):
            st.markdown(f"### 📄 문단 {i+1}")

            # 텍스트 문서
            if d.metadata.get("type") != "table":
                st.markdown(f"- **Article**: {d.metadata.get('article')}")
                st.markdown(f"- **Section**: {d.metadata.get('section')}")
                st.text(d.page_content)

            # 표 문서
            else:
                st.markdown(f"- **표 인덱스**: {d.metadata.get('table_index')}")
                st.markdown(f"- **페이지**: {d.metadata.get('page')}")

                try:
                    df = pd.DataFrame(json.loads(d.page_content))
                    st.table(df)
                except Exception:
                    st.text(d.page_content)

            st.markdown("---")