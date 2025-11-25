import json
from langchain_openai import ChatOpenAI
from retriever import retrieve_across_all

# ==========================================================
#  Translator (KOR ↔ ENG)
# ==========================================================
translator = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def translate_to_english(query):
    prompt = f"""
Translate this into FIA Sporting Regulations style English.
Do NOT simplify terms. Maintain technical vocabulary.

Query:
{query}
"""
    return translator.invoke(prompt).content.strip()

def translate_to_korean(text):
    prompt = f"""
아래 영문 내용을 FIA 기술/스포팅 규정 문체에 맞게 자연스러운 한국어로 번역하세요.
숫자, 단어, 용어는 원문을 정확하게 유지하세요.

텍스트:
{text}
"""
    return translator.invoke(prompt).content.strip()


# ==========================================================
#  중복 제거 로직 (store가 달라도 같은 chunk는 제거)
# ==========================================================
def dedupe_docs(docs):
    seen = set()
    unique = []
    for d in docs:
        sig = d.page_content.strip()[:200]   # 앞 200자 기준 signature
        if sig not in seen:
            seen.add(sig)
            unique.append(d)
    return unique


# ==========================================================
#  Relevance Scoring (Text only)
# ==========================================================
def relevance_score(doc, query):
    if doc.metadata.get("type") == "table":
        return 0
    text = doc.page_content.lower()
    q_words = query.lower().split()
    return sum(1 for w in q_words if w in text)


# ==========================================================
#  Table Parsing
# ==========================================================
def parse_table_json(doc):
    try:
        return json.loads(doc.page_content)
    except:
        return None


# ==========================================================
#  규정 문장 스타일러 (Streamlit-safe)
# ==========================================================
def style_regulation_sentence(text, citation):
    return f"""
<div style="color:#1a73e8; font-weight:500; margin-top:4px;">
{text}
</div>
<div style="color:#666; font-size:0.8em; margin-bottom:8px;">
📎 {citation}
</div>
"""


# ==========================================================
#  UI-Friendly Formatter
# ==========================================================
def format_output(main_answer, regulation_blocks):
    """
    regulation_blocks = [
        {"text": "...", "citation": "..."},
        ...
    ]
    """
    html = f"""
### 📘 답변
{main_answer}
<br/><br/>
"""

    if regulation_blocks:
        html += "### 📎 규정 인용<br/>"
        for blk in regulation_blocks:
            html += style_regulation_sentence(blk["text"], blk["citation"])

    return html


# ==========================================================
#                     MAIN RAG Q&A
# ==========================================================
def ask_question(query: str, k: int = 8):

    llm = ChatOpenAI(model="gpt-4o", temperature=0)

    # ------------------------------------------------------
    #  1) Query EN 변환
    # ------------------------------------------------------
    query_en = translate_to_english(query)

    # ------------------------------------------------------
    #  2) 한국어 + 영어 검색 → 결과 병합 후 중복 제거
    # ------------------------------------------------------
    docs_ko = retrieve_across_all(query, k=k)
    docs_en = retrieve_across_all(query_en, k=k)

    docs = dedupe_docs(docs_ko + docs_en)

    if not docs:
        return format_output("검색된 문서가 없습니다.", [])

    # ------------------------------------------------------
    #  3) 문서 분리
    # ------------------------------------------------------
    text_docs = [d for d in docs if d.metadata.get("type") != "table"]
    table_docs = [d for d in docs if d.metadata.get("type") == "table"]

    # relevance sorting
    text_docs = sorted(text_docs, key=lambda d: relevance_score(d, query_en), reverse=True)
    text_docs = text_docs[:3]
    table_docs = table_docs[:2]

    # ------------------------------------------------------
    #  4) Context 구성
    # ------------------------------------------------------
    context_blocks = []
    citation_raw = []

    for d in text_docs:
        context_blocks.append(d.page_content)
        citation_raw.append({
            "text": d.page_content[:300].replace("\n", " "),
            "citation": f"{d.metadata.get('source_store')} · p.{d.metadata.get('page')}"
        })

    for d in table_docs:
        table_data = parse_table_json(d)
        if table_data:
            context_blocks.append("TABLE_DATA:\n" + json.dumps(table_data, indent=2))
            citation_raw.append({
                "text": str(table_data),
                "citation": f"{d.metadata.get('source_store')} · p.{d.metadata.get('page')}"
            })

    context = "\n\n".join(context_blocks)

    # ------------------------------------------------------
    #  5) Overlap 판단
    # ------------------------------------------------------
    q_words = query_en.lower().split()
    overlap = sum(1 for w in q_words if w in context.lower())

    # ------------------------------------------------------
    #  6) Fallback — 문서 기반 내용 없음
    # ------------------------------------------------------
    if overlap == 0:
        prompt = f"""
You are an F1 expert. The question does not appear in the regulations.

Provide ONLY commonly-known F1 knowledge.
Do not invent article numbers or regulations.

Question:
{query}

Answer:
"""
        raw = llm.invoke(prompt).content.strip()
        answer_ko = translate_to_korean(raw)

        return format_output(answer_ko, [])

    # ------------------------------------------------------
    #  7) 문서 기반 RAG 답변
    # ------------------------------------------------------
    prompt = f"""
You are an FIA Sporting Regulations expert.

Use ONLY information appearing in Context.
If sentences are duplicated in Context, summarize them once.

[Context]
{context}

[Question]
{query}

[Answer]
"""
    raw_answer_en = llm.invoke(prompt).content.strip()
    answer_ko = translate_to_korean(raw_answer_en)

    # ------------------------------------------------------
    #  8) 규정 인용 (중복 제거)
    # ------------------------------------------------------
    seen = set()
    reg_blocks = []
    for c in citation_raw:
        key = c["text"][:150]
        if key not in seen:
            seen.add(key)
            reg_blocks.append(c)

    return format_output(answer_ko, reg_blocks)
