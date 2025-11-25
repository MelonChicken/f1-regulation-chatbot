def format_output(body: str, citations: list):
    """
    UI-Friendly한 카드 형태로 출력.
    body : LLM이 생성한 답변 텍스트
    citations : ["문단 정보", "TABLE 정보"...]
    """

    # 본문 카드
    answer_block = f"📘 **답변**\n{body.strip()}"

    # 출처 카드
    if citations and len(citations) > 0:
        src_text = "\n".join(f"- {c}" for c in citations)
    else:
        src_text = "- (문서에 명시되지 않음 — F1 상식 기반)"

    src_block = f"\n\n📝 **참고 근거**\n{src_text}"

    return f"{answer_block}{src_block}"
