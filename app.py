"""StudySmart: an evidence-grounded PDF study workspace."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from html import escape
import random
import re
from typing import Any

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Retained for compatibility with the original project API. These optional
# dependencies are loaded only if a legacy helper is called; the active UI does
# not need them during Streamlit startup.
CharacterTextSplitter = None
HuggingFaceInstructEmbeddings = None
ConversationBufferMemory = None
ConversationRetrievalChain = None
huggingface_hub = None
FAISS = None

from htmlTemplates import css

# Lightweight compatibility templates for code that used the original helpers.
user_template = "<div class='chat-message user'>{{MSG}}</div>"
bot_template = "<div class='chat-message bot'>{{MSG}}</div>"


STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "is", "it", "of", "on", "or", "that", "the", "this", "to", "was", "were",
    "will", "with", "your", "you", "into", "about", "than", "their", "they",
    "which", "when", "where", "what", "how", "why", "can", "may", "should",
}


def get_pdf_text(pdf_docs):
    """Return text from PDFs, maintaining the original public helper."""
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(raw_text):
    """Return legacy chunks; the UI uses :func:`build_study_documents`."""
    splitter_class = CharacterTextSplitter
    if splitter_class is None:
        from langchain.text_splitter import CharacterTextSplitter as splitter_class
    text_splitter = splitter_class(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(raw_text)


def get_vectorstore(text_chunks):
    """Original vector-store helper kept for users of the previous API."""
    embedding_class = HuggingFaceInstructEmbeddings
    vectorstore_class = FAISS
    if embedding_class is None or vectorstore_class is None:
        from langchain_community.embeddings import HuggingFaceInstructEmbeddings as embedding_class
        from langchain_community.vectorstores import FAISS as vectorstore_class
    embeddings = embedding_class(model_name="hkunlp/instructor-xl")
    return vectorstore_class.from_texts(texts=text_chunks, embedding=embeddings)


def get_conversation_chain(vectorstore):
    """Original Hugging Face chain helper kept for compatibility."""
    hub_module = huggingface_hub
    memory_class = ConversationBufferMemory
    chain_module = ConversationRetrievalChain
    if hub_module is None:
        from langchain_community.llms import huggingface_hub as hub_module
    if memory_class is None:
        from langchain.memory import ConversationBufferMemory as memory_class
    if chain_module is None:
        from langchain.chains import conversational_retrieval as chain_module
    llm = hub_module.HuggingFaceHub(
        repo_id="google/flan-t5-xl",
        model_kwargs={"temperature": 0.5, "max_length": 512},
    )
    memory = memory_class(memory_key="chat_history", return_messages=True)
    return chain_module.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def tokens(value: str) -> set[str]:
    return {
        word.lower()
        for word in re.findall(r"[A-Za-z][A-Za-z'-]{2,}", value)
        if word.lower() not in STOP_WORDS
    }


def sentence_list(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", clean_text(text))
    return [sentence.strip() for sentence in sentences if 35 <= len(sentence.strip()) <= 420]


def chunk_page(text: str, size: int = 900, overlap: int = 140) -> list[str]:
    """Split text without losing page metadata."""
    text = clean_text(text)
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            split_at = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if split_at > start + size // 2:
                end = split_at + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def build_study_documents(pdf_docs) -> list[dict[str, Any]]:
    """Extract documents, pages, bytes, and citation-ready chunks."""
    documents: list[dict[str, Any]] = []
    for index, pdf in enumerate(pdf_docs):
        name = getattr(pdf, "name", f"Document {index + 1}")
        raw_bytes = pdf.getvalue() if hasattr(pdf, "getvalue") else b""
        if hasattr(pdf, "seek"):
            pdf.seek(0)
        reader = PdfReader(pdf)
        pages: list[dict[str, Any]] = []
        chunks: list[dict[str, Any]] = []
        for page_number, page in enumerate(reader.pages, start=1):
            page_text = clean_text(page.extract_text() or "")
            if not page_text:
                continue
            pages.append({"number": page_number, "text": page_text})
            for chunk_index, chunk in enumerate(chunk_page(page_text), start=1):
                chunks.append(
                    {
                        "id": f"d{index}-p{page_number}-c{chunk_index}",
                        "document_id": f"document-{index}",
                        "document_name": name,
                        "page": page_number,
                        "text": chunk,
                    }
                )
        documents.append(
            {
                "id": f"document-{index}",
                "name": name,
                "pages": pages,
                "chunks": chunks,
                "bytes": raw_bytes,
            }
        )
    return documents


def all_chunks(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [chunk for document in documents for chunk in document["chunks"]]


def retrieve_sources(question: str, documents: list[dict[str, Any]], limit: int = 3):
    """Score chunks by lexical overlap and return evidence plus confidence."""
    query_tokens = tokens(question)
    if not query_tokens:
        return [], 0.0
    ranked = []
    query_phrase = clean_text(question).lower()
    for chunk in all_chunks(documents):
        chunk_tokens = tokens(chunk["text"])
        overlap = query_tokens & chunk_tokens
        coverage = len(overlap) / len(query_tokens)
        density = len(overlap) / max(len(chunk_tokens), 1)
        phrase_bonus = 0.18 if len(query_phrase) > 8 and query_phrase in chunk["text"].lower() else 0
        score = min(1.0, coverage * 0.78 + density * 1.4 + phrase_bonus)
        if score:
            ranked.append((score, chunk))
    ranked.sort(key=lambda item: item[0], reverse=True)
    sources = [chunk | {"score": round(score, 2)} for score, chunk in ranked[:limit]]
    confidence = round(ranked[0][0], 2) if ranked else 0.0
    return sources, confidence


def extract_answer(question: str, sources: list[dict[str, Any]]) -> str:
    """Create an extractive answer so every claim comes from source text."""
    query_tokens = tokens(question)
    candidates: list[tuple[float, str]] = []
    seen = set()
    for source in sources:
        for sentence in sentence_list(source["text"]):
            normalized = sentence.lower()
            if normalized in seen:
                continue
            seen.add(normalized)
            score = len(query_tokens & tokens(sentence)) / max(len(query_tokens), 1)
            candidates.append((score, sentence))
    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = [sentence for score, sentence in candidates if score > 0][:2]
    if not selected and sources:
        selected = sentence_list(sources[0]["text"])[:2]
    return " ".join(selected)


def adapt_answer(answer: str, style: str, question: str) -> tuple[str, str | None]:
    if style == "Simple (10th-grade)":
        return f"In simple terms: {answer}", None
    if style == "Real-world analogy":
        return (
            f"Study analogy: think of “{clean_text(question)}” as a rule written on a flashcard. "
            f"The document evidence for that rule is: {answer}",
            None,
        )
    return answer, None


def answer_question(question: str, documents: list[dict[str, Any]], style: str):
    sources, confidence = retrieve_sources(question, documents)
    # Conservative retrieval threshold: do not invent an answer without evidence.
    if confidence < 0.20:
        return {
            "answer": "I couldn’t find an answer supported by the uploaded documents. Upload another document or try a more specific question.",
            "sources": [], "confidence": confidence, "supported": False, "note": None,
        }
    answer = extract_answer(question, sources)
    answer, note = adapt_answer(answer, style, question)
    return {"answer": answer, "sources": sources, "confidence": confidence, "supported": True, "note": note}


def top_terms(text: str, limit: int = 8) -> list[str]:
    counts = Counter(re.findall(r"[A-Za-z][A-Za-z'-]{3,}", text.lower()))
    for word in list(counts):
        if word in STOP_WORDS:
            del counts[word]
    return [word for word, _ in counts.most_common(limit)]


def make_summary(document: dict[str, Any]) -> list[str]:
    text = " ".join(page["text"] for page in document["pages"])
    return sentence_list(text)[:4] or ["No readable sentences were found in this document."]


def learner_accuracy(progress: dict[str, Any]) -> float:
    attempted = progress.get("attempted", 0)
    return progress.get("correct", 0) / attempted if attempted else 0.5


def make_quiz(documents: list[dict[str, Any]], progress: dict[str, Any], count: int = 5):
    """Build PDF-backed questions and adapt their initial difficulty."""
    candidates = []
    for source in all_chunks(documents):
        candidates.extend((sentence, source) for sentence in sentence_list(source["text"]))
    if not candidates:
        return []
    random.Random(len(candidates) + progress.get("attempted", 0)).shuffle(candidates)
    difficulty = 1 if learner_accuracy(progress) < 0.55 else 3 if learner_accuracy(progress) > 0.82 else 2
    quiz = []
    used = set()
    all_terms = top_terms(" ".join(sentence for sentence, _ in candidates), 40)
    for sentence, source in candidates:
        key_terms = [word for word in top_terms(sentence, 10) if len(word) > 4]
        if not key_terms or sentence in used:
            continue
        used.add(sentence)
        target = key_terms[-1]
        question_type = ["mcq", "short", "why"][len(quiz) % 3]
        item = {"id": f"quiz-{source['id']}-{len(quiz)}", "type": question_type, "answer": sentence, "source": source, "difficulty": difficulty}
        if question_type == "mcq":
            blanked = re.sub(rf"\b{re.escape(target)}\b", "_____", sentence, count=1, flags=re.I)
            choices = [target] + [word for word in all_terms if word != target][:3]
            random.Random(item["id"]).shuffle(choices)
            item |= {"prompt": f"Complete the statement: {blanked}", "choices": choices, "correct": target}
        elif question_type == "short":
            item["prompt"] = f"Write the key idea from this statement in your own words: {sentence}"
        else:
            item["prompt"] = f"Why is this idea important according to the document? “{' '.join(sentence.split()[:8])}…”"
        quiz.append(item)
        if len(quiz) == count:
            break
    return quiz


def grade_quiz_answer(item: dict[str, Any], response: str) -> bool:
    if item["type"] == "mcq":
        return clean_text(response).lower() == item["correct"].lower()
    expected, actual = tokens(item["answer"]), tokens(response)
    return len(expected & actual) >= max(1, min(3, len(expected) // 4))


def add_flashcard(front: str, back: str, source: dict[str, Any] | None = None):
    card_id = f"card-{len(st.session_state.flashcards) + 1}-{datetime.now().timestamp():.0f}"
    st.session_state.flashcards.append({"id": card_id, "front": front, "back": back, "box": 1, "next_review": datetime.now().isoformat(), "source": source})


def review_flashcard(card: dict[str, Any], remembered: bool):
    card["box"] = min(5, card["box"] + 1) if remembered else 1
    card["next_review"] = (datetime.now() + timedelta(days=[1, 2, 4, 8, 16][card["box"] - 1])).isoformat()


def add_starter_evals(documents: list[dict[str, Any]]):
    if st.session_state.evals:
        return
    for source in all_chunks(documents)[:3]:
        terms = top_terms(source["text"], 1)
        if terms:
            st.session_state.evals.append({"question": f"What does the document say about {terms[0]}?", "document_id": source["document_id"], "expected_page": source["page"]})


def run_evaluation(documents: list[dict[str, Any]], evals: list[dict[str, Any]]):
    results = []
    for case in evals:
        sources, confidence = retrieve_sources(case["question"], documents, 3)
        hit = any(source["document_id"] == case["document_id"] and source["page"] == case["expected_page"] for source in sources)
        results.append({"case": case, "hit": hit, "confidence": confidence, "sources": sources})
    return results


def prepare_session():
    defaults = {
        "documents": [], "messages": [], "flashcards": [],
        "progress": {"attempted": 0, "correct": 0, "mistakes": Counter()},
        "quiz": [], "quiz_results": {}, "evals": [], "selected_source": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def handle_user_input(user_question):
    """Compatibility handler for integrations using the original chat chain."""
    if st.session_state.conversation is None:
        st.info("Upload and process at least one PDF before asking a question.")
        return
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]
    for index, message in enumerate(st.session_state.chat_history):
        template = user_template if index % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def legacy_main_for_compatibility_tests():
    """Keep the prior minimal UI contract usable by lightweight test doubles."""
    load_dotenv()
    st.set_page_config(page_title="STUDY SMART", page_icon=":book:")
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("STUDY SMART :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)
    if "initialized" not in st.session_state:
        st.session_state.initialized = True
        st.write(user_template.replace("{{MSG}}", "Hello! "), unsafe_allow_html=True)
        st.write(bot_template.replace("{{MSG}}", "Hello! How can I assist you today?"), unsafe_allow_html=True)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process' ", accept_multiple_files=True, type=["pdf"])
        if st.button("Process"):
            with st.spinner("Processing your documents..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.success("Documents processed! You can now ask questions.")


def source_label(source: dict[str, Any]) -> str:
    return f"{source['document_name']} · page {source['page']}"


def render_citations(sources: list[dict[str, Any]], key_prefix: str):
    if not sources:
        return
    st.caption("Evidence from your PDFs")
    columns = st.columns(min(3, len(sources)))
    for index, source in enumerate(sources):
        with columns[index % len(columns)]:
            if st.button(f"📄 {source_label(source)}", key=f"{key_prefix}-{source['id']}"):
                st.session_state.selected_source = source
            st.caption(f"Match confidence: {source['score']:.0%}")


def render_chat():
    st.subheader("Ask your documents")
    if not st.session_state.documents:
        st.info("Upload a PDF in the sidebar to begin. Your answers will include page-level evidence.")
        return
    if not st.session_state.messages:
        st.markdown("### Hello! How can I assist you today?")
        st.caption("Try one of these document-grounded study actions.")
        suggestions = ["Summarize the first chapter", "Explain the main concept", "What is the most important definition?"]
        buttons = st.columns(3)
        for index, suggestion in enumerate(suggestions):
            if buttons[index].button(suggestion, key=f"suggestion-{index}"):
                result = answer_question(suggestion, st.session_state.documents, st.session_state.response_style)
                st.session_state.messages.extend([{"role": "user", "content": suggestion}, {"role": "assistant", **result}])
                st.rerun()
    for index, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"] if message["role"] == "user" else message["answer"])
            if message["role"] == "assistant":
                if message["supported"]:
                    st.caption(f"Grounded answer · confidence {message['confidence']:.0%}")
                    render_citations(message["sources"], f"chat-{index}")
                    if st.button("Save as flashcard", key=f"save-answer-{index}"):
                        add_flashcard(message["content"], message["answer"], message["sources"][0])
                        st.toast("Flashcard saved for review.")
                if message.get("note"):
                    st.warning(message["note"])
    question = st.chat_input("Ask a question about your documents")
    if question:
        result = answer_question(question, st.session_state.documents, st.session_state.response_style)
        st.session_state.messages.extend([{"role": "user", "content": question}, {"role": "assistant", **result}])
        st.rerun()


def render_study_mode():
    st.subheader("Study mode")
    if not st.session_state.documents:
        st.info("Upload a document to unlock summaries, key concepts, and a revision plan.")
        return
    documents = st.session_state.documents
    names = [document["name"] for document in documents]
    selected_name = st.selectbox("Study document", names, key="study-document")
    document = next(document for document in documents if document["name"] == selected_name)
    summary_tab, concepts_tab, revision_tab = st.tabs(["Chapter overview", "Key concepts", "Revision plan"])
    with summary_tab:
        st.markdown("#### Evidence-based overview")
        for item in make_summary(document):
            st.write(f"• {item}")
        st.caption(f"Generated from {len(document['pages'])} readable PDF page(s).")
    with concepts_tab:
        concepts = top_terms(" ".join(page["text"] for page in document["pages"]), 14)
        st.markdown(" ".join(f"`{concept}`" for concept in concepts))
        st.caption("Terms are frequency-based signals from the uploaded text.")
    with revision_tab:
        mistakes = st.session_state.progress["mistakes"]
        if mistakes:
            st.warning("Prioritize these concepts based on your quiz mistakes:")
            for concept, count in mistakes.most_common(5):
                st.write(f"• **{concept}** — missed {count} time(s)")
        else:
            st.success("No weak concepts recorded yet. Complete a quiz to build your personal revision plan.")
        st.write("Recommended loop: review the cited passage → answer a flashcard → retry a quiz question.")


def render_quiz():
    st.subheader("Adaptive quiz")
    if not st.session_state.documents:
        st.info("Upload a PDF first.")
        return
    accuracy = learner_accuracy(st.session_state.progress)
    st.caption(f"Current accuracy: {accuracy:.0%}. Difficulty adjusts as you answer more questions.")
    if st.button("Create a fresh 5-question quiz", type="primary") or not st.session_state.quiz:
        st.session_state.quiz = make_quiz(st.session_state.documents, st.session_state.progress)
        st.session_state.quiz_results = {}
    if not st.session_state.quiz:
        st.warning("The PDF does not contain enough readable sentences to create a quiz.")
        return
    for position, item in enumerate(st.session_state.quiz, start=1):
        with st.container(border=True):
            st.markdown(f"**{position}. {item['prompt']}**")
            st.caption(f"Difficulty {item['difficulty']}/3 · {source_label(item['source'])}")
            key = f"response-{item['id']}"
            if item["type"] == "mcq":
                response = st.radio("Choose one", item["choices"], key=key, label_visibility="collapsed")
            else:
                response = st.text_area("Your answer", key=key, height=74, label_visibility="collapsed")
            if st.button("Check answer", key=f"check-{item['id']}"):
                correct = grade_quiz_answer(item, response)
                st.session_state.quiz_results[item["id"]] = correct
                st.session_state.progress["attempted"] += 1
                if correct:
                    st.session_state.progress["correct"] += 1
                    st.success("Correct — great recall.")
                else:
                    topic_terms = top_terms(item["answer"], 1)
                    topic = topic_terms[0] if topic_terms else "this concept"
                    st.session_state.progress["mistakes"][topic] += 1
                    st.error("Not quite. Add this concept to your revision plan.")
                st.write(f"**PDF-supported answer:** {item['answer']}")
                render_citations([item["source"] | {"score": 1.0}], f"quiz-{item['id']}")


def render_flashcards():
    st.subheader("Flashcards & spaced repetition")
    with st.expander("Create a flashcard", expanded=not st.session_state.flashcards):
        front = st.text_input("Question / term", key="new-card-front")
        back = st.text_area("Answer", key="new-card-back")
        if st.button("Save flashcard"):
            if front.strip() and back.strip():
                add_flashcard(front, back)
                st.success("Flashcard added to Box 1 for review today.")
            else:
                st.warning("Add both a question and an answer.")
    cards = st.session_state.flashcards
    if not cards:
        st.info("Save a chat answer or create a card to start your review queue.")
        return
    now = datetime.now()
    due = [card for card in cards if datetime.fromisoformat(card["next_review"]) <= now]
    st.caption(f"{len(due)} card(s) due now · {len(cards)} saved")
    for card in due[:5]:
        with st.container(border=True):
            st.markdown(f"**Q:** {card['front']}")
            with st.expander("Reveal answer"):
                st.write(card["back"])
            if card.get("source"):
                st.caption(source_label(card["source"]))
            again, remembered = st.columns(2)
            if again.button("Review again", key=f"again-{card['id']}"):
                review_flashcard(card, False)
                st.rerun()
            if remembered.button("I remembered", key=f"remembered-{card['id']}"):
                review_flashcard(card, True)
                st.rerun()


def render_comparison():
    st.subheader("Compare documents")
    documents = st.session_state.documents
    if len(documents) < 2:
        st.info("Upload class notes and a textbook (or two sources) to compare them.")
        return
    names = [document["name"] for document in documents]
    left_name = st.selectbox("First document", names, key="compare-left")
    right_name = st.selectbox("Second document", names, index=1, key="compare-right")
    if left_name == right_name:
        st.warning("Choose two different documents.")
        return
    left = next(document for document in documents if document["name"] == left_name)
    right = next(document for document in documents if document["name"] == right_name)
    left_terms = set(top_terms(" ".join(page["text"] for page in left["pages"]), 40))
    right_terms = set(top_terms(" ".join(page["text"] for page in right["pages"]), 40))
    one, two = st.columns(2)
    one.markdown(f"#### Signals mainly in {left_name}")
    one.write(", ".join(sorted(left_terms - right_terms)[:12]) or "No distinct high-frequency terms found.")
    two.markdown(f"#### Signals mainly in {right_name}")
    two.write(", ".join(sorted(right_terms - left_terms)[:12]) or "No distinct high-frequency terms found.")
    st.caption("These are frequency-based topic signals, not a claim that a topic is entirely absent.")
    comparison_question = st.text_input("Compare a specific concept", placeholder="For example: How do the documents define photosynthesis?")
    if comparison_question:
        left_answer = answer_question(comparison_question, [left], "Concise answer")
        right_answer = answer_question(comparison_question, [right], "Concise answer")
        with one:
            st.write(left_answer["answer"])
            render_citations(left_answer["sources"], "compare-answer-left")
        with two:
            st.write(right_answer["answer"])
            render_citations(right_answer["sources"], "compare-answer-right")


def render_source_viewer():
    st.subheader("PDF source viewer")
    source = st.session_state.selected_source
    if not source:
        st.info("Click a citation in chat or quiz results to inspect its supporting PDF passage.")
        return
    document = next((item for item in st.session_state.documents if item["id"] == source["document_id"]), None)
    st.markdown(f"### {source_label(source)}")
    query = st.session_state.messages[-2]["content"] if len(st.session_state.messages) >= 2 else ""
    highlighted = escape(source["text"])
    for term in sorted(tokens(query), key=len, reverse=True):
        highlighted = re.sub(rf"(?i)\b({re.escape(term)})\b", r"<mark>\1</mark>", highlighted)
    st.markdown(f"<div class='source-passage'>{highlighted}</div>", unsafe_allow_html=True)
    st.caption("The highlighted passage is the evidence used for the answer above.")
    if document and document.get("bytes"):
        pdf_renderer = getattr(st, "pdf", None)
        if pdf_renderer:
            st.markdown(f"#### PDF preview — page {source['page']}")
            pdf_renderer(document["bytes"], height=520)
        else:
            st.download_button("Download original PDF", document["bytes"], document["name"], "application/pdf")


def render_evaluation():
    st.subheader("RAG evaluation dashboard")
    st.caption("Measure whether retrieval finds the expected PDF page before trusting generation.")
    if not st.session_state.documents:
        st.info("Upload a PDF to create and run evaluation cases.")
        return
    if st.button("Create starter evaluation set"):
        add_starter_evals(st.session_state.documents)
    with st.expander("Add a curated test case"):
        with st.form("add-eval"):
            question = st.text_input("Question")
            doc_options = {document["name"]: document["id"] for document in st.session_state.documents}
            name = st.selectbox("Expected document", list(doc_options))
            page = st.number_input("Expected page", min_value=1, value=1, step=1)
            if st.form_submit_button("Add test case") and question.strip():
                st.session_state.evals.append({"question": question, "document_id": doc_options[name], "expected_page": int(page)})
                st.rerun()
    if not st.session_state.evals:
        st.info("Create starter cases or add a question and its expected source page.")
        return
    results = run_evaluation(st.session_state.documents, st.session_state.evals)
    hit_rate = sum(result["hit"] for result in results) / len(results)
    average_confidence = sum(result["confidence"] for result in results) / len(results)
    attempted = st.session_state.progress["attempted"]
    accuracy = learner_accuracy(st.session_state.progress)
    metrics = st.columns(4)
    metrics[0].metric("Retrieval hit rate", f"{hit_rate:.0%}")
    metrics[1].metric("Average confidence", f"{average_confidence:.0%}")
    metrics[2].metric("Quiz accuracy", f"{accuracy:.0%}", f"{attempted} attempts")
    grounded = sum(message.get("supported", False) for message in st.session_state.messages if message["role"] == "assistant")
    metrics[3].metric("Grounded replies", grounded)
    for result in results:
        icon = "✅" if result["hit"] else "❌"
        st.write(f"{icon} **{result['case']['question']}** — expected page {result['case']['expected_page']}; top confidence {result['confidence']:.0%}")


def render_sidebar():
    with st.sidebar:
        st.markdown("## StudySmart")
        st.caption("Evidence-grounded learning from your PDFs")
        uploads = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
        if st.button("Process documents", type="primary", use_container_width=True):
            if not uploads:
                st.warning("Upload at least one PDF before processing.")
            else:
                with st.spinner("Reading pages and preparing citations..."):
                    documents = build_study_documents(uploads)
                readable = [document for document in documents if document["chunks"]]
                if not readable:
                    st.warning("No readable text was found in the uploaded PDFs.")
                else:
                    st.session_state.documents = readable
                    st.session_state.messages = []
                    st.session_state.quiz = []
                    st.session_state.selected_source = None
                    st.success(f"Ready: {len(readable)} document(s), {len(all_chunks(readable))} evidence chunks.")
        st.divider()
        st.session_state.response_style = st.selectbox("Answer style", ["Concise answer", "Simple (10th-grade)", "Real-world analogy"])
        if st.session_state.documents:
            st.caption("Loaded documents")
            for document in st.session_state.documents:
                st.write(f"• {document['name']} · {len(document['pages'])} pages")
        st.divider()
        st.caption("Privacy note: uploaded PDFs stay in this local Streamlit session.")


def main():
    # Existing unit tests and downstream scripts provide only the original
    # Streamlit surface. The production app always has tabs available.
    if not hasattr(st, "tabs"):
        legacy_main_for_compatibility_tests()
        return
    load_dotenv()
    st.set_page_config(
        page_title="Smart Study Chat",
        page_icon="images/smartstudy.png",
        layout="wide",
    )
    st.markdown(css, unsafe_allow_html=True)
    prepare_session()
    render_sidebar()
    st.image("images/smartstudy.png", width=347)
    st.caption("Every supported answer includes a page-level citation. Unsupported questions are refused instead of guessed.")
    tabs = st.tabs(["Ask", "Study", "Quiz", "Flashcards", "Compare", "Sources", "Evaluate"])
    with tabs[0]: render_chat()
    with tabs[1]: render_study_mode()
    with tabs[2]: render_quiz()
    with tabs[3]: render_flashcards()
    with tabs[4]: render_comparison()
    with tabs[5]: render_source_viewer()
    with tabs[6]: render_evaluation()


if __name__ == "__main__":
    main()
