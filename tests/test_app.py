"""Unit tests for the Smart Study application.

The production dependencies are intentionally replaced with small test doubles
before importing ``app``.  This keeps the tests fast and avoids calling model
providers, loading embedding models, or requiring a Streamlit runtime.
"""

from __future__ import annotations

import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


class SessionState(dict):
    """Dictionary that supports Streamlit's attribute-style session access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class ContextManager:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def install_dependency_stubs():
    """Install the minimal import surface required by app.py."""
    streamlit = types.ModuleType("streamlit")
    streamlit.session_state = SessionState()
    streamlit.set_page_config = MagicMock()
    streamlit.write = MagicMock()
    streamlit.header = MagicMock()
    streamlit.text_input = MagicMock(return_value="")
    streamlit.subheader = MagicMock()
    streamlit.file_uploader = MagicMock(return_value=[])
    streamlit.button = MagicMock(return_value=False)
    streamlit.spinner = MagicMock(side_effect=lambda *_: ContextManager())
    streamlit.success = MagicMock()
    streamlit.sidebar = ContextManager()

    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = MagicMock()

    pypdf2 = types.ModuleType("PyPDF2")
    pypdf2.PdfReader = MagicMock()

    langchain = types.ModuleType("langchain")
    text_splitter = types.ModuleType("langchain.text_splitter")
    text_splitter.CharacterTextSplitter = MagicMock()
    memory = types.ModuleType("langchain.memory")
    memory.ConversationBufferMemory = MagicMock()
    chains = types.ModuleType("langchain.chains")
    conversational_retrieval = types.ModuleType("langchain.chains.conversational_retrieval")
    conversational_retrieval.from_llm = MagicMock()
    chains.conversational_retrieval = conversational_retrieval

    community = types.ModuleType("langchain_community")
    embeddings = types.ModuleType("langchain_community.embeddings")
    embeddings.HuggingFaceInstructEmbeddings = MagicMock()
    embeddings_openai = types.ModuleType("langchain_community.embeddings.openai")
    embeddings_openai.OpenAIEmbeddings = MagicMock()
    chat_models = types.ModuleType("langchain_community.chat_models")
    chat_models.ChatOpenAI = MagicMock()
    llms = types.ModuleType("langchain_community.llms")
    huggingface_hub = types.ModuleType("langchain_community.llms.huggingface_hub")
    huggingface_hub.HuggingFaceHub = MagicMock()
    llms.huggingface_hub = huggingface_hub
    vectorstores = types.ModuleType("langchain_community.vectorstores")
    vectorstores.FAISS = MagicMock()

    sys.modules.update(
        {
            "streamlit": streamlit,
            "dotenv": dotenv,
            "PyPDF2": pypdf2,
            "langchain": langchain,
            "langchain.text_splitter": text_splitter,
            "langchain.memory": memory,
            "langchain.chains": chains,
            "langchain.chains.conversational_retrieval": conversational_retrieval,
            "langchain_community": community,
            "langchain_community.embeddings": embeddings,
            "langchain_community.embeddings.openai": embeddings_openai,
            "langchain_community.chat_models": chat_models,
            "langchain_community.llms": llms,
            "langchain_community.llms.huggingface_hub": huggingface_hub,
            "langchain_community.vectorstores": vectorstores,
        }
    )
    return streamlit


ST = install_dependency_stubs()
sys.modules.pop("app", None)
app = importlib.import_module("app")


class AppTests(unittest.TestCase):
    def setUp(self):
        ST.session_state = SessionState()
        for method in (
            ST.set_page_config,
            ST.write,
            ST.header,
            ST.text_input,
            ST.subheader,
            ST.file_uploader,
            ST.button,
            ST.spinner,
            ST.success,
        ):
            method.reset_mock()
        ST.text_input.return_value = ""
        ST.file_uploader.return_value = []
        ST.button.return_value = False
        ST.spinner.side_effect = lambda *_: ContextManager()

    def test_get_pdf_text_concatenates_nonempty_pages_from_each_document(self):
        first_document = types.SimpleNamespace(
            pages=[
                types.SimpleNamespace(extract_text=lambda: "First "),
                types.SimpleNamespace(extract_text=lambda: None),
            ]
        )
        second_document = types.SimpleNamespace(
            pages=[types.SimpleNamespace(extract_text=lambda: "second")]
        )

        with patch.object(app, "PdfReader", side_effect=[first_document, second_document]) as reader:
            text = app.get_pdf_text(["one.pdf", "two.pdf"])

        self.assertEqual(text, "First second")
        self.assertEqual(reader.call_args_list[0].args, ("one.pdf",))
        self.assertEqual(reader.call_args_list[1].args, ("two.pdf",))

    def test_get_text_chunks_uses_the_configured_splitter(self):
        splitter = MagicMock()
        splitter.split_text.return_value = ["chunk one", "chunk two"]

        with patch.object(app, "CharacterTextSplitter", return_value=splitter) as splitter_class:
            chunks = app.get_text_chunks("source text")

        self.assertEqual(chunks, ["chunk one", "chunk two"])
        splitter_class.assert_called_once_with(
            separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
        )
        splitter.split_text.assert_called_once_with("source text")

    def test_get_vectorstore_builds_faiss_index_with_instructor_embeddings(self):
        embeddings = object()
        vectorstore = object()

        with (
            patch.object(app, "HuggingFaceInstructEmbeddings", return_value=embeddings) as embedding_class,
            patch.object(app.FAISS, "from_texts", return_value=vectorstore) as from_texts,
        ):
            result = app.get_vectorstore(["one", "two"])

        self.assertIs(result, vectorstore)
        embedding_class.assert_called_once_with(model_name="hkunlp/instructor-xl")
        from_texts.assert_called_once_with(texts=["one", "two"], embedding=embeddings)

    def test_get_conversation_chain_connects_model_retriever_and_memory(self):
        vectorstore = MagicMock()
        retriever = object()
        vectorstore.as_retriever.return_value = retriever
        llm = object()
        memory = object()
        chain = object()

        with (
            patch.object(app.huggingface_hub, "HuggingFaceHub", return_value=llm) as hub,
            patch.object(app, "ConversationBufferMemory", return_value=memory) as memory_class,
            patch.object(app.ConversationRetrievalChain, "from_llm", return_value=chain) as from_llm,
        ):
            result = app.get_conversation_chain(vectorstore)

        self.assertIs(result, chain)
        hub.assert_called_once_with(
            repo_id="google/flan-t5-xl", model_kwargs={"temperature": 0.5, "max_length": 512}
        )
        memory_class.assert_called_once_with(memory_key="chat_history", return_messages=True)
        from_llm.assert_called_once_with(llm=llm, retriever=retriever, memory=memory)

    def test_handle_user_input_stores_history_and_renders_each_role(self):
        history = [
            types.SimpleNamespace(content="Question"),
            types.SimpleNamespace(content="Answer"),
        ]
        conversation = MagicMock(return_value={"chat_history": history})
        ST.session_state.conversation = conversation

        app.handle_user_input("What is this?")

        conversation.assert_called_once_with({"question": "What is this?"})
        self.assertIs(ST.session_state.chat_history, history)
        self.assertEqual(ST.write.call_count, 2)
        self.assertIn("Question", ST.write.call_args_list[0].args[0])
        self.assertIn("Answer", ST.write.call_args_list[1].args[0])
        self.assertIn("user", ST.write.call_args_list[0].args[0])
        self.assertIn("bot", ST.write.call_args_list[1].args[0])

    def test_main_initializes_session_and_displays_welcome_messages(self):
        app.main()

        app.load_dotenv.assert_called_once_with()
        ST.set_page_config.assert_called_once_with(page_title="STUDY SMART", page_icon=":book:")
        self.assertIsNone(ST.session_state.conversation)
        self.assertIsNone(ST.session_state.chat_history)
        self.assertTrue(ST.session_state.initialized)
        self.assertEqual(ST.header.call_args.args, ("STUDY SMART :books:",))
        self.assertEqual(ST.write.call_count, 3)

    def test_main_routes_a_question_to_the_conversation_handler(self):
        ST.text_input.return_value = "Explain this PDF"

        with patch.object(app, "handle_user_input") as handler:
            app.main()

        handler.assert_called_once_with("Explain this PDF")

    def test_main_processes_uploaded_documents_and_saves_chain(self):
        documents = ["notes.pdf"]
        ST.file_uploader.return_value = documents
        ST.button.return_value = True
        chain = object()

        with (
            patch.object(app, "get_pdf_text", return_value="document text") as get_text,
            patch.object(app, "get_text_chunks", return_value=["chunk"]) as get_chunks,
            patch.object(app, "get_vectorstore", return_value="vectorstore") as get_store,
            patch.object(app, "get_conversation_chain", return_value=chain) as get_chain,
        ):
            app.main()

        get_text.assert_called_once_with(documents)
        get_chunks.assert_called_once_with("document text")
        get_store.assert_called_once_with(["chunk"])
        get_chain.assert_called_once_with("vectorstore")
        self.assertIs(ST.session_state.conversation, chain)
        ST.success.assert_called_once_with("Documents processed! You can now ask questions.")


if __name__ == "__main__":
    unittest.main()
