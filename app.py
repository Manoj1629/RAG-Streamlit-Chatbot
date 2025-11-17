.\.venv\Scripts\activate
import streamlit as st
from pathlib import Path
import os, logging, tempfile
from models.embeddings import EmbeddingModel
from models.llm import LLM
from utils.rag_utils import build_index_from_documents, retrieve_relevant
from utils.web_search import serpapi_search, duckduckgo_search
from config.config import EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title='RAG Chatbot', layout='wide')

st.title('RAG-enabled Streamlit Chatbot (Demo)')
st.sidebar.header('Settings')

mode = st.sidebar.selectbox('Response mode', ['Concise', 'Detailed'])
use_web = st.sidebar.checkbox('Enable web search (when needed)', value=True)
uploaded = st.file_uploader('Upload plain text documents (.txt) — multiple allowed', type=['txt'], accept_multiple_files=True)

# Initialize components (lazily)
if 'embedder' not in st.session_state:
    try:
        st.session_state.embedder = EmbeddingModel(EMBEDDING_MODEL)
    except Exception as e:
        st.error(f'Failed to load embedding model: {e} — install sentence-transformers and download model.')
        st.stop()

if 'index' not in st.session_state:
    st.session_state.index = None

if uploaded:
    docs = []
    for f in uploaded:
        try:
            t = f.read().decode('utf-8')
        except:
            try:
                t = f.getvalue().decode('utf-8')
            except:
                t = ''
        if t:
            docs.append(t)
    if docs:
        with st.spinner('Building index...'):
            try:
                st.session_state.index = build_index_from_documents(docs, st.session_state.embedder)
                st.success('Index built with uploaded documents.')
            except Exception as e:
                st.error(f'Failed building index: {e}')

query = st.text_area('Ask a question (the bot will look up uploaded docs + web if enabled):', height=120)

if st.button('Get Answer'):
    if not query.strip():
        st.warning('Provide a query.')
    else:
        results = []
        if st.session_state.index is not None:
            try:
                results = retrieve_relevant(query, st.session_state.index, st.session_state.embedder, k=4)
            except Exception as e:
                st.error(f'Error retrieving from index: {e}')

        # If not enough local context, optionally perform web search
        if use_web and (not results or len(results) < 2):
            try:
                web_results = serpapi_search(query, num=3)
            except Exception:
                web_results = duckduckgo_search(query, num=3)
            # append web snippets into context
            for wr in web_results:
                txt = f"{wr.get('title','')} - {wr.get('snippet','')} {wr.get('link','')}"
                results.append((txt, 0.0))

        # Prepare system and user prompt
        system_prompt = 'You are a helpful assistant. Use provided context to answer the user.'
        context_text = '\n\n'.join([r[0] for r in results]) if results else 'No contextual documents found.'
        if mode == 'Concise':
            user_prompt = f"Answer concisely. Context:\n{context_text}\n\nQuestion: {query}\nPlease keep the answer short (1-3 sentences)."
        else:
            user_prompt = f"Answer in detail. Context:\n{context_text}\n\nQuestion: {query}\nProvide an expanded, detailed answer and cite any context you used."

        # Call LLM
        llm = LLM()
        answer = llm.chat(system_prompt, user_prompt, temperature=0.0, max_tokens=512)
        st.markdown('**Answer:**')
        st.write(answer)
        if results:
            st.markdown('**Sources / Retrieved context (top results):**')
            for i, r in enumerate(results[:6]):
                st.write(f"{i+1}. {r[0][:800]}{'...' if len(r[0])>800 else ''}")
