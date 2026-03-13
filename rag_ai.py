import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="RAG_AI")

st.markdown("### Ask away. Get precise answers straight from the source text.")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.divider()

# --- 2. CLOUD RAG PIPELINE ---
@st.cache_resource
def setup_rag_pipeline():
    # 1. API Keys from Streamlit Secrets
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # 2. Connect directly to Pinecone Database
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name="axiom-db", 
        embedding=embeddings
    )

    # 3. Connect to Groq Cloud LLM
    llm = ChatGroq(model_name="llama3-8b-8192", temperature=0.0, groq_api_key=groq_api_key)

    # 4. Re-Ranking Setup
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 20})
    cross_encoder = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=cross_encoder, top_n=5)
    retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

    # 5. History-Aware Prompting
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a brilliant, highly rigorous Computer Science Professor. You have been provided with excerpts from authoritative textbooks.\n\n<context>\n{context}\n</context>\n\nINSTRUCTIONS:\n1. First, silently analyze the provided context.\n2. Write a comprehensive, multi-paragraph answer. Use direct quotes from the text where appropriate, and explain complex ideas simply.\n\nIf the answer is not contained in the context, do not guess. Simply state that the information is missing from the texts."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain

# Initialize the pipeline
qa_chain = setup_rag_pipeline()

# --- 3. STREAMLIT CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask me a computer science question..."):
    st.chat_message("user").markdown(prompt)
    
    # Format chat history
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    st.session_state.messages.append({"role": "user", "content": prompt})

    # Get bot response
    with st.chat_message("assistant"):
        with st.spinner("Searching cloud database..."):
            response = qa_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            
            answer = response["answer"]
            sources = response["context"]
            
            st.markdown(answer)
            
            # Cleaned up source viewer
            with st.expander("📚 View Text Sources"):
                for i, doc in enumerate(sources):
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown Book'))
                    page_num = doc.metadata.get('page', 0) + 1
                    
                    st.markdown(f"**Source {i+1}: {source_file} (Page {page_num})**")
                    st.info(doc.page_content) # Displays the exact extracted chunk
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})