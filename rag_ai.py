import streamlit as st
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

from langchain_classic.chains import create_retrieval_chain, create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Axiom AI - Academic Tutor", page_icon="📚")
st.markdown("### Ask away. Get precise answers straight from the source text.")

with st.sidebar:
    st.header("⚙️ Controls")
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    st.divider()

# --- 2. LEAN CLOUD PIPELINE ---
@st.cache_resource
def setup_rag_pipeline():
    # Load API keys securely from Streamlit
    os.environ["PINECONE_API_KEY"] = st.secrets["PINECONE_API_KEY"]
    groq_api_key = st.secrets["GROQ_API_KEY"]

    # 1. Connect to Pinecone (Cloud Database)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name="rag-ai", 
        embedding=embeddings
    )

    # 2. Connect to Groq (Cloud LLM)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.0, groq_api_key=groq_api_key)
    
    # 3. Direct Pinecone Search (Retrieving top 3 sources to keep it fast)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 4. History Prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question... formulate a standalone question. Do NOT answer the question."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # 5. Answering Prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a brilliant, highly rigorous Computer Science Professor. You have been provided with excerpts from authoritative textbooks.\n\n<context>\n{context}\n</context>\n\nINSTRUCTIONS:\n1. Silently analyze the context.\n2. Write a comprehensive answer. Use direct quotes where appropriate.\nIf the answer is missing, do not guess."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain, llm

qa_chain, llm = setup_rag_pipeline()

# --- 3. STREAMLIT CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about your CS textbooks..."):
    st.chat_message("user").markdown(prompt)
    
    chat_history = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        else:
            chat_history.append(AIMessage(content=msg["content"]))

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        with st.spinner("Searching Pinecone and thinking..."):
            
            response = qa_chain.invoke({
                "input": prompt,
                "chat_history": chat_history
            })
            
            answer = response["answer"]
            sources = response["context"]
            
            st.markdown(answer)
            
            # Source Viewer with AI Cleanup
            with st.expander("View Sources"):
                for i, doc in enumerate(sources):
                    source_file = os.path.basename(doc.metadata.get('source', 'Unknown Book'))
                    page_num = doc.metadata.get('page', 0) 
                    display_page = page_num + 1
                    
                    st.markdown(f"**Source {i+1}: {source_file} (Page {display_page})**")
                    
                    raw_text = doc.page_content
                    last_period_idx = raw_text.rfind('.')
                    clean_text = raw_text[:last_period_idx + 1] if last_period_idx != -1 else raw_text 
                    
                    with st.expander("Read"):
                        with st.spinner("Groq is cleaning up this excerpt..."):
                            editor_prompt = f"""
                            You are an academic editor. Rewrite this raw textbook extraction so it flows 
                            as a single coherent passage. Fix broken sentences and remove weird PDF formatting. 
                            CRITICAL: Do NOT add outside knowledge. Only use the facts provided.
                            
                            RAW TEXT:
                            {clean_text}
                            """
                            refined_source = llm.invoke(editor_prompt)
                            st.write(refined_source.content)
                        
                    st.divider()

    st.session_state.messages.append({"role": "assistant", "content": answer})