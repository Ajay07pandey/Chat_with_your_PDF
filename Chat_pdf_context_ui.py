import streamlit as st
import os
import tempfile
import datetime
from dotenv import load_dotenv
import logging

# --- Core LangChain components ---
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Setup logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file (optional)
load_dotenv()

# --- Functions ---

def get_vectorstore_from_file(file_path, api_key):
    """
    Loads a file, splits it into chunks, creates embeddings, and stores them in a FAISS vector store.
    This function can handle PDFs, text files, etc.
    
    Args:
        file_path (str): The file path to the document.
        api_key (str): The Google API key.

    Returns:
        FAISS: The vector store object containing the document chunks.
    """
    if not os.path.exists(file_path):
        logger.error(f"The file {file_path} was not found.")
        raise FileNotFoundError(f"The file {file_path} was not found.")

    # Initialize embeddings model
    model_path = os.getenv("MODEL_PATH", "models/embedding-001")
    embeddings = GoogleGenerativeAIEmbeddings(model=model_path, google_api_key=api_key)

    # --- 2. Load the Document ---
    ext = file_path.split('.')[-1].lower()
    if ext == 'pdf':
        loader = PyPDFLoader(file_path)
    elif ext == 'txt':
        loader = TextLoader(file_path)
    else:
        raise ValueError("Unsupported document type")

    documents = loader.load()

    # --- 3. Split the Document into Chunks ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    document_chunks = text_splitter.split_documents(documents)
    
    # --- 4. Create and Persist the Vector Store ---
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    
    return vector_store


def get_conversational_rag_chain(vector_store, api_key):
    """
    Creates a retrieval-augmented generation (RAG) chain for general conversation.

    Args:
        vector_store (FAISS): The vector store.
        api_key (str): The Google API key.

    Returns:
        Runnable: A LangChain runnable object for the RAG chain.
    """
    # Initialize the LLM
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.6, google_api_key=api_key)

    # Models = Gemini 2.0 Flash-Lite, gemini-1.5-flash
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever()

    # Define the prompt to make it general for various documents
    prompt = ChatPromptTemplate.from_template(""" 
    You are a highly knowledgeable assistant that answers questions based on the provided document.
    Your task is to respond to the question using the context from the document.
    If you are not able to answer the queries then give this link and say for more info visit this Masterfile.
    Link - https://docs.google.com/spreadsheets/d/1LMKSRrAnaE6Jz6y3EqCO-jF-hIJX9s0u_vgVdLNjsnI/edit?gid=193279165#gid=193279165
    Always Try to answer point wise and in detail also in structured and well maintained format if in a process steps are there
    so they should be in new line every step.
                                                                                                                                   
    Context:
    {context}

    Question:
    {input}

    Answer:
    """)

    # Create the RAG chain
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, stuff_documents_chain)
    
    return retrieval_chain


# --- Streamlit App Interface ---
st.set_page_config(page_title="Elixir AI Assistant", page_icon="📄")
st.title("📄 Elixir AI Assistant")
st.write("Hi How can i help you.")

with st.sidebar:
    st.header("Setup")
    # Get Google API key from user
    google_api_key = st.text_input(
        "Google API Key", 
        key="google_api_key", 
        type="password",
        help="Get your API key from [Google AI Studio](https://aistudio.google.com/app/apikey)."
    )
    if not google_api_key:
        google_api_key = os.getenv("GOOGLE_API_KEY") # Fallback to .env

    if not google_api_key or len(google_api_key) != 40:
        st.error("Invalid Google API Key. Please check and enter the correct API Key.")
    # File uploader for PDF or text file
    uploaded_file = st.file_uploader("Upload your PDF or Text File", type=["pdf", "txt"])

    if uploaded_file and google_api_key:
        st.success("API Key and document loaded successfully!")

# Session state initialization
if "conversation" not in st.session_state:
    st.session_state.conversation = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_file" not in st.session_state:
    st.session_state.processed_file = []

# Main app logic
if uploaded_file and google_api_key:
    # Check if the file has changed
    if st.session_state.processed_file != uploaded_file.name:
        with st.spinner("Processing your document... This may take a moment."):
            try:
                # Create a temporary file to store the uploaded document
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name
                
                # Create vector store and conversation chain
                vector_store = get_vectorstore_from_file(tmp_file_path, google_api_key)
                st.session_state.conversation = get_conversational_rag_chain(vector_store, google_api_key)
                
                # Clean up the temporary file
                os.remove(tmp_file_path)
                
                # Reset chat history for the new document
                st.session_state.chat_history = []
                st.session_state.processed_file = uploaded_file.name
                st.success("Now you can ask your assistant about the document.")

            except Exception as e:
                logger.error(f"An error occurred during document processing: {e}")
                st.error(f"An error occurred during document processing: {e}")
                st.session_state.conversation = None

# Display chat history
if st.session_state.conversation:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Ask a question about your document"):
    # Store user's message in the chat history
    st.chat_message("user").markdown(prompt)

    if st.session_state.conversation:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get the assistant's response to the question
                    response = st.session_state.conversation.invoke({"input": prompt})
                    answer = response.get('answer', 'Sorry, I could not find an answer in the document.')
                    expanded_answer = f"\n\n{answer}"
                    st.markdown(expanded_answer)
                    # Add AI response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": expanded_answer})
                except Exception as e:
                    logger.error(f"An error occurred while getting the response: {e}")
                    st.error(f"An error occurred while getting the response: {e}")
    else:
        st.warning("Please upload a document and enter your API key to start chatting.")
