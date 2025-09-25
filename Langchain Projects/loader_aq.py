from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, CSVLoader, WebBaseLoader
)
from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import streamlit as st
import tempfile
import os

# Load environment variables
load_dotenv()

# ====================== APP TITLE =========================
st.set_page_config(page_title="Document QA Chatbot", layout="wide")
st.title("📄 Loader Document QA Chatbot")
st.markdown(
    "Upload a document or provide a URL, ask your questions, "
    "and get structured answers using Google Gemini LLM!"
)

# ====================== SIDEBAR =========================
with st.sidebar:
    st.header("Select Document Type")
    document_type = st.selectbox(
        "Document Type",
        options=["Text Document", "WebBase Document", "CSV Document", "PDF Document"]
    )
    st.markdown("---")

# ====================== LLM & Prompt ======================
llm_model = init_chat_model(model="gemini-2.0-flash", model_provider="google-genai")

template = PromptTemplate(
    template="""Answer the user question: {question}.\n
                Only find the answer in the uploaded text:\n{text}.
                Format the answer in a structured way.
                If the content contains a table, display it properly.""",
    input_variables=['question', 'text']
)

parser = StrOutputParser()
qa_chain = template | llm_model | parser

# ====================== USER QUESTION INPUT ======================
user_question = st.text_input("💬 Ask a question about the uploaded document:")

# ====================== HELPER FUNCTION =========================
def save_uploaded_file(uploaded_file, suffix):
    """Save uploaded file temporarily and return path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

# ====================== TEXT DOCUMENT ===========================
if document_type == "Text Document":
    st.info("📄 Upload a plain text file (.txt). Each line or full text will be read.")
    text_file = st.file_uploader("Upload a text file", type=["txt"])
    
    if text_file:
        temp_path = save_uploaded_file(text_file, ".txt")
        loader = TextLoader(temp_path)
        docs = loader.load()
        
        if user_question and docs:
            result = qa_chain.invoke({
                "question": user_question,
                "text": docs[0].page_content[:2000]
            })
            st.subheader("✅ Answer")
            st.write(result)
        
        if st.button("Show Full Text Content"):
            st.subheader("📄 File Content")
            st.write(docs[0].page_content)

# ====================== WEBPAGE DOCUMENT ========================
elif document_type == "WebBase Document":
    st.info("🌐 Enter a public webpage URL (HTML only). Not for PDFs or login-protected pages.")
    url = st.text_input("Enter URL:", help="Example: https://example.com")
    
    if url:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        if user_question:
            result = qa_chain.invoke({
                "question": user_question,
                "text": docs[0].page_content[:2000]
            })
            st.subheader("✅ Answer")
            st.write(result)
        
        if st.button("Show Page Content"):
            st.subheader("🌐 Webpage Content")
            st.write(docs[0].page_content)

# ====================== CSV DOCUMENT ============================
elif document_type == "CSV Document":
    st.info("📊 Upload a CSV file. Each row will be read as a document.")
    csv_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if csv_file:
        temp_path = save_uploaded_file(csv_file, ".csv")
        loader = CSVLoader(file_path=temp_path)
        docs = list(loader.lazy_load())
        
        if st.button("Show CSV Content"):
            for i, doc in enumerate(docs):
                st.subheader(f"Row {i+1}")
                st.write(doc.page_content)
                st.write("Metadata:", doc.metadata)
        
        if user_question:
            # Combine all rows as text for LLM
            combined_text = "\n".join([d.page_content for d in docs[:20]])  # limit first 20 rows
            result = qa_chain.invoke({
                "question": user_question,
                "text": combined_text
            })
            st.subheader("✅ Answer")
            st.write(result)

# ====================== PDF DOCUMENT ============================
elif document_type == "PDF Document":
    st.info("📕 Upload a PDF file. Each page will be read as a separate document.")
    pdf_file = st.file_uploader("Upload PDF file", type=['pdf'])
    
    if pdf_file:
        temp_path = save_uploaded_file(pdf_file, ".pdf")
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        if user_question and docs:
            result = qa_chain.invoke({
                "question": user_question,
                "text": docs[0].page_content[:2000]
            })
            st.subheader("✅ Answer")
            st.write(result)
        
        if st.button("Show PDF Content"):
            st.subheader("📕 PDF Content")
            st.write(docs[0].page_content)
