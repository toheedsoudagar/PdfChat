import os
import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Load environment variables and configure Google API
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDF files
def extract_text_from_pdfs(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def split_text_into_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# Function to create and save a vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    FAISS.from_texts(chunks, embedding=embeddings).save_local("faiss_index")

# Function to load the QA chain
def load_qa_conversational_chain():
    prompt_template = """
    Answer the question with details from the context provided. If the answer is not in the context, respond with 'Answer not available in the context.'\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    return load_qa_chain(
        llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro", client=genai, temperature=0.7),
        chain_type="stuff",
        prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"]),
    )

# Function to handle user queries
def handle_user_query(question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    docs = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True).similarity_search(question)
    chain = load_qa_conversational_chain()
    return chain({"input_documents": docs, "question": question}, return_only_outputs=True)

# Main Streamlit app
def main():
    st.set_page_config(page_title="Nu-Pie Q&A", page_icon="🔍")
    
    st.title("Nu-Pie PDF Q&A Bot")

    # Upload PDFs
    st.sidebar.title("Upload PDF Documents")
    pdf_files = st.sidebar.file_uploader("Choose PDF files", accept_multiple_files=True)
    
    if st.sidebar.button("Process PDFs"):
        if pdf_files:
            with st.spinner("Processing..."):
                text_chunks = split_text_into_chunks(extract_text_from_pdfs(pdf_files))
                create_vector_store(text_chunks)
                st.sidebar.success("PDFs processed successfully!")
        else:
            st.sidebar.warning("Please upload PDFs before processing.")

    # User input for question
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Finding the answer..."):
                try:
                    response = handle_user_query(question)
                    st.write("**Answer:**", response['output_text'])
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a question.")

    st.sidebar.markdown("""
        ## About
        This app is a Q&A tool that lets you upload PDF documents and ask questions about their content. Powered by a Large Language Model and Google Generative AI.
        \nMade with ❤️ by [Nu-Pie Data Science Team](https://nu-pie.com/data-team-as-a-service-dtaas/)
    """)

if __name__ == "__main__":
    main()
