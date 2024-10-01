import os
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from multiple PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            with pdfplumber.open(pdf) as pdf_reader:
                for page in pdf_reader.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text + "\n"
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")
    return text.strip()

# Function to split the extracted text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    return splitter.split_text(text)

# Function to create a vector store for fast search
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to set up the conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    You are a knowledgeable assistant. Provide detailed answers based on the context provided. If the answer is not available, state that clearly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
    ]

# Function to handle user input and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Search for relevant documents
    docs = vector_store.similarity_search(user_question)
    if not docs:
        return "I couldn't find any relevant information in the provided context."
    
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="PDF Chatbot", page_icon="🤖")

    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
        ]

    with st.sidebar:
        st.title("PDF Chatbot")
        pdf_docs = st.file_uploader("Upload your PDF Files and click 'Submit & Process'", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                st.session_state.pdf_docs = pdf_docs
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Processing completed!")
                    else:
                        st.warning("No text was extracted from the PDFs.")
            else:
                st.warning("Please upload at least one PDF file.")

        st.markdown('## About This App')
        st.write('This app uses a language model to answer questions based on the contents of uploaded PDFs.')

    st.title("Chat with PDF Assistant")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.pdf_docs:
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = user_input(prompt)
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.write(response)
        else:
            with st.chat_message("assistant"):
                st.write("Please upload PDFs to get answers.")

if __name__ == "__main__":
    main()
