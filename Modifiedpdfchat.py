import os
from PyPDF2 import PdfReader
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
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text + "\n"  # Add a newline for separation
        except Exception as e:
            st.error(f"Error reading {pdf.name}: {e}")  # Error handling for PDF reading
    return text

# Function to split the extracted text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,  # Maximum size of each chunk
        chunk_overlap=1000  # Overlap between chunks for context
    )
    return splitter.split_text(text)

# Function to create a vector store for fast retrieval
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")  # Save vector store locally

# Function to set up the conversational chain for question answering
def get_conversational_chain():
    prompt_template = """
    Answer the question based on the provided context. If the answer is not available,
    respond with "answer is not available in the context." Do not guess.\n\n
    Context:\n{context}?\n
    Question:\n{question}\n
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

# Function to process user input and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

    # Retrieve relevant document chunks based on the user's question
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response['output_text']

# Main function for the Streamlit app
def main():
    st.set_page_config(page_title="Nu-Pie ChatBot", page_icon="🤖")

    # Initialize session state for storing uploaded PDFs and messages
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question."}
        ]

    with st.sidebar:
        st.title("🤗💬 Nu-Pie LLM Personalized Q&A Chatbot App")
        pdf_docs = st.file_uploader("Upload your PDF Files and click 'Submit & Process'", accept_multiple_files=True)

        if st.button("Submit & Process"):
            if pdf_docs:
                st.session_state.pdf_docs = pdf_docs  # Store uploaded PDFs
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)  # Extract text from PDFs
                    if raw_text:  # Ensure text was extracted
                        text_chunks = get_text_chunks(raw_text)  # Split text into chunks
                        get_vector_store(text_chunks)  # Create and save vector store
                        st.success("Processing completed!")
                    else:
                        st.warning("No text was extracted from the PDFs.")
            else:
                st.warning("Please upload at least one PDF file.")

        st.markdown('''## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)''')
        st.write('Made with ❤️ by [Nu-Pie Data Science Team](https://nu-pie.com/data-team-as-a-service-dtaas/)')

    st.title("Chat with Nu-Pie Companion💬")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Handle user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Proceed only if PDFs have been processed
        if st.session_state.pdf_docs:
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = user_input(prompt)  # Generate AI response
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        st.write(response)  # Display response
        else:
            with st.chat_message("assistant"):
                st.write("Please upload PDFs to get answers.")

if __name__ == "__main__":
    main()
