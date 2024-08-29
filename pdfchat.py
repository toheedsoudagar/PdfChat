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
    return "".join([PdfReader(pdf).pages[page].extract_text() for pdf in pdf_files for page in range(len(PdfReader(pdf).pages))])

# Function to split text into chunks
def split_text_into_chunks(text):
    return RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000).split_text(text)

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
    return load_qa_conversational_chain()({"input_documents": docs, "question": question}, return_only_outputs=True)

# Function to clear chat history
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs and ask a question"}]

# Main Streamlit app
def main():
    st.set_page_config(page_title="Nu-Pie ChatBot", page_icon="🤖")
    st.sidebar.title("🤗💬 Nu-Pie LLM Q&A Chatbot")
    
    # Upload PDFs
    pdf_files = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True)
    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            text_chunks = split_text_into_chunks(extract_text_from_pdfs(pdf_files))
            create_vector_store(text_chunks)
            st.sidebar.success("Processing completed!")

    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
    st.sidebar.markdown("### About\nThis app is an LLM-powered chatbot built using Streamlit.\n\nMade with ❤️ by [Nu-Pie Data Science Team](https://nu-pie.com/data-team-as-a-service-dtaas/)")
    
    st.title("Chat with Nu-Pie Companion💬")
    
    # Initialize chat session state
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Upload PDFs and ask a question"}]

    # Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message["role"]).write(message["content"])

    # Handle new user input
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"), st.spinner("Thinking..."):
            response = handle_user_query(prompt)
            full_response = ''.join(response['output_text'])
            st.write(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
