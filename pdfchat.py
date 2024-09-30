# Main Streamlit app function
def main():
    st.set_page_config(page_title="Nu-Pie ChatBot", page_icon="🤖")

    # Reset session state if starting a new session
    if "pdf_docs" not in st.session_state:
        st.session_state.pdf_docs = []
        st.session_state.messages = [
            {"role": "assistant", "content": "Upload some PDFs and ask me a question"}
        ]

    with st.sidebar:
        st.title("🤗💬 Nu-Pie LLM Personalized Q&A Chatbot App")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        
        if st.button("Submit & Process"):
            if pdf_docs:
                st.session_state.pdf_docs = pdf_docs  # Store uploaded PDFs
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Processing completed!")
            else:
                st.warning("Please upload at least one PDF file.")
        
        st.markdown('''## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)''')
        
        st.write('Made with ❤️ by [Nu-Pie Data Science Team](https://nu-pie.com/data-team-as-a-service-dtaas/)')

    st.title("Chat with Nu-Pie Companion💬")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Only proceed if PDFs have been processed
        if st.session_state.pdf_docs:
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = user_input(prompt)
                        placeholder = st.empty()
                        placeholder.markdown(response)
                if response:
                    st.session_state.messages.append({"role": "assistant", "content": response})
        else:
            with st.chat_message("assistant"):
                st.write("Please upload PDFs to get answers.")

if __name__ == "__main__":
    main()
