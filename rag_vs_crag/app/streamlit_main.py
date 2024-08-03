import os
import subprocess
import streamlit as st
from crag_main import chat_with_crag_ui, crag_main
from rag_main import chat_with_rag_ui, rag_main


def pull_model_once(local_llm: str):
    flag_file = "llm_pulled.flag"
    if not os.path.exists(flag_file):
        st.write("Flag file not found. Pulling the model...")
        command = f"ollama pull {local_llm}"
        try:
            with open(os.devnull, 'w') as devnull:
                subprocess.run(command, shell=True, check=True, stdout=devnull, stderr=devnull)
            with open(flag_file, 'w') as f:
                f.write('Model pulled successfully.')
            st.write("Flag file created.")
        except subprocess.CalledProcessError:
            st.write("Error occurred while pulling the model.")
    else:
        st.write("Model has already been pulled. Skipping the pull command.")


url = "https://theworldtravelguy.com/the-great-pyramids-of-giza-in-egypt-facts-tours-pictures/"
local_llm = "llama3"
embedding_model_name = "all-MiniLM-L6-v2"


def main(url, local_llm, embedding_model_name):
    st.title("Corrective RAG Chat Interface")
    st.write("Welcome to the Corrective RAG chat interface!")

    pull_model_once(local_llm=local_llm)

    urls = st.text_input("Enter the URL(s) for the documents (comma-separated if multiple):", url)
    

    st.write("Initializing the RAG and CRAG agents...")
    custom_graph_rag = rag_main(urls, local_llm=local_llm, embedding_model_name=embedding_model_name)
    custom_graph_crag = crag_main(urls, local_llm=local_llm, embedding_model_name=embedding_model_name, max_results_k=5)

    chain_type = st.selectbox("Select the agent type you want to use:", ("rag", "crag"))

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    user_input = st.text_input("You:", key="user_input")
    if st.button("Send"):
        if user_input:
            st.session_state.chat_history.append(("User", user_input))
            with st.spinner("Agent is typing..."):
                if chain_type == "rag":
                    response = chat_with_rag_ui(custom_graph_rag, user_input)
                elif chain_type == "crag":
                    response = chat_with_crag_ui(custom_graph_crag, user_input)
            st.session_state.chat_history.append(("Agent", response))
            st.write(f"**You:** {user_input}")
            st.write(f"**Agent:** {response}")

    if st.session_state.chat_history:
        st.write("## Chat History")
        for speaker, text in st.session_state.chat_history:
            st.write(f"**{speaker}:** {text}")

if __name__ == "__main__":
    main(url=url, local_llm=local_llm, embedding_model_name=embedding_model_name)
