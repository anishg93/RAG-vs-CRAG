from crag_main import chat_with_crag, crag_main
from rag_main import chat_with_rag, rag_main
import os
import subprocess


def pull_model_once(local_llm: str):
    
    ## Define the path for the flag file
    flag_file = "llm_pulled.flag"

    ## Check if the flag file exists
    if not os.path.exists(flag_file):
        print("Flag file not found. Pulling the model...")

        ## Construct the shell command
        command = f"ollama pull {local_llm}"

        try:
            ## Execute the command, suppressing the output
            with open(os.devnull, 'w') as devnull:
                result = subprocess.run(command, shell=True, check=True, stdout=devnull, stderr=devnull)
            
            ## Create the flag file to indicate the model has been pulled
            with open(flag_file, 'w') as f:
                f.write('Model pulled successfully.')
            print("Flag file created.")

        except subprocess.CalledProcessError as e:
            print(f"Error occurred while pulling the model.")
    else:
        print("Model has already been pulled. Skipping the pull command.")



if __name__ == "__main__":

    ## Define the local LLM model name
    local_llm = "llama3"

    ## Pull the model once from Ollama
    pull_model_once(local_llm=local_llm)

    ## Test urls
    # urls = "https://theworldtravelguy.com/the-great-pyramids-of-giza-in-egypt-facts-tours-pictures/"

    urls = "https://www.findingtheuniverse.com/two-weeks-in-ukmy-perfect-itinerary/"

    # urls = "https://indiacurrents.com/tracing-the-history-of-biryani/"

    # urls = "https://waitbutwhy.com/2014/06/taming-mammoth-let-peoples-opinions-run-life.html"

    ## Define the Sentence Transformer model name
    embedding_model_name = "all-MiniLM-L6-v2"

    ## Call the main functions for both RAG and CRAG
    custom_graph_rag = rag_main(urls, local_llm=local_llm, embedding_model_name=embedding_model_name)
    custom_graph_crag = crag_main(urls, local_llm=local_llm, embedding_model_name=embedding_model_name, max_results_k=5)

    ## Chat with the user and take the agent type as input
    chain_type = input("Enter the agent type you want to use ('rag' or 'crag'): ")

    ## Call the chat function for the respective chain type
    if chain_type == "rag":
        chat_with_rag(custom_graph_rag)
    elif chain_type == "crag":
        chat_with_crag(custom_graph_crag)