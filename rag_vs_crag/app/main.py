from crag_main import chat_with_crag, crag_main
from rag_main import chat_with_rag, rag_main

if __name__ == "__main__":

    ## Test urls
    # urls = "https://theworldtravelguy.com/the-great-pyramids-of-giza-in-egypt-facts-tours-pictures/"

    urls = "https://www.findingtheuniverse.com/two-weeks-in-ukmy-perfect-itinerary/"

    ## Call the main functions for both RAG and CRAG
    custom_graph_rag = rag_main(urls)
    custom_graph_crag = crag_main(urls)

    ## Chat with the user and take the chain type as input
    chain_type = input("Enter the chain type you want to use ('rag' or 'crag'): ")

    ## Call the chat function for the respective chain type
    if chain_type == "rag":
        chat_with_rag(custom_graph_rag)
    elif chain_type == "crag":
        chat_with_crag(custom_graph_crag)
