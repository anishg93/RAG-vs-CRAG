from rag_main import rag_main, chat_with_rag
from crag_main import crag_main, chat_with_crag


if __name__ == "__main__":

    urls = "https://theworldtravelguy.com/the-great-pyramids-of-giza-in-egypt-facts-tours-pictures/"

    custom_graph_rag = rag_main(urls)
    custom_graph_crag = crag_main(urls)

    chain_type = input("Enter the chain type you want to use: ")

    if chain_type == "rag":
        chat_with_rag(custom_graph_rag)
    elif chain_type == "crag":
        chat_with_crag(custom_graph_crag)