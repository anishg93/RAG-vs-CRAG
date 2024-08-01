import os
import sys
from typing import List, Union

from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
)

from prompt_template import *
from utils import *


## Define the RAG pipeline to generate the answer to the prvioded query
def rag_main(
    urls: Union[str, List[str]],
    local_llm: str,
    embedding_model_name: str,
):

    ## Get the documents from the provided URLs (one URL or a list of URLs)
    doc_from_url = get_document_from_url(urls)

    ## Split the document into chunks
    document_chunks = get_chunks_from_document(doc_from_url)

    ## Get the retriever for the documents
    retriever = get_retriever(
        embedding_model_name, document_chunks=document_chunks
    )

    ## Get the RAG generation pipeline
    rag_chain = get_rag_pipeline(local_llm)

    ## Define the graph state class to orchestrate the RAG pipeline
    class GraphState(TypedDict):
        """
        Represents the state of our graph.

        Attributes:
            question: question
            generation: LLM generation
            web_search: whether to add search
            documents: list of documents
        """

        question: str
        generation: str
        documents: List[str]

    ## Define the retrieve node
    def retrieve(state):
        """
        Retrieve documents

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """
        print("---RETRIEVE---")
        question = state["question"]

        ## Documents retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}

    ## Define the generate node
    def generate(state):
        """
        Generate answer

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, generation, that contains LLM generation
        """
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        ## RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    ## Define the workflow graph
    workflow = StateGraph(GraphState)

    ## Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("generate", generate)  # generatae

    ## Build the graph with the nodes
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    ## Compile the graph
    custom_graph = workflow.compile()

    return custom_graph


## Define the function to chat with the RAG pipeline
def chat_with_rag(custom_graph: StateGraph):

    while True:

        # Prompt the user for a query as input
        question = input("Enter your question (or type 'stop' to exit): ")

        # Check if the user wants to stop the interaction
        if question.lower() == "stop":
            print("Exiting the RAG pipeline. Goodbye!")
            break

        ## Create an example dictionary with the user's query
        example = {"question": question}

        ## Get the response from the custom graph
        response = get_crag_response(custom_graph=custom_graph, example=example)

        ## Print the answer and the steps taken
        print("\nAnswer:\n", response)
        print("\n")
