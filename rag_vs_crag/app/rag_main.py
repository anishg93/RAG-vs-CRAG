import os
import sys
from typing import List
from typing_extensions import TypedDict
# from IPython.display import Image, display
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
from typing import List, Union, Any

from utils import *
from prompt_template import *


def rag_main(
    urls: Union[str, List[str]],
):


    doc_from_url = get_document_from_url(urls)

    document_chunks = get_chunks_from_document(doc_from_url)

    retriever = get_retriever(embedding_model_name="all-MiniLM-L6-v2", document_chunks=document_chunks)

    # retrieval_grader = get_retrieval_grading_pipeline(retriever=retriever, input_vars=['question', 'documents'])
    # retrieval_grader = get_retrieval_grading_pipeline()

    # docs = retriever.invoke("How are you?")
    # doc_txt = docs[1].page_content
    # response_temp = retrieval_grader.invoke({"question": "How are you?", "documents": doc_txt})
    # print(response_temp)

    # rag_chain = get_rag_pipeline(retriever=retriever, input_vars=['question', 'documents'])
    rag_chain = get_rag_pipeline()

    # question_rewriter = get_query_rewriter(input_vars=["question"])
    # question_rewriter = get_query_rewriter()

    # web_search_tool = get_web_search(k=5)


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
        # web_search: str
        documents: List[str]


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

        # Retrieval
        documents = retriever.invoke(question)
        return {"documents": documents, "question": question}


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

        # RAG generation
        generation = rag_chain.invoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    # Graph
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("generate", generate)  # generatae

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()

    return custom_graph

def chat_with_rag(custom_graph):
    while True:
        # Prompt the user for a question
        question = input("Enter your question (or type 'stop' to exit): ")
        
        # Check if the user wants to stop the interaction
        if question.lower() == 'stop':
            print("Exiting the RAG pipeline. Goodbye!")
            break
        
        # Create an example dictionary with the user's question
        example = {"question": question}
        
        # Get the response from the custom RAG pipeline
        response = get_crag_response(custom_graph=custom_graph, example=example)

        # answer = response["response"]
        # steps = response["steps"]
        
        # Print the answer and the steps taken
        print("\nAnswer:\n", response)
        # print("\nSteps:\n", steps)
        print("\n")


if __name__ == "__main__":
    
    chat_with_rag(custom_graph)
