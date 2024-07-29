import os
import sys
from typing import List, Union, Any
from typing_extensions import TypedDict
# from IPython.display import Image, display
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))

from utils import *
from prompt_template import *

os.environ["TAVILY_API_KEY"] = "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"


def crag_main(
    urls: Union[str, List[str]],
):

    doc_from_url = get_document_from_url(urls)

    document_chunks = get_chunks_from_document(doc_from_url)

    retriever = get_retriever(embedding_model_name="all-MiniLM-L6-v2", document_chunks=document_chunks)

    retrieval_grader = get_retrieval_grading_pipeline()

    rag_chain = get_rag_pipeline()

    question_rewriter = get_query_rewriter()

    web_search_tool = get_web_search(k=5)


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
        web_search: str
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
        documents = retriever.get_relevant_documents(question)
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


    def grade_documents(state):
        """
        Determines whether the retrieved documents are relevant to the question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with only filtered relevant documents
        """

        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        # Score each doc
        filtered_docs = []
        web_search = "No"
        for d in documents:
            score = retrieval_grader.invoke(
                {"document": d.page_content, "question": question}
            )
            grade = score["binary_score"]
            if grade == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(d)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")
                web_search = "Yes"
                continue
        return {"documents": filtered_docs, "question": question, "web_search": web_search}


    def transform_query(state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = question_rewriter.invoke({"question": question})
        print(better_question)
        return {"documents": documents, "question": better_question}


    def web_search(state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        print("---WEB SEARCH---")
        question = state["question"]
        # documents = state["documents"]
        documents = state.get("documents", [])

        # Web search
        docs = web_search_tool.invoke({"query": question})
        documents.extend(
            [
                Document(page_content=d["content"], metadata={"url": d["url"]})
                for d in docs if isinstance(d, dict)
            ]
        )
        return {"documents": documents, "question": question}
    

    def decide_to_generate(state):
        """
        Determines whether to generate an answer, or re-generate a question.

        Args:
            state (dict): The current graph state

        Returns:
            str: Binary decision for next node to call
        """

        print("---ASSESS GRADED DOCUMENTS---")
        state["question"]
        web_search = state["web_search"]
        state["documents"]

        if web_search == "Yes":
            # All documents have been filtered check_relevance
            # We will re-generate a new query
            print(
                "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
            )
            return "transform_query"
            # return "web_search"
        else:
            # We have relevant documents, so generate answer
            print("---DECISION: GENERATE---")
            return "generate"

    # Graph
    workflow = StateGraph(GraphState)

    # Define the nodes
    workflow.add_node("retrieve", retrieve)  # retrieve
    workflow.add_node("grade_documents", grade_documents)  # grade documents
    workflow.add_node("generate", generate)  # generatae
    workflow.add_node("transform_query", transform_query)  # transform_query
    workflow.add_node("web_search_node", web_search)  # web search

    # Build graph
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "transform_query": "transform_query",
            "generate": "generate",
        },
    )
    workflow.add_edge("transform_query", "web_search_node")
    workflow.add_edge("web_search_node", "generate")
    workflow.add_edge("generate", END)

    custom_graph = workflow.compile()

    return custom_graph


def chat_with_crag(custom_graph):
    while True:
        # Prompt the user for a question
        question = input("Enter your question (or type 'stop' to exit): ")
        
        # Check if the user wants to stop the interaction
        if question.lower() == 'stop':
            print("Exiting the CRAG pipeline. Goodbye!")
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
    
    chat_with_crag(custom_graph)
