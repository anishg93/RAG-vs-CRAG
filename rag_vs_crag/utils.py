from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain import hub
from typing import List
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Union, Any
from typing_extensions import TypedDict
from IPython.display import Image, display
from langchain.schema import Document
from langgraph.graph import START, END, StateGraph
import uuid
from prompt_template import GradePrompt, GeneratePrompt, RewritePrompt
import os
from pprint import pprint


os.environ["TAVILY_API_KEY"] = "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"
local_llm = "llama3"
model_tested = "llama3-8b"
metadata = f"CRAG, {model_tested}"


def get_document_from_url(
    urls: Union[str, List[str]],
):
    if isinstance(urls, str):
        urls = [urls]

    docs = [WebBaseLoader(url).load() for url in urls]
    docs_list = [item for sublist in docs for item in sublist]

    return docs_list


def get_chunks_from_document(
    document,
    text_chunk_size: int = 250,
    overlap: int = 0,
):
    
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=text_chunk_size, chunk_overlap=overlap
    )
    doc_splits = text_splitter.split_documents(document)

    return doc_splits


def get_retriever(
    embedding_model_name: str,
    document_chunks,
):
    embedding_model = SentenceTransformer(embedding_model_name)

    class CustomSentenceEmbeddings:
        def __init__(self, model: SentenceTransformer):
            self.model = model

        def embed_documents(self, documents: List[str]) -> List[List[float]]:
            return self.model.encode(documents).tolist()
        
        def embed_query(self, query):
            return self.model.encode(query).tolist()

    # Create the embedding object
    custom_embeddings = CustomSentenceEmbeddings(embedding_model)

    # Add to Chroma vectorDB
    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        collection_name="rag-chroma",
        embedding=custom_embeddings
    )
    retriever = vectorstore.as_retriever()

    return retriever


def check_retriever(
    retriever: Any,
    query: str,
    top_k: int = 3,
):
    
    results = retriever.get_relevant_documents(query, top_k=top_k)

    assert len(results) > 0, "No results found"

    return results


# class GradeDocuments(BaseModel):
#     """Binary score for relevance check on retrieved documents."""

#     binary_score: str = Field(...,
#         description="Documents are relevant to the question, 'yes' or 'no'"
#     )


def get_retrieval_grading_pipeline():
    
    grading_llm = ChatOllama(model=local_llm, format="json", temperature=0)

    grading_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GradePrompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    retrieval_grader = grading_prompt | grading_llm | JsonOutputParser()

    return retrieval_grader


def get_rag_pipeline():

    generation_llm = ChatOllama(model=local_llm, temperature=0)

    generation_prompt = hub.pull("rlm/rag-prompt")

    rag_chain_pipeline = generation_prompt | generation_llm | StrOutputParser()

    return rag_chain_pipeline


def get_query_rewriter():
    
    rewriter_llm = ChatOllama(model=local_llm, temperature=0)

    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RewritePrompt),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | rewriter_llm | StrOutputParser()

    return question_rewriter


def get_web_search(
    k: int = 3,
):
    web_search_tool = TavilySearchResults(k=k)

    return web_search_tool


def predict_custom_agent_local_answer(
    custom_graph: StateGraph,
    example: dict
):
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    state_dict = custom_graph.invoke(
        {"question": example["input"], "steps": []}, config
    )
    return {"response": state_dict["generation"], "steps": state_dict["steps"]}


def get_crag_response(
    custom_graph: StateGraph,
    example: dict
):
    # inputs = {"question": "What are the types of agent memory?"}
    for output in custom_graph.stream(example):
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    return value["generation"]