import os
from pprint import pprint
from typing import Any, List, Union

from dotenv import load_dotenv
from langchain import hub
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph
from sentence_transformers import SentenceTransformer

from prompt_template import GradePrompt, RewritePrompt

## Load the API key from the .env file
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")


## Util function to get documents from URL(s) provided or the pdf file(s) provided
def get_document_from_url(
    sources: Union[str, List[str]],
):
    ## Check if the input is a string or a list
    if isinstance(sources, str):
        sources = [sources]

    docs = []
    for source in sources:
        if source.startswith(('http://', 'https://')):
            # If it's a web URL
            loader = WebBaseLoader(source)
        elif source.lower().endswith('.pdf') and os.path.isfile(source):
            # If it's a local PDF file
            loader = PyPDFLoader(source)
        else:
            raise ValueError(f"Unsupported source or file not found: {source}")
        
        docs.extend(loader.load())

    return docs


## Util function to split the document into chunks
def get_chunks_from_document(
    document,
    text_chunk_size: int = 250,
    overlap: int = 0,
):

    ## Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=text_chunk_size, chunk_overlap=overlap
    )
    doc_splits = text_splitter.split_documents(document)

    return doc_splits


## Util function to get the retriever for the documents
def get_retriever(
    embedding_model_name: str,
    document_chunks,
):

    ## Load the Sentence Transformer model for document embeddings
    embedding_model = SentenceTransformer(embedding_model_name)

    ## Custom class for Sentence Embeddings so that it has embed_documents and embed_query methods
    class CustomSentenceEmbeddings:
        def __init__(self, model: SentenceTransformer):
            self.model = model

        def embed_documents(self, documents: List[str]) -> List[List[float]]:
            return self.model.encode(documents).tolist()

        def embed_query(self, query):
            return self.model.encode(query).tolist()

    ## Create the embedding object
    custom_embeddings = CustomSentenceEmbeddings(embedding_model)

    ## Add to Chroma vectorDB
    vectorstore = Chroma.from_documents(
        documents=document_chunks,
        collection_name="rag-chroma",
        embedding=custom_embeddings,
    )

    ## Get the retriever
    retriever = vectorstore.as_retriever()

    return retriever


## Util function to check if the retriever is working
def check_retriever(
    retriever: Any,
    query: str,
    top_k: int = 3,
):

    ## Get the relevant documents using the retriever
    results = retriever.get_relevant_documents(query, top_k=top_k)

    ## Check if results are found
    assert len(results) > 0, "No results found"

    return results


## Util function to get the retrieval grading pipeline
def get_retrieval_grading_pipeline(
    local_llm
):

    ## Load the local LLM model for grading
    grading_llm = ChatOllama(model=local_llm, format="json", temperature=0)

    ## Define the grading prompt template
    grading_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GradePrompt),
            (
                "human",
                "Retrieved document: \n\n {document} \n\n User question: {question}",
            ),
        ]
    )

    ## Define the retrieval grader pipeline using the grading prompt, LLM model and JSON output parser
    retrieval_grader = grading_prompt | grading_llm | JsonOutputParser()

    return retrieval_grader


def get_rag_pipeline(
    local_llm
):

    ## Load the local LLM model for generation
    generation_llm = ChatOllama(model=local_llm, temperature=0)

    ## Define the generation prompt template using Langchain Hub
    generation_prompt = hub.pull("rlm/rag-prompt")

    ## Define the RAG pipeline using the generation prompt, LLM model and string output parser
    rag_chain_pipeline = generation_prompt | generation_llm | StrOutputParser()

    return rag_chain_pipeline


def get_query_rewriter(
    local_llm
):

    ## Load the local LLM model for rewriting thye initial query
    rewriter_llm = ChatOllama(model=local_llm, temperature=0)

    ## Define the rewrite prompt template
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", RewritePrompt),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    ## Define the question rewriter using the rewrite prompt, LLM model and string output parser
    question_rewriter = re_write_prompt | rewriter_llm | StrOutputParser()

    return question_rewriter


def get_web_search(
    k: int = 3,
):

    ## Define the web search tool using Tavily Search Results
    web_search_tool = TavilySearchResults(max_results=k)

    return web_search_tool


def get_crag_response(custom_graph: StateGraph, example: dict):

    ## Stream the output from the custom graph defined using all the individual nodes
    for output in custom_graph.stream(example):

        ## Print the output
        for key, value in output.items():
            # Node
            pprint(f"Node '{key}':")
            # Optional: print full state at each node
            # pprint.pprint(value["keys"], indent=2, width=80, depth=None)
        pprint("\n---\n")

    return value["generation"]
