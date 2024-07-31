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
from dotenv import load_dotenv


## Load the API key from the .env file
load_dotenv()
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

## Define the local LLM model
local_llm = "llama3"
model_tested = "llama3-8b"
metadata = f"CRAG, {model_tested}"


## Util function to get documents from URL provided
def get_document_from_url(
    urls: Union[str, List[str]],
):
    
    ## Check if the input is a string or a list
    if isinstance(urls, str):
        urls = [urls]

    ## Load the documents from the URLs
    docs = [WebBaseLoader(url).load() for url in urls]

    ## Flatten the list of documents
    docs_list = [item for sublist in docs for item in sublist]

    return docs_list


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
        embedding=custom_embeddings
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
def get_retrieval_grading_pipeline():
    
    ## Load the local LLM model for grading
    grading_llm = ChatOllama(model=local_llm, format="json", temperature=0)

    ## Define the grading prompt template
    grading_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", GradePrompt),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ]
    )

    ## Define the retrieval grader pipeline using the grading prompt, LLM model and JSON output parser
    retrieval_grader = grading_prompt | grading_llm | JsonOutputParser()

    return retrieval_grader


def get_rag_pipeline():

    ## Load the local LLM model for generation
    generation_llm = ChatOllama(model=local_llm, temperature=0)

    ## Define the generation prompt template using Langchain Hub
    generation_prompt = hub.pull("rlm/rag-prompt")

    ## Define the RAG pipeline using the generation prompt, LLM model and string output parser
    rag_chain_pipeline = generation_prompt | generation_llm | StrOutputParser()

    return rag_chain_pipeline


def get_query_rewriter():
    
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
    web_search_tool = TavilySearchResults(k=k)

    return web_search_tool


# def predict_custom_agent_local_answer(
#     custom_graph: StateGraph,
#     example: dict
# ):
#     config = {"configurable": {"thread_id": str(uuid.uuid4())}}
#     state_dict = custom_graph.invoke(
#         {"question": example["input"], "steps": []}, config
#     )
#     return {"response": state_dict["generation"], "steps": state_dict["steps"]}


def get_crag_response(
    custom_graph: StateGraph,
    example: dict
):

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