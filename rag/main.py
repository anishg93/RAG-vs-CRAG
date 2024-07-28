from langchain.embeddings import HuggingFaceEmbeddings
from utils import *
from prompt_template import *


## Define the data path
data_path = "databricks/databricks-dolly-15k"

## Load the data
dataset = load_dataset(data_path, split="train")

## Process the data or get the reference documents to be used in the RAG pipeline
data_texts = [f"Instruction: {item['instruction']}\nResponse: {item['response']}" for item in dataset]

def rag_main(
    texts: List[str], 
    embedding_model_name: str,
    query: str
) -> str:
    
    ## Split the texts into chunks
    split_texts = get_chunked_texts(texts)

    ## Get the embeddings based on the model name
    hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    ## Get the retriever based on the embeddings which is the vectorstore retriever
    retriever = get_vectorstore_retriever(text_embeddings=hf_embeddings, text_chunks=split_texts, k=2)

    ## Get the text generation pipeline which is the geneeration part of the RAG pipeline
    custom_llm = get_text_generation_pipeline(llm_model_name="openai-community/gpt2", max_new_tokens=100, do_sample=True, temperature=0.1, top_k=100, top_p=0.95)

    ## Get the RAG pipeline which is the combination of the retriever and the generation pipeline
    rag_pipeline = get_rag_pipeline(template=BasePromptTemplate, retriever=retriever, custom_llm=custom_llm, input_variables=["context", "question"])

    ## Get the answer to the user query
    answer = get_rag_output(rag_pipeline, query)

    ## Remove repetitions from the answer if there is any
    response = remove_repetitions(answer)

    return response

if __name__ == "__main__":

    ## Define the embedding model name
    embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"

    ## Define the query
    query = "What is the capital of France?"
    # import pdb; pdb.set_trace()

    ## Get the response from the RAG pipeline
    response = rag_main(texts=data_texts, embedding_model_name=embedding_model_name, query=query)
    print(response)