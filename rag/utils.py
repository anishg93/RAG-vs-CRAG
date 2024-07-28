import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from datasets import load_dataset
from langchain import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.llms.base import LLM
from typing import Any, List, Mapping, Optional


def get_chunked_texts(
    texts: List[str], 
    chunk_size: int = 512,
    overlap: int = 50,
):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    split_texts = text_splitter.split_text("\n".join(texts))

    return split_texts


def get_vectorstore_retriever(
    text_embeddings: Any,
    text_chunks: List[str],
    k: int=5,
):
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=text_embeddings)

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    return retriever


def get_text_generation_pipeline(
    llm_model_name: str = "openai-community/gpt2",
    max_new_tokens: int = 100,
    do_sample: bool = True,
    temperature: float = 0.0,
    top_k: int = 50,
    top_p: float = 0.95
) -> LLM:
    
    # Load the tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
    model = AutoModelForCausalLM.from_pretrained(llm_model_name)

    # Handle the pad token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set up the text generation pipeline
    text_generation = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        truncation=True
    )

    class CustomHuggingFaceLLM(LLM):
        pipeline: Any

        def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
            response = self.pipeline(prompt, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature)
            return response[0]['generated_text']

        @property
        def _identifying_params(self) -> Mapping[str, Any]:
            return {"name_of_model": self.pipeline.model.config._name_or_path}

        def llm_type(self) -> str:
            return "custom_huggingface_pipeline"

        @property
        def _llm_type(self) -> str:
            return "custom_huggingface_pipeline"

    # Create an instance of the custom LLM class
    custom_llm = CustomHuggingFaceLLM(pipeline=text_generation)

    return custom_llm




def get_rag_pipeline(
    template: str,
    retriever: Any,
    custom_llm: Any,
    input_variables: List[str] = ["context", "question"],
):
    prompt = PromptTemplate(template=template, input_variables=input_variables)

    # Set up the RAG chain
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | custom_llm
        | StrOutputParser()
    )

    return rag_chain


def get_rag_output(
    rag_chain: Any,
    query: str,
):
    result=None

    try:
        result = rag_chain.invoke(query)

        # print("Result:\n", result)
        # print("\n\n")

        if isinstance(result, list):
            # If result is a list, remove duplicates
            result = list(set(result))
             # Join the unique responses into a single string
            result = " ".join(result)
        elif isinstance(result, dict) and 'generated_text' in result:
            # If result is a dict with 'generated_text', extract it
            result = result['generated_text']

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Error type:", type(e))

    return result


def remove_repetitions(
    text
):
    # Step 1: Split the text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    # Step 2: Keep the first sentence
    result = sentences[0]
    
    # Step 3: Process subsequent sentences
    for sentence in sentences[1:]:
        # Check if this sentence is not a substring of what we already have
        if sentence not in result:
            # Check if the sentence is complete (ends with punctuation)
            if sentence[-1] in '.!?':
                result += ' ' + sentence
            else:
                # If it's an incomplete sentence, we stop processing
                break
    
    return result.strip()
