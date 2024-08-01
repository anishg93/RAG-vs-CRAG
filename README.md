# RAG-vs-CRAG

## Table of Contents
- [RAG-vs-CRAG](#rag-vs-crag)
  - [Table of Contents](#table-of-contents)
  - [Project Title](#project-title)
  - [Project Description](#project-description)
  - [Installation and Usage](#installation-and-usage)
    - [Comparison Between RAG and CRAG](#comparison-between-rag-and-crag)
  - [Approach](#approach)
  - [Conclusion](#conclusion)

## Project Title

In this project, we delve into the comparison between two advanced approaches in the field of Generative AI : Retrieval-Augmented Generation (RAG) and Corrective Retrieval-Augmented Generation (CRAG). Both methodologies leverage external knowledge sources to enhance the generation of more accurate and contextually relevant responses. However, they differ in their mechanisms and effectiveness in various applications.

## Project Description

This project aims to provide a detailed analysis of RAG and CRAG, highlighting their key differences, advantages, and potential use cases. We will explore:

- The fundamental difference between RAG and CRAG
- Comparative performance
- Practical implementation details

By the end of this project, we hope to offer insights into which method may be more suitable for specific scenarios and how these technologies can be further developed to improve NLP systems.

## Installation and Usage

How you can run this system in your local machine is described below:

- Clone this repo
- Create a virtual env with `Python >= 3.10.0`
- In the terminal, install all the necessary libraries using 
    ```python
    pip install -r requirements.txt
    ```
- Download `ollama` for desktop from [Ollama Official Site](https://ollama.com/)
- Choose some LLM models for your task. I chose `Llama3 8B` (you can choose some other models as well)
- You can change the model name in `/rag_vs_crag/app/main.py` in the variable `local_llm` as shown below,
  ```python
  local_llm = "llama3"
  ```
  - You can select any models from [Ollama Model Hub](https://ollama.com/library) 
- You can also change the embedding model name in `../main.py` in `embedding_model_name` as shown below,
  ```python
  embedding_model_name = "all-MiniLM-L6-v2"
  ```
  - You can select any [Hugging Face Sentence Transformers Models](https://www.sbert.net/docs/sentence_transformer/pretrained_models.html) for text embedding for your retriever
- 

### Comparison Between RAG and CRAG

## Approach

## Conclusion