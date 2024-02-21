# Contract Advisor RAG: Towards Building A High-Precision Legal Expert LLM APP

This repo contains the code and data for the 10 Academy Cohort A weekly challenge: Week 11. The goal of this challenge is to build, evaluate and improve a RAG system for Contract Q&A (chatting with a contract and asking questions about the contract).

## Business Objective

Developing the next-generation contract AI by leveraging Hybrid LLM technology (edge, private cloud and LLM services), to build the first, fully autonomous, artificial contract lawyer. The first step in our journey is a powerful contract assistant, with the ultimate goal of developing a fully autonomous contract bot, capable of drafting, reviewing, and negotiating contracts, independently, end-to-end, without human assistance.

## RAG System Overview

RAG, or Retrieval Augmented Generation, is a hybrid AI model that marries the expertise of powerful language models with the richness of external data sources. At its core, RAG leverages a large language model for generating responses, but with a twist – it first retrieves relevant information from a vast pool of external data. This retrieval phase empowers the model to augment its generated responses with information that goes beyond its initial training data, offering more accurate, informed, and context-rich outputs.

## Project Structure

The project consists of the following tasks:

- Task 1: Research ways to improve RAG systems in general.
- Task 2: Build simple Q&A pipeline with RAG using Langchain
- Task 3: Build a RAG evaluation pipeline with RAGAS
- Task 4: Idea to optimize Contract Q&A
- Task 5: Implement at least two enhancements
- Task 6: Interpretation & Reporting

## Data

The data for this project includes:

- Evaluation set which contains two Contracts (a short one and a long one) with a list of 10 questions and correct answers for each.
- RAG datasets from Hugging Face: mini-bioasq and mini-wikipedia

## Dependencies

The project requires the following libraries and frameworks:

- Langchain: a leading LLM application framework
- RAGAS: a RAG evaluation framework
- Hugging Face Transformers: a library for state-of-the-art NLP models

## Usage

To run the project, follow these steps:

- Clone the repo: git clone https://github.com/abdimussa87/legal_contract_advisor_rag.git
- Install the dependencies: pip install -r requirements.txt
- Run the main script: streamlit run app/main.py
- Access the web interface: http://localhost:8501

## References

The project is based on the following references:

- [Langchain for LLM Applications (video course)](https://learn.deeplearning.ai/langchain/lesson/1/introduction)
- [Advanced Retrieval for AI with Chroma (video course)](https://learn.deeplearning.ai/advanced-retrieval-for-ai/lesson/1/introduction)
- [RAGAS Evaluation with Langchain (blog post)](https://blog.langchain.dev/evaluating-rag-pipelines-with-ragas-langsmith/)
