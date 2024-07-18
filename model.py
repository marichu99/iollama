import chromadb
import logging
import sys

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import (Settings, VectorStoreIndex, SimpleDirectoryReader, PromptTemplate)
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global query_engine
query_engine = None

def init_llm():
    llm = Ollama(model="llama2", request_timeout=300.0)
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")

    Settings.llm = llm
    Settings.embed_model = embed_model


def init_index(embed_model):
    reader = SimpleDirectoryReader(input_dir="./docs", recursive=True)
    documents = reader.load_data()

    logging.info("index creating with `%d` documents", len(documents))

    # EphemeralClient does not store any data on disk and Creates an in-memory instance of Chroma
    chroma_client = chromadb.EphemeralClient()
    chroma_collection = chroma_client.create_collection("iollama")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, embed_model=embed_model)

    return index


def init_query_engine(index):
    global query_engine

    # custome prompt template
    template =(
        """
            Imagine you are an intelligent agent trained on pension administration. Your goal is to provide insightful, accurate, and concise answers to questions in this domain.
            Here is some context related to the query:
            -----------------------------------------
            {context_str}
            -----------------------------------------
            Considering the above information, please respond to the following inquiry with detailed references to applicable Java concepts, 
            best practices, or principles where appropriate:

            Question: {query_str}
        """
    )
    qa_template = PromptTemplate(template)

    # build query engine with custom template
    # text_qa_template specifies custom template
    # similarity_top_k configure the retriever to return the top 3 most similar documents,
    # the default value of similarity_top_k is 2
    query_engine = index.as_query_engine(text_qa_template=qa_template, similarity_top_k=3)

    return query_engine


def chat(input_question, user):
    global query_engine
    
    print(f"The query engine is of type {query_engine is None}")

    response = query_engine.query(input_question)
    logging.info("got response from llm - %s", response)

    return response.response