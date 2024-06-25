# Import Required Libraries
import os
from dotenv import load_dotenv

import openai
import fitz  # PyMuPDF
import pandas as pd
from transformers import pipeline
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
import chainlit as cl
import tiktoken

# Specific imports from the libraries
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
#old import from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
#old import from langchain_openai import ChatOpenAI
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

# Set Environment Variables
load_dotenv()

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client after loading the environment variables
openai.api_key = OPENAI_API_KEY

# Load and split documents
loader = PyMuPDFLoader("/home/user/app/data/airbnb_q1_2024.pdf")
#old file path is loader = PyMuPDFLoader("/Users/sampazar/AIE3-Midterm/data/airbnb_q1_2024.pdf")
documents = loader.load()

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-4o").encode(text)
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, 
    chunk_overlap=100,
    length_function = tiktoken_len
)

split_chunks = text_splitter.split_documents(documents)


# Load OpenAI Embeddings Model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Creating a Qdrant Vector Store
qdrant_vector_store = Qdrant.from_documents(
    split_chunks,
    embeddings,
    location=":memory:",
    collection_name="Airbnb_Q1_2024",
)

# Create a Retriever
retriever = qdrant_vector_store.as_retriever()

# Create a prompt template
template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""

prompt = ChatPromptTemplate.from_template(template)

# Define the primary LLM
primary_llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Creating a Retrieval Augmented Generation (RAG) Chain
retrieval_augmented_qa_chain = (
    # INVOKE CHAIN WITH: {"question" : "<>"}
    # "question" : populated by getting the value of the "question" key
    # "context"  : populated by getting the value of the "question" key and chaining it into the base_retriever
    {"context": itemgetter("question") | retriever, "question": itemgetter("question")}
    # "context"  : is assigned to a RunnablePassthrough object (will not be called or considered in the next step)
    #              by getting the value of the "context" key from the previous step
    | RunnablePassthrough.assign(context=itemgetter("context"))
    # "response" : the "context" and "question" values are used to format our prompt object and then piped
    #              into the LLM and stored in a key called "response"
    # "context"  : populated by getting the value of the "context" key from the previous step
    | {"response": prompt | primary_llm, "context": itemgetter("context")}
)

# Chainlit integration for deployment
@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-4o",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    cl.user_session.set("settings", settings)

@cl.on_message  # marks a function that should be run each time the chatbot receives a message from a user
async def handle_message(message: cl.Message):
    settings = cl.user_session.get("settings")

    response = retrieval_augmented_qa_chain.invoke({"question": message.content})
    
    #msg = cl.Message(content=response["response"])
    #await msg.send()

    # Extracting and sending just the content
    content = response["response"].content
    pretty_content = content.strip()  # Remove any leading/trailing whitespace

    await cl.Message(content=pretty_content).send()
