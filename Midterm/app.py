from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
import chainlit as cl
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.memory import ChatMessageHistory, ConversationBufferMemory
from langchain.chains import (
    ConversationalRetrievalChain,
)
from langchain_openai import ChatOpenAI
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import tiktoken
from typing import List
#from langchain_community.callbacks import wandb_tracing_enabled
import os

def tiktoken_len(text):
    tokens = tiktoken.encoding_for_model("gpt-3.5-turbo").encode(
        text,
    )
    return len(tokens)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 800,
    chunk_overlap = 50,
    length_function = tiktoken_len,
)

template = """Answer the question based only on the following context. If you cannot answer the question with the context, please respond with 'I don't know':

Context:
{context}

Question:
{question}
"""


load_dotenv()

@cl.on_chat_start
async def on_chat_start():
    msg = cl.Message(content="Processing Nvidia filing...", disable_feedback=True)
    await msg.send()

    docs = PyMuPDFLoader("https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf").load()

    # split the documents into chunks
    split_chunks = text_splitter.split_documents(docs)

    # Create a FAISS vector store
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vector_store = FAISS.from_documents(split_chunks, embeddings)

    message_history = ChatMessageHistory()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_template(template)

     # Create a chain that uses the Chroma vector store
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    # Let the user know that the system is ready
    msg.content = "Nvidia filing processed. You can now ask questions against this document https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4.pdf!"
    await msg.update()

    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    # unset the environment variable and use a context manager instead
    #if "LANGCHAIN_WANDB_TRACING" in os.environ:
    #    del os.environ["LANGCHAIN_WANDB_TRACING"]

    #with wandb_tracing_enabled():
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    res = await chain.ainvoke(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]
    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
            source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()