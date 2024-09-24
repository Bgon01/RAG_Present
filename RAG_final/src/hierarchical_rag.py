# src/hierarchical_rag.py

import asyncio
import os
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from helper_functions import retry_with_exponential_backoff

class HierarchicalRAG:
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.summary_store = None
        self.detailed_store = None

    async def encode_pdf_hierarchical(self):
        loader = PyPDFLoader(self.pdf_path)
        documents = await asyncio.to_thread(loader.load)

        summary_llm = ChatOpenAI(temperature=0, model_name="gpt-4", max_tokens=4000)
        summary_chain = load_summarize_chain(summary_llm, chain_type="map_reduce")

        async def summarize_doc(doc):
            summary_output = await retry_with_exponential_backoff(summary_chain.ainvoke([doc]))
            summary = summary_output['output_text']
            return Document(page_content=summary, metadata={"source": self.pdf_path, "page": doc.metadata["page"], "summary": True})

        summaries = []
        batch_size = 5
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_summaries = await asyncio.gather(*[summarize_doc(doc) for doc in batch])
            summaries.extend(batch_summaries)
            await asyncio.sleep(1)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap, length_function=len
        )
        detailed_chunks = await asyncio.to_thread(text_splitter.split_documents, documents)

        for i, chunk in enumerate(detailed_chunks):
            chunk.metadata.update({"chunk_id": i, "summary": False, "page": int(chunk.metadata.get("page", 0))})

        embeddings = OpenAIEmbeddings()

        async def create_vectorstore(docs):
            return await retry_with_exponential_backoff(asyncio.to_thread(FAISS.from_documents, docs, embeddings))

        self.summary_store, self.detailed_store = await asyncio.gather(
            create_vectorstore(summaries),
            create_vectorstore(detailed_chunks)
        )

        # Save the vector stores
        self.summary_store.save_local("vector_stores/summary_store")
        self.detailed_store.save_local("vector_stores/detailed_store")

    def load_vectorstores(self):
        embeddings = OpenAIEmbeddings()
        self.summary_store = FAISS.load_local("vector_stores/summary_store", embeddings, allow_dangerous_serialization=True)
        self.detailed_store = FAISS.load_local("vector_stores/detailed_store", embeddings, allow_dangerous_serialization=True)

    def retrieve_hierarchical(self, query, k_summaries=3, k_chunks=5):
        top_summaries = self.summary_store.similarity_search(query, k=k_summaries)
        relevant_chunks = []
        for summary in top_summaries:
            page_number = summary.metadata["page"]
            page_filter = lambda metadata: metadata["page"] == page_number
            page_chunks = self.detailed_store.similarity_search(query, k=k_chunks, filter=page_filter)
            relevant_chunks.extend(page_chunks)
        return relevant_chunks
