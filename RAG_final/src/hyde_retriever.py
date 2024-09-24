# src/hyde_retriever.py

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS

class HyDERetriever:
    def __init__(self, llm, chunk_size=500):
        self.llm = llm
        self.chunk_size = chunk_size

        self.hyde_prompt = PromptTemplate(
            input_variables=["query", "chunk_size"],
            template="""Given the question '{query}', generate a hypothetical document that directly answers this question. The document should be detailed and in-depth.
The document size has to be exactly {chunk_size} characters.""",
        )
        self.hyde_chain = self.hyde_prompt | self.llm

    def generate_hypothetical_document(self, query):
        input_variables = {"query": query, "chunk_size": self.chunk_size}
        return self.hyde_chain.invoke(input_variables).content

    def retrieve(self, query, vectorstore, k=3):
        hypothetical_doc = self.generate_hypothetical_document(query)
        similar_docs = vectorstore.similarity_search(hypothetical_doc, k=k)
        return similar_docs, hypothetical_doc
