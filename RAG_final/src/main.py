# src/main.py

import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# Ensure the src directory is in the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from hierarchical_rag import HierarchicalRAG
from hyde_retriever import HyDERetriever
from reranker import rerank_documents
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from ragas_evaluation import RAGASEvaluator, load_test_data

def parse_args():
    parser = argparse.ArgumentParser(description="Run the combined Hierarchical RAG with HYDE and Reranking.")
    parser.add_argument("--pdf_path", type=str, default="../data/ALLIANZ.pdf", help="Path to the PDF document.")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Size of each text chunk.")
    parser.add_argument("--chunk_overlap", type=int, default=200, help="Overlap between consecutive chunks.")
    parser.add_argument("--query", type=str, help="Query to search in the document.")
    parser.add_argument("--evaluate", action='store_true', help="Run evaluation with RAGAS.")
    parser.add_argument("--test_data_path", type=str, default="data/test_queries.csv", help="Path to test queries CSV file.")
    return parser.parse_args()

class CombinedRAGPipeline:
    def __init__(self, pdf_path, chunk_size=1000, chunk_overlap=200):
        self.pdf_path = pdf_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        self.embeddings = OpenAIEmbeddings()

        self.hierarchical_rag = HierarchicalRAG(pdf_path, chunk_size, chunk_overlap)
        self.hyde_retriever = HyDERetriever(self.llm, chunk_size)

    async def process_query(self, query):
        # Generate hypothetical document
        hypothetical_doc = self.hyde_retriever.generate_hypothetical_document(query)

        # Retrieve relevant documents using the hypothetical document
        relevant_chunks = self.hierarchical_rag.retrieve_hierarchical(hypothetical_doc)

        # Rerank the retrieved documents
        reranked_docs = rerank_documents(query, relevant_chunks)
        return reranked_docs

    async def run(self, query):
        # Load or encode vector stores
        if not (os.path.exists("vector_stores/summary_store") and os.path.exists("vector_stores/detailed_store")):
            print("Encoding PDF and creating vector stores...")
            await self.hierarchical_rag.encode_pdf_hierarchical()
        else:
            print("Loading existing vector stores...")
            self.hierarchical_rag.load_vectorstores()

        # Process the query and get reranked documents
        reranked_docs = await self.process_query(query)

        # Display the final documents
        print("\nTop Reranked Documents:")
        for i, doc in enumerate(reranked_docs):
            print(f"\nDocument {i + 1}:")
            print(f"Page: {doc.metadata.get('page', 'N/A')}")
            print(f"Content: {doc.page_content[:500]}...")  # Limit to 500 chars
            print("---")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

    args = parse_args()
    pipeline = CombinedRAGPipeline(args.pdf_path, args.chunk_size, args.chunk_overlap)

    if args.evaluate:
        # Run evaluation
        llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        evaluator = RAGASEvaluator(pipeline, llm)
        queries, expected_answers = load_test_data(args.test_data_path)
        asyncio.run(pipeline.hierarchical_rag.encode_pdf_hierarchical())
        results = asyncio.run(evaluator.generate_answers(queries))
        scores = evaluator.evaluate(results, expected_answers)
        print("\nEvaluation Scores:")
        print(scores)
    elif args.query:
        # Run the pipeline with a single query
        asyncio.run(pipeline.run(args.query))
    else:
        print("Please provide a query or use the --evaluate flag.")
