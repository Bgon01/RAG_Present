# src/ragas_evaluation.py

import pandas as pd
from tqdm import tqdm
from ragas import evaluate_runs, RagasEvaluator
from langchain.schema import Document
from langchain.chat_models import ChatOpenAI
from main import CombinedRAGPipeline  # Import your pipeline
import os

class RAGASEvaluator:
    def __init__(self, pipeline: CombinedRAGPipeline, llm: ChatOpenAI):
        self.pipeline = pipeline
        self.llm = llm

    async def generate_answers(self, queries):
        results = []
        for query in tqdm(queries, desc="Generating answers"):
            # Run the pipeline for each query
            retrieved_docs = await self.pipeline.process_query(query)
            # Combine the content of retrieved documents
            context = "\n".join([doc.page_content for doc in retrieved_docs])
            # Generate an answer using the LLM
            prompt = f"Answer the question: '{query}'\n\nBased on the following context:\n{context}"
            answer = self.llm.predict(prompt)
            results.append({
                "query": query,
                "answer": answer,
                "context": context
            })
        return results

    def evaluate(self, results, expected_answers):
        # Prepare data for RAGAS
        runs = []
        for res, expected in zip(results, expected_answers):
            runs.append({
                "query": res["query"],
                "response": res["answer"],
                "ground_truths": [expected],
                "contexts": [res["context"]]
            })

        # Evaluate using RAGAS
        evaluator = RagasEvaluator()
        scores = evaluator.evaluate_runs(runs)
        return scores

def load_test_data(filepath):
    df = pd.read_csv(filepath)
    queries = df['query'].tolist()
    expected_answers = df['expected_answer'].tolist()
    return queries, expected_answers
