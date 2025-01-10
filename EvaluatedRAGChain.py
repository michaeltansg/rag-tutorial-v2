from typing import Any, Dict
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from langchain.chains.llm import LLMChain
from langfuse import Langfuse
from langfuse.client import StatefulTraceClient

class EvaluatedRAGChain:
    def __init__(self, llm_chain: LLMChain, langfuse_client: Langfuse):
        self.llm_chain = llm_chain
        self.langfuse_client = langfuse_client
        # self.evaluator = LLMEvaluator()
        
    async def evaluate_response(self, response: str, context: str, query: str) -> Dict[str, Any]:
        """
        Evaluate the response using deepeval metrics and return scores
        """
        # Initialize metrics
        hallucination_metric = HallucinationMetric(
            threshold=0.7
        )
        
        relevancy_metric = AnswerRelevancyMetric(
            threshold=0.7
        )
        
        context=[context]

        actual_output=response

        test_case = LLMTestCase(
            input=query,
            actual_output=actual_output,
            context=context
        )
        relevancy_metric.measure(test_case=test_case)
        hallucination_metric.measure(test_case=test_case)
        
        return {
            "hallucination_score": hallucination_metric.score,
            "relevancy_score": relevancy_metric.score,
            "hallucination_reason": hallucination_metric.reason,
            "relevancy_reason": relevancy_metric.reason
        }
    
    async def run_with_evaluation(self, query: str, context: str) -> Dict[str, Any]:
        """
        Run the RAG chain with evaluation and log to Langfuse
        """
        
        # Create a new trace
        trace: StatefulTraceClient = self.langfuse_client.trace(
            name="query_rag",
            metadata={"query": query, "context": context},
            input={"query": query}
        )
        
        try:
            # Run the LLM chain
            generation_span = trace.span(
                name="generation",
                metadata={"context": context},
                input={"query": query}
            )
            
            response = await self.llm_chain.ainvoke(
                input={"context": context, "question": query},
            )
            
            generation_span.end()
            
            # Evaluate the response
            evaluation_span = trace.span(name="evaluation")
            evaluation_results = await self.evaluate_response(
                response=response,
                context=context,
                query=query
            )
            evaluation_span.end()
            
            # Update trace with scores
            trace.score(
                name="hallucination_score",
                value=evaluation_results["hallucination_score"],
                comment=evaluation_results["hallucination_reason"]
            )
            trace.score(
                name="relevancy_score",
                value=evaluation_results["relevancy_score"],
                comment=evaluation_results["relevancy_reason"]
            )
            trace.update(
                output={response["text"]},
                metadata={
                    "hallucination_reason": evaluation_results["hallucination_reason"],
                    "relevancy_reason": evaluation_results["relevancy_reason"]
                }
            )
            
            
            return {
                "response": response,
                "evaluation": evaluation_results,
                "trace_id": trace.id
            }
            
        except Exception as e:
            print(e)
            raise
        finally:
            print("Complete tracing")
            self.langfuse_client.flush()