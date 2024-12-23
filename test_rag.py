from main import query_rag
from langchain_community.llms.ollama import Ollama
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_rules():
    test_case = query_and_generate_test_case(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
        seed="fp_5f20662549"
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=1.0)
    assert_test(test_case, [answer_relevancy_metric])

# def test_monopoly_ticket_ride_combination():
#     test_case = query_and_generate_test_case(
#         question="how much money I should earn from Monopoly game to travel in train to go to Phileas?",
#         expected_response="$1500",
#     )
#     answer_relevancy_metric = AnswerRelevancyMetric(threshold=1.0)
#     hallucination_metric = HallucinationMetric()
#     faithfulness_metric = FaithfulnessMetric(threshold=1.0)
#     assert_test(test_case, [answer_relevancy_metric, hallucination_metric, faithfulness_metric])

# Q1: "how much money I should earn from Monopoly game to travel in train to go to Phileas?",
# Q2: on average, How much do hotels cost in Japan?
# A: The provided context is about rules for buying and erecting hotels in a board game, not about the real-world cost of hotels in Japan. Therefore, the context does not offer any information regarding the average cost of hotels in Japan.


def test_ticket_to_ride_rules():
    test_case = query_and_generate_test_case(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])


def query_and_generate_test_case(question: str, expected_response: str, seed: str = None) -> LLMTestCase:
    response_text, context = query_rag(question)
    return LLMTestCase(
        input=question,
        actual_output=response_text,
        expected_output=expected_response,
        retrieval_context=context
    )
    
    
    
    # prompt = EVAL_PROMPT.format(
    #     expected_response=expected_response, actual_response=response_text
    # )

    # model = Ollama(model="mistral")
    # evaluation_results_str = model.invoke(prompt)
    # evaluation_results_str_cleaned = evaluation_results_str.strip().lower()

    # print(prompt)

    # if "true" in evaluation_results_str_cleaned:
    #     # Print response in Green if it is correct.
    #     print("\033[92m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
    #     return True
    # elif "false" in evaluation_results_str_cleaned:
    #     # Print response in Red if it is incorrect.
    #     print("\033[91m" + f"Response: {evaluation_results_str_cleaned}" + "\033[0m")
    #     return False
    # else:
    #     raise ValueError(
    #         f"Invalid evaluation result. Cannot determine if 'true' or 'false'."
    #     )
