from main import query_rag
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, FaithfulnessMetric, SummarizationMetric
from deepeval.test_case import LLMTestCase

EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Does the actual response match the expected response? 
"""


def test_monopoly_max_players():
    test_case = query_and_generate_test_case(
        question="Can you tell maximum how many players can play the monopoly game?",
        expected_response="The maximum number of players that can play the Monopoly game is 8.",
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=1.0)
    summary_metric = SummarizationMetric(
        assessment_questions=[
            "Is the coverage score based on a percentage of 'yes' answers?",
            "Does the score ensure the summary's accuracy with the source?",
            "Does a higher score mean a more comprehensive summary?"
        ]
    )
    assert_test(test_case, [answer_relevancy_metric, summary_metric])


def test_monopoly_rules():
    test_case = query_and_generate_test_case(
        question="How much total money does a player start with in Monopoly? (Answer with the number only)",
        expected_response="$1500",
        seed="fp_5f20662549"
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])


def test_ticket_to_ride_rules():
    test_case = query_and_generate_test_case(
        question="How many points does the longest continuous train get in Ticket to Ride? (Answer with the number only)",
        expected_response="10 points",
    )
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])


def test_monopoly_ticket_ride_combination():
    test_case = query_and_generate_test_case(
        question="how much money I should earn from Monopoly game to travel in train to travel around the world?",
        expected_response='The context provided does not mention "Phileas" or any specific monetary goal related to traveling by train. The rules given focus on gameplay mechanics for Monopoly, including the use of dice, purchasing property, and utilizing the Speed Die. They do not specify the amount of money needed for travel to any external location, such as "Phileas." Thus, it\'s not possible to answer your question based solely on the provided information.',
    )
    
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=1.0)
    hallucination_metric = HallucinationMetric()
    faithfulness_metric = FaithfulnessMetric()

    assert_test(test_case=test_case, metrics=[answer_relevancy_metric, hallucination_metric, faithfulness_metric])


def query_and_generate_test_case(question: str, expected_response: str) -> LLMTestCase:
    response_text, context = query_rag(question)
    print(context)
    return LLMTestCase(
            input=question,
            actual_output=response_text,
            expected_output=expected_response,
            context=context,
            retrieval_context=context
        )
    