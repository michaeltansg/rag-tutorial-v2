from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset


dataset = EvaluationDataset()

# Add as test cases
dataset.add_test_cases_from_json_file(
    file_path="test/dataset_source.json",
    input_key_name="input",
    actual_output_key_name="actual_output",
    expected_output_key_name="expected_output",
    context_key_name="context",
    retrieval_context_key_name="retrieval_context",
)

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
results = dataset.evaluate(metrics=[answer_relevancy_metric])