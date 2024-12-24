from deepeval.evaluate import EvaluationResult
from main import query_rag
from deepeval import assert_test
from deepeval.metrics import AnswerRelevancyMetric, HallucinationMetric, FaithfulnessMetric, SummarizationMetric
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



# metric_mapping = {
#     "answer_relevancy": lambda config: AnswerRelevancyMetric(**config),
#     "summarization": lambda config: SummarizationMetric(**config),
#     "hallucination": HallucinationMetric,
#     "faithfulness": FaithfulnessMetric,
# }


# def get_metrics(test_case_data):
#     metrics = []
#     for metric_data in test_case_data["metrics"]:
#         metric_name = metric_data["name"]
#         metric_config = metric_data.get("config", {})  # Get config if available
#         metric_class = metric_mapping.get(metric_name)
#         if metric_class is None:
#             raise ValueError(f"Unknown metric: {metric_name}")
#         if callable(metric_class):
#             metric = metric_class(metric_config)  # Instantiate with config
#         else:
#             metric = metric_class()  # Instantiate without config
#         metrics.append(metric)
#     return metrics


# test_cases = []
# metrics: list = []
# for item in data:
#   test_cases.append(LLMTestCase(input=item['input'], actual_output=item['actual_output'], context=item['context'] if item['context'] is not None else None))
#   metrics.append(item['metrics'] if get_metrics(test_case_data=item['metrics']) is not None else None )

# dataset = EvaluationDataset(test_cases=test_cases)

# results: EvaluationResult = dataset.evaluate(
#     metrics=get_metrics(test_case_data=data)
# )
# Findings
# 1. Cannot use different mertics in a json file
# 2. 

