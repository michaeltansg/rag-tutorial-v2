# rag-tutorial-v2

# About the project
This project comes with 2 PDF documents found in the data folder. These are used for the RAG example. The vector database is a local Chroma database. 

# Setting Up the Virtual Environment and Installing Dependencies
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuring Environment Variables
1. Create a .env file by copying the .env.example file:
```bash
cp .env.example .env
```
2. Open the newly created .env file and adjust the environment variables as needed.

## Document ingestion
Begin by creating embeddings of the document using the following command: `python populate_database.py`

If you change the embedding model, you need to reset the database, since different embedding models have different dimensions. Even if the dimensions are the same (eg: for the same embedding model, but newer version), it is necessary to reset the database with this command: `python populate_database.py --reset`.

## Command to run Deepeval
`deepeval test run test_rag.py`

# Configure Local Model
https://docs.confident-ai.com/docs/metrics-introduction#using-local-llm-models


# Example test
```
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
from deepeval.metrics import AnswerRelevancyMetric

def test_answer_relevancy():
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output of your LLM application
        actual_output="We offer a 30-day full refund at no extra cost.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra cost."]
    )
    assert_test(test_case, [answer_relevancy_metric])
```
