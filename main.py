import argparse
from importlib import metadata
import json
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI
from enum import Enum

from get_embedding_function import get_embedding_function

load_dotenv()

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)


def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    score_threshold: float = 0.5
    filtered_results = [(doc, score) for doc, score in results if score >= score_threshold]

    context_arr = [doc.page_content for doc, _score in filtered_results]
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in filtered_results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = OpenAI(base_url=os.getenv('OPENAI_BASE_URL'), api_key=os.getenv('OPENAI_API_KEY'))
    # response_text = model.invoke(prompt)
    response_text = model.completions.create(
        model="codesmart.ide",
        prompt=prompt,
        user="sathishkumar.gunasekaran@scbtechx.io"
    )

    # answer = json.dumps(response_text, indent=4)
    # print(answer)
    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text.choices[0].text, context_arr


if __name__ == "__main__":
    main()
