import argparse
import os
from dotenv import load_dotenv
from langchain_community.vectorstores.chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from EvaluatedRAGChain import EvaluatedRAGChain
from get_embedding_function import get_embedding_function
from langfuse import Langfuse
import asyncio
from langchain.chains import LLMChain

load_dotenv()


# To access Synthetic data Uncomment below line and comment chroma. Make sure You run test_goldens_synthesizer.py for populating the data.
# CHROMA_PATH = ".vector_db"
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


def query_rag(query_text: str, seed: str = None):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    score_threshold: float = 0.5
    filtered_results = [
        (doc, score) for doc, score in results if score >= score_threshold
    ]

    context_arr = [doc.page_content for doc, _score in filtered_results]
    context_text = "\n\n---\n\n".join(
        [doc.page_content for doc, _score in filtered_results]
    )
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.invoke({"context": context_text, "question": query_text})
    print(f"prompt: {prompt.to_string()}")

    chat_model = ChatOpenAI(
        openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("INFERENCE_MODEL", "gpt-3.5-turbo"),
    )

    response_text: AIMessage = chat_model.invoke(input=prompt)

    asyncio.run(evaluation(chat_model=chat_model, 
                           prompt_template=prompt_template, 
                           query_text=query_text, 
                           context_text=context_text))

    sources = [doc.metadata.get("id", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text.content, context_arr

async def evaluation(chat_model, prompt_template, query_text, context_text):
    # Initialising Langfuse
    langfuse_client = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY")
    )
    # langfuse & DeepEval integration
    llm_chain = LLMChain(llm=chat_model, prompt=prompt_template)
    evaluated_chain = EvaluatedRAGChain(llm_chain, langfuse_client=langfuse_client)
    result = evaluated_chain.run_with_evaluation(
        query=query_text,
        context=context_text
    )
    print(f"Response: {result['response']}")
    print(f"Evaluation scores: {result['evaluation']}")
    print(f"Trace ID: {result['trace_id']}")

if __name__ == "__main__":
    main()