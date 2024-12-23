import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

def get_embedding_function():
    # Replace 'your-api-key' with your actual OpenAI API key
    openai_api_key = os.getenv('OPENAI_API_KEY')
    
    # Create the OpenAIEmbeddings instance with your API key
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, base_url=os.getenv('OPENAI_BASE_URL'), model='codesmart.embedding')
    
    return embeddings