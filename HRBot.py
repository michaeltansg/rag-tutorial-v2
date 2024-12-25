from openai import OpenAI, AsyncOpenAI
from deepeval.models import DeepEvalBaseLLM
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

class HRBot(DeepEvalBaseLLM):

    def __init__(self, model_name = None, *args, **kwargs):
        super().__init__(model_name, *args, **kwargs)
        load_dotenv()

    # Load the model
    def load_model(self):
        return OpenAI()

    # Generate responses using the provided user prompt
    def generate(self, prompt: str) -> str:
        client = self.load_model()
        chat_model = ChatOpenAI(
            openai_api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model=os.getenv("INFERENCE_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ]
        )
        response = chat_model.invoke(input=prompt)
        return response.content

    # Async version of the generate method
    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    # Retrieve the model name
    def get_model_name(self) -> str:
        return self.model_name

    ##########################################################################
    # Optional:  Define the system prompt for the financial advisor scenario #
    ##########################################################################

    def get_system_prompt(self) -> str:
        return (
            "You are HR Bot, a Policy advisor bot. Your task is to provide policy & benafits information "
            "recommendations based on the company's data. Always prioritize user privacy."
        )