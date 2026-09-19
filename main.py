from dotenv import load_dotenv
from importlib.metadata import version
load_dotenv()

core_version = version("langchain-core")
lg_version = version("langgraph")
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
import os


def main():
    print("Hello, World!")
    print(f"LangChain Core Version: {core_version}")
    print(f"LangGraph Version: {lg_version}")
    # Initialize OpenAI LLM
    llm = ChatOpenAI(
    model="openai/gpt-5.6-luna-pro",
    api_key=os.environ["KAYA_API_KEY"],
    base_url="https://kayaai.ir/api",
    temperature=0.2,
    )
    print(f"OpenAI LLM initialized with model: {llm.model}")
    # response = llm.invoke("Say 'setup complete !' in one word")
    # print(f"Response from OpenAI LLM: {response}")

if __name__ == "__main__":
    main()