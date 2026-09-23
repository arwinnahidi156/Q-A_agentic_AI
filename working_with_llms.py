"""
working with LLMs in LangChain V.1
Multiple providers , configuration , streaming and cost optimization
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage
import os

load_dotenv()


def get_model(name: str = "openai/gpt-5.6-luna-pro",
              temperature: float = 0.2,
              max_tokens: int = 1500,
              timeout: int = 60,
              max_retries: int = 3,cache: bool = True,streaming: bool = True):
    return init_chat_model(
        model=name,
        model_provider="openai",   # Kaya همه‌چیز را با فرمت OpenAI سرو می‌کند
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        
    )


def demo_init_chat_model(name: str = "openai/gpt-5.6-luna-pro", temperature: float = 0.2,max_tokens: int = 1500,timeout: int = 60,max_retries: int = 3,cache: bool = True,streaming: bool = True):
    chat_model = init_chat_model(
        model=name,
        model_provider="openai",   # برای مدل‌های Anthropic عوضش کن
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        

    )
    response = chat_model.invoke("What is the capital of France?")
    print(f"Response content from the model: {response.content}")

def demo_model_comparison():
    """Compare models from different providers on the same prompt"""
    prompt = "explain recursion in one sentence"

    models = {
        "openai/gpt-5.6-luna-pro": get_model("openai/gpt-5.6-luna-pro"),
        "anthropic/claude-sonnet-5": get_model("anthropic/claude-sonnet-5"),
    }

    print(f"Prompt: {prompt}\n")
    for model_name, model in models.items():
        print(f"--- {model_name} ---")
        for chunk in model.stream(prompt):
            print(chunk.content, end="", flush=True)
        print("\n")

def demo_message():
    """Demonstrate the use of messages with a chat model"""
    model = get_model("openai/gpt-5.6-luna-pro")
    # using message objects (more control over roles)
    messages = [
        SystemMessage(content="You are a political analyst. Answer in a neutral tone."),
        HumanMessage(content="What is the reason for the current war between USA and Iran?"),
    ]

    # print("using message objects (more control over roles) : ")
    # print(f"messages : {messages[0]} | {messages[1]}")

    response = model.invoke(messages)
    print(f"Response content from the mode: {response.content}")

    # multi-turn conversation using message objects
    messages.append(response)
    messages.append(HumanMessage(content="What is the solution for this war?"))
    print("\nMulti-turn conversation : ")
    response2 = model.invoke(messages)
    print(f"Response content from the mode: {response2.content}")
    print('follow up')


def demo_model_comparison_without_streaming():
    """Compare models from different providers on the same prompt"""
    prompt = "tell me about the future of agentic AI for new comers in one sentence"

    models = {
        "openai/gpt-5.6-luna-pro": get_model("openai/gpt-5.6-luna-pro", streaming=False),
        "anthropic/claude-sonnet-5": get_model("anthropic/claude-sonnet-5", streaming=False),
    }



    print(f"Prompt: {prompt}\n")
    for model_name, model in models.items():
        print(f"--- {model_name} ---")
        try:
            response = model.invoke(prompt)
            print(f"{response.content}\n")
        except Exception as e:
            print(f"⚠️ failed: {e}\n")


def exercise_multi_model(question,model_names):
    """
    EXERCISE : create a function that : 
    1.Takes a question and a list of model names 
    2. Gets responses from all models 
    3. Returns a dict of {moedel_name : response}
    """

    models={}
    for model in model_names : 
        models[model]=get_model(model)
    
    responses={}
    for model_name , model in models.items():
        try : 
            response = model.invoke(question)
            responses[model_name]=response.content
        

        except Exception as e :
            print(f"failed : {e}\n")

    return responses





if __name__ == "__main__":
    # demo_init_chat_model()
    # demo_model_comparison()
    # demo_message()
    # demo_model_comparison_without_streaming()
    responses=exercise_multi_model("what is a RAG ?", model_names=["openai/gpt-5.6-luna-pro","meta/muse-spark-1.3-contributor","deepseek/deepseek-v4-pro-0813"])
    for model,content in responses.items():
        print(f"{model} : {content}")

