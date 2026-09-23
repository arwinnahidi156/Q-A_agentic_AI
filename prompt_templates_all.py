"""
Prompt Teplates and Messages in LangChain V.1
"""

from dotenv import load_dotenv
from langchain_core.prompts import (ChatPromptTemplate,
                                     FewShotChatMessagePromptTemplate,
                                     MessagesPlaceholder)
from langchain_core.messages import (SystemMessage, HumanMessage, AIMessage)
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
import os

load_dotenv()
set_llm_cache(InMemoryCache())

def get_model(name: str = "openai/gpt-5.6-luna-pro",
              temperature: float = 0.2,
              max_tokens: int = 1500,
              timeout: int = 60,
              max_retries: int = 3,
              cache: bool = True,
              streaming: bool = True):
    return init_chat_model(
        model=name,
        model_provider="openai",   # Kaya serves everything in OpenAI format
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        max_retries=max_retries,
        cache=cache,
        streaming=streaming,
    )

def get_model_with_fallback(**kwargs):
    """
    Try models in order. If one fails (403, 429, 500, timeout...),
    LangChain automatically retries with the next one.
    """
    candidates = [
        "openai/gpt-5.6-luna-pro",
        "deepseek/deepseek-v4-pro-0813",   # <- your working model
        "meta/muse-spark-1.3-contributor",
                  
    ]
    models = [get_model(name, **kwargs) for name in candidates]
    return models[0].with_fallbacks(models[1:])

model=get_model_with_fallback()

def demo_basic_templates():
    """
    Basic ChatPromptTemplate usage
    """

    simple = ChatPromptTemplate.from_template("translate {text} to {language}")

    messages=simple.format_messages(text="Hello, how are you?", language="Italian")
    print("Simple template : ")
    print(f"{messages}")

    multi=ChatPromptTemplate.from_messages(
        [
            ("system", "You are a translator . be consise and accurate."),
            ("human", "Translate the following text : {text} to {language}"),
        ]
    )

    messages=multi.format_messages(text="good morning", language="Italian")
    print("\nMulti message template : ")
    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content}")


def demo_message_types():
    """
    Demonstrate different message types in LangChain
    """

    messages=[
        SystemMessage(content="You are a math tutor . Be brief"),
        HumanMessage(content="what's 5 * 5 ?"),
        AIMessage(content="25"),
        HumanMessage(content="and if i add 10 ?")
    ]

    print("\nDemonstrating different message types : ")
    for msg in messages:
        print(f"{type(msg).__name__}: {msg.content}")

    model=get_model_with_fallback()
    response=model.invoke(messages)
    print(f"Conversation result : {response.content}")


def demo_messages_placeholder():
    """
    Use MessagePlaceholder for dynamic conversation history.
    """

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="history"),
            ("human","{question}")
        ]
    )

    history = [
        HumanMessage(content="my name is Paulo"),
        AIMessage(content="nice to meet you Paulo!")
    ]


    messages=prompt.format_messages(history=history,question="what's my name ?")

    print("with history placeholder : ")
    for msg in messages : 
        print(f"{type(msg).__name__} : {msg.content[:50]}...")


    model=get_model_with_fallback()
    response=model.invoke(messages)
    print(f"\nResponse : {response.content}")


def demo_few_shot():
    """
    Few-shot prompting with examples.
    """


    # Define examples
    examples = [
        {"word":"happy", "opposite":"sad"},
        {"word":"fast","opposite":"slow"},
        {"word":"big","opposite":"small"},
                
    ]

    # template for each example 
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human","what's the opposite of '{word}'?"),
            ("ai","the opposite of '{word}' is '{opposite}'.")
        ]
    )

    # few-shot wrapper 
    few_shot= FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples
    )

    final_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',"yougvet opposite of words. follow the examples"),
            few_shot,
            ('human',"what's the opposite of '{word}' ?")
        ]
    )

    # test 
    model=get_model_with_fallback()
    chain=final_prompt | model

    response=chain.invoke({"word":"bright"})
    print(f"Few-shot results: {response.content}")


def demo_prompt_composition():
    """
    compose prompts from reusable parts.

    """
    # reusable system prompt
    persona=ChatPromptTemplate.from_messages(
        [
            ("system","you are a {role}. your tone is {tone}.")

        ]
    )

    # reusable task prompt 
    task=ChatPromptTemplate.from_messages([('human',"{task}")])

    # combine
    full_prompt = persona + task 

    # test different combinations 
    model = get_model_with_fallback()
    chain= full_prompt | model

    # as a teacher 
    response = chain.invoke(
        {
            "role":"teacher",
            "tone":"adventurous",
            "task":"tell about langchain "
        }
    )

    print(f"teacher : {response.content}")


    # as a scientist 
    response = chain.invoke(
        {
            "role":"scientist",
            "tone":"precise and academic",
            "task":"tell why lamgchain for agentic AI is good and why not another option ? "
        }
    )

    print(f"scientist : {response.content}")


if __name__ == "__main__":
    print("="*50)
    print("Demo 1 : basic templates")
    print("="*50)
    demo_basic_templates()

    print("="*50)
    print("Demo 2 : Message types")
    print("="*50)
    demo_message_types()

    print("="*50)
    print("Demo 3 : MessagePlaceholder")
    print("="*50)
    demo_messages_placeholder()

    print("="*50)
    print("Demo 4 : Few-shots")
    print("="*50)
    demo_few_shot()

    print("="*50)
    print("Demo 5 : prompt_composition")
    print("="*50)
    demo_prompt_composition()

    




