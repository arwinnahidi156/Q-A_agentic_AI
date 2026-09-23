import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage , AIMessage , ChatMessage , ToolMessage , FunctionMessage , BaseMessage


load_dotenv()

prompt = ChatPromptTemplate.from_template("tell me a {adjective} joke about {topic}")

message=prompt.format_messages(adjective="funny", topic="RAG")

print(message)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that translates {input_language} to {output_language}."),
        ("human", "Translate the following text : {text}") ,


    ]
)

message=prompt.format_messages(input_language="English", output_language="French", text="i love coding.")

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
model=get_model()

# response=model.invoke(message)
# print(response.content)



# messages=[
#     SystemMessage(content="You are a helpful assistant that translates English to French."),
#     HumanMessage(content="Translate the following text: I love coding."),
#     AIMessage(content="J'adore coder."),
#     ToolMessage(content="This is a tool message."),
#     FunctionMessage(content="This is a function message."),
#     ChatMessage(content="This is a chat message."),

# ]

from langchain_core.prompts import FewShotChatMessagePromptTemplate

examples=[
    {"input": "happy", "output": "sad"},
    {"input": "good", "output": "bad"},

]
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples
)

final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "give the opposite of each word."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

model=get_model()
# response=model.invoke(final_prompt.format_messages(input="happy"))
# print(response.content)


# reusable components 
system_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a {role}")
    ]
)

user_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{question}")
    ]
)

full_prompt = system_prompt + user_prompt
fin = full_prompt.format_messages(role="helpful assistant", question="What is the capital of France?")

print(fin)