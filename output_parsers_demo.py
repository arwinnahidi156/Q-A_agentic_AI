from dotenv import load_dotenv
from langchain_core.prompts import (ChatPromptTemplate,
                                     FewShotChatMessagePromptTemplate,
                                     MessagesPlaceholder)
from langchain_core.messages import (SystemMessage, HumanMessage, AIMessage)
from langchain_core.output_parsers import StrOutputParser,PydanticOutputParser,JsonOutputParser
from pydantic import BaseModel,Field
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

parser = StrOutputParser()
model=get_model_with_fallback()
# prompt = ChatPromptTemplate.from_messages([
#     ("human","write a short poem about {topic}.")
#     ]
#     )
prompt = ChatPromptTemplate.from_template("write a short poem about {topic}.")
chain = prompt | model | parser 

# response= chain.invoke({"topic": "iran"})

# print(type(response))
# print(isinstance(response,str))
# print(str.upper(response))

# parser=JsonOutputParser()
# prompt = ChatPromptTemplate.from_template(
#     "return a JSON object with 'name' and 'age' for : {description}. "
# )
# chain = prompt | model | parser
# result = chain.invoke({"description" : " a 25-years-old developer named Alex."})
# print(result)
# print(type(result))

# class Person(BaseModel):
#     name : str = Field(description="The person's name ")
#     age : int = Field(description="The person's age ")
#     occupation : str = Field(description="The person's occupation")


# parser=PydanticOutputParser(pydantic_object=Person)
# model=get_model_with_fallback()
# prompt=ChatPromptTemplate.from_template("return a JSON object with 'name', 'age', and 'occupation' for : {description}"
#                                         ).partial(format_instructions=parser.get_format_instructions())
# chain = prompt | model | parser

# result=chain.invoke(
#     {"description": "A 30-years-old artist named Maria"}
# )
# print(result)


    
    
    
    
class MovieReview(BaseModel):
    title : str = Field(description="The movie's title")
    review : str = Field(description="The movie's review")
    rating : int = Field(description="The movie's rating")
model=get_model_with_fallback()
# bind the schema to the model 
structured_model = model.with_structured_output(MovieReview)
result=structured_model.invoke("Review: Inception is a mind-bending thriller . 9/10 ")
print(result)
