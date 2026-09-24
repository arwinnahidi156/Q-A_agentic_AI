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
from typing import List, Optional

load_dotenv()
# set_llm_cache(InMemoryCache())



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

# model=get_model_with_fallback()
model = get_model(name="deepseek/deepseek-v4-pro-0813", cache=False, streaming=False)

def demo_str_parser():
    """
    Basic string output parser.
    """
    
    prompt = ChatPromptTemplate.from_template(
        "give me a one-word answer : what color is the sky ?"
    )
    
    parser = StrOutputParser()
    
    chain = prompt | model | parser
    result = chain.invoke()
    print(f"Result : '{result}' (type : {type(result).__name__})")
    

def demo_json_parser():
    """
    Basic JSON output parser.
    """
    
    prompt = ChatPromptTemplate.from_template(
        "return a JSON object with keys 'city' and 'country' for : {place}\n"
        "return ONLY valid JSON , no explanations"
    )
    
    parser = JsonOutputParser()
    
    chain = prompt | model | parser
    result = chain.invoke({"place": "the statue of liberty"})
    print(f"Result : {result}")
    print(f"City : {result['city']} , Country : {result['country']}")
    
def demo_pydantic_parser():
    """
    pydantic output parser for  type-safe structured data. 
    """
    
    class Recipe(BaseModel):
        name: str = Field(description="The name of the recipe")
        ingredients: List[str] = Field(description="The ingredients of the recipe")
        prep_time_minutes: int = Field(description="The prep time of the recipe in minutes")
        difficulty: str = Field(description="The difficulty of the recipe . can be easy , medium or hard")
        
    parser = PydanticOutputParser(pydantic_object=Recipe)
    
    prompt = ChatPromptTemplate.from_template(
        "Create  simple recipe for : {dish}\n\n{format_instructions}"
    ).partial(format_instructions=parser.get_format_instructions())
    
    chain=prompt | model | parser
    
    result = chain.invoke({"dish": "a simple recipe for chicken parmesan"})
    print(f"Recipe : {result.name}")
    print(f"Ingredients : {result.ingredients}")
    print(f"Prep time : {result.prep_time_minutes} minutes")
    print(f"Difficulty : {result.difficulty}")
    
    # type-safe access 
    
    print(
        f"\nType check - prep_time is int : {isinstance(result.prep_time_minutes,int)}"
    )
    print(
        f"\nType check - difficulty is str : {isinstance(result.difficulty,str)}"
    )
    
    
def demo_structured_output():
    """
    Modern with_structured_output() method.
    """
    
    class TaskExtraction(BaseModel):
        """Extract task information.
        """
        
        task: str = Field(description="The main task to do")
        priority: str = Field(description="The priority of the task. high or low or medium")
        assignee: Optional[str] = Field(description="The assignee of the task if mentioned")
        deadline: Optional[str] = Field(description="The deadline of the task . if mentioned")
    
    # bind schema to the model
    structured_model = model.with_structured_output(TaskExtraction)
    
    prompt = ChatPromptTemplate.from_template(
        "Extract task information from the following text : {text}\n"
        "return ONLY valid JSON , no explanations"
    )
    
    chain = prompt | structured_model
    
    texts = [
        "john needs to finish the report by tomorrow - it's urgent",
        "we should update the docs sometime this week",
        "critical: fix the login bug ASAP",
        
    ]
    
    print(f"Extracted tasks :")
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"Task : {result.task}")
        print(f"Priority : {result.priority}")
        print(f"Assignee : {result.assignee}")
        print(f"Deadline : {result.deadline}")
        
def demo_complex_schema():
    """
    Complex nested schema with structured output.
    """
    
    class Address(BaseModel):
        street: str = Field(description="The street of the address")
        city: str = Field(description="The city of the address")
        state: str = Field(description="The state of the address")
        zip_code: str = Field(description="The zip code of the address")
        
    class Company(BaseModel):
        name: str = Field(description="The name of the company")
        industry: str = Field(description="The industry of the company")
        employee_count : int = Field(description="The number of employees at the company")
        headquarters : Address = Field(description="The headquarters of the company")
        products : List[str] = Field(description="The products offered by the company")
        
    structured_model = model.with_structured_output(Company)
    
    
    prompt = ChatPromptTemplate.from_template(
        "Extract company information from the following text : {text}\n"
        "return ONLY valid JSON , no explanations"
    )
    
    chain = prompt | structured_model
    
    texts = [
        "The company is called Google and it is located in Mountain View, California. It has 1,000,000 employees and is the largest company in the world. Their headquarters is in Mountain View, California.",
        "The company is called Microsoft and it is located in Redmond, Washington. It has 500,000 employees and is the second largest company in the world. Their headquarters is in Redmond, Washington.",
        "The company is called Amazon and it is located in Seattle, Washington. It has 300,000 employees and is the third largest company in the world. Their headquarters is in Seattle, Washington.",
    ]
    
    print(f"Extracted companies :")
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"Company : {result.name}")
        print(f"Industry : {result.industry}")
        print(f"Employee count : {result.employee_count}")
        print(f"Headquarters : {result.headquarters.street}, {result.headquarters.city}, {result.headquarters.state}, {result.headquarters.zip_code}")        
        print(f"Products : {result.products}")      
        
    
def exercise_structured_extraction():
    """
    Exercise : create a schema and chain that extracts : 
    - movie title 
    - year of release
    - director
    - main actors(list)
    -genre
    -rating (1-10)
    
    test with a movie description "
    
    """
    
    
    class Movie(BaseModel):
        title : str = Field(description="The movie's title")
        year : int = Field(description="The movie's release year")
        director : str = Field(description="The movie's director")
        actors : List[str] = Field(description="The movie's main actors")
        genre : str = Field(description="The movie's genre")
        rating : int = Field(description="The movie's rating. from 1 to 10", ge=1,le=10)
        
    structured_model = model.with_structured_output(Movie)
    prompt = ChatPromptTemplate.from_template(
        "Extract movie information from the following text : {text}\n"
        "return ONLY valid JSON , no explanations"
    )
    chain = prompt | structured_model
    texts = [
        "Inception is a mind-bending thriller directed by Christopher Nolan. It stars Leonardo DiCaprio, Joseph Gordon-Levitt, Ellen Page, Tom Hardy, and Joseph Gordon-Levitt. The movie was released in 2010. ir's a great movie and 9/10",
        "The Dark Knight is an action-thriller directed by Christopher Nolan. It stars Christian Bale, Heath Ledger, and Michael Caine. The movie was released in 2012.good movie but not so good and a straight up 8/10",
        "The Matrix is a science-fiction action-thriller directed by the Wachowskis. It stars Keanu Reeves, Laurence Fishburne, Carrie-Anne Moss, Hugo Weaving, and Laurence Fishburne. The movie was released in 1999. it's awesome and my all time favorite and 10/10",
    ]
    
    print(f"Extracted movies :")
    for text in texts:
        result = chain.invoke({"text": text})
        print(f"Movie : {result.title}")
        print(f"Year : {result.year}")
        print(f"Director : {result.director}")
        print(f"Actors : {result.actors}")
        print(f"Genre : {result.genre}")
        print(f"Rating : {result.rating}")
        

if __name__ == "__main__":
    # demo_str_parser()
    # demo_json_parser()
    # demo_pydantic_parser()
    # demo_structured_output()  
    # demo_complex_schema()
    exercise_structured_extraction()