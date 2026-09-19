from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os


load_dotenv()

def demo_basic_chain():
    """ Demonstrate a basic chain using LCEL and Runnables """
    # Component 1 : define the prompt template using LCEL
    prompt=ChatPromptTemplate.from_template("you are a helpful assistant . answer in one sentence : {question}") 

    model = ChatOpenAI(
        model="openai/gpt-5.6-luna-pro",
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=0.2,
    )

    parser = StrOutputParser()

    # Component 2 : compose with pipe operator
    chain = prompt | model | parser

    # Component 3 : execute the chain with an input
    result = chain.invoke({"question": "What is langchain ?"})
    print(f"Result from the chain: {result}")

    return chain

if __name__ == "__main__":
    demo_basic_chain()
