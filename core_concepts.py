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


def demo_batch_execution():
    """ Demonstrate batch execution for multiple inputs of a chain using LCEL and Runnables """
    # Component 1 : define the prompt template using LCEL
    prompt=ChatPromptTemplate.from_template("Translate to italian : {text} ") 

    model = ChatOpenAI(
        model="openai/gpt-5.6-luna-pro",
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=0.2,
    )

    parser = StrOutputParser()

    # Component 2 : compose with pipe operator
    chain = prompt | model | parser

    # Component 3 : execute the chain with a batch of inputs
    inputs = [
        {"text": "i would like to go abroad."},
        {"text": "the sample has been tested successfully."},
        {"text": "i need a part time job in italy to aford my living expenses."},
    ]
    results = chain.batch(inputs)
    for text in zip (inputs, results):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")
    return chain

def demo_streaming():
    """ Demonstrate streaming output for a chain using LCEL and Runnables """
    # Component 1 : define the prompt template using LCEL
    prompt=ChatPromptTemplate.from_template("write a poem in italian about {topic} in 4 lines ") 

    model = ChatOpenAI(
        model="openai/gpt-5.6-luna-pro",
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=0.2,
        streaming=True,
    )

    parser = StrOutputParser()

    # Component 2 : compose with pipe operator
    chain = prompt | model | parser

    # Component 3 : execute the chain with an input
    print("Streaming output:")
    for chunk in chain.stream({"topic": input("Enter a topic for the poem: ")}):
        print(chunk, end="", flush=True)
    print("\nStreaming complete.")

    return chain



def demo_schema_inspection():
    """ Demonstrate schema inspection for a chain using LCEL and Runnables """
    # Component 1 : define the prompt template using LCEL
    prompt=ChatPromptTemplate.from_template("talk about the recent US vs Irasn war in one sentence : {text}") 

    model = ChatOpenAI(
        model="openai/gpt-5.6-luna-pro",
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=0.2,
    )

    parser = StrOutputParser()

    # Component 2 : compose with pipe operator
    chain = prompt | model | parser

    # Component 3 : inspect the schema of the chain
    print("Chain input schema:", chain.input_schema.model_json_schema())
    print("Chain output schema:", chain.output_schema.model_json_schema())

    return chain


def exercise_chain():
    """ Exercise: create a chain that:
        1. takes a product name and target audience as input
        2. generates a marketing slogan in one sentence
        3. returns the slogan as output

        test with product name = "Smartphone" and target audience = "teenagers"
    """
    # Component 1: prompt template with TWO named input variables
    prompt = ChatPromptTemplate.from_template(
        "You are a creative marketing assistant. "
        "Write a marketing slogan for the product '{product_name}' "
        "targeted at {target_audience}. Answer in one sentence only."
    )

    model = ChatOpenAI(
        model="openai/gpt-5.6-luna-pro",
        api_key=os.environ["KAYA_API_KEY"],
        base_url="https://kayaai.ir/api",
        temperature=0.2,
    )

    parser = StrOutputParser()

    # Component 2: compose
    chain = prompt | model | parser

    # Component 3: execute with structured input dict
    result = chain.invoke({
        "product_name": "portfolio advisor",
        "target_audience": "s and p 500 traders",
    })
    print(f"Slogan: {result}")

    return chain

if __name__ == "__main__":
    # demo_basic_chain()
    # demo_batch_execution()
    # demo_streaming()
    # demo_schema_inspection()    
    # demo_streaming()
    exercise_chain()
