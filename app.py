import streamlit as st
import openai
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

import os
from dotenv import load_dotenv

load_dotenv()

langsmith_api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
if langsmith_api_key:
    os.environ["LANGSMITH_API_KEY"] = langsmith_api_key
    os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = "Q&A CHATBOT WITH OPENAI"
else:
    os.environ["LANGSMITH_TRACING"] = "false"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. please respond to the user query"),
        ("user","Question:{question}")
    ]

)

def generate_response(question,api_key,llm,temprature,max_tokens):
    if not api_key:
        raise ValueError("Please enter your OpenAI API key in the sidebar.")

    llm = ChatOpenAI(
        model=llm,
        api_key=api_key,
        temperature=temprature,
        max_tokens=max_tokens,
    )
    output_parser = StrOutputParser()
    chain = prompt|llm|output_parser
    answer = chain.invoke({'question':question})
    return answer

st.title("Enhanced Q&A chatbot with openai")

st.sidebar.title("settings")
api_key = st.sidebar.text_input("enter api key",type="password")

llm = st.sidebar.selectbox("select open ai model",["gpt-6.2-astra","gpt-6.2-sol","gpt-6.1-terra","gpt-6-luna"])

temprature = st.sidebar.slider("temprature",min_value=0.0,max_value=1.0,value = 0.7)
max_tokens = st.sidebar.slider("max token",min_value=50,max_value=300,value=150)

st.write("go ahead and ask any question")
user_input = st.text_input("you:")

if user_input:
    if not api_key:
        st.warning("Please enter your OpenAI API key in the sidebar.")
    else:
        try:
            response = generate_response(user_input,api_key,llm,temprature,max_tokens)
            st.write(response)
        except Exception as error:
            st.error(str(error))
else:
    st.write("please provide the query")