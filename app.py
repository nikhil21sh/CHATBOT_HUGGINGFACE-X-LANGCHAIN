
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFaceEndpoint
from dotenv import load_dotenv
import os


load_dotenv()
st.markdown("""
    <style>
        body {
            background-color: #0d0d0d;
            color: #ffffff;
        }
        .main {
            background-color: #0d0d0d;
        }
        .message {
            padding: 10px 20px;
            margin: 10px 0;
            border-radius: 20px;
            max-width: 75%;
            word-wrap: break-word; <!-- this is for wrapping long words in container -->
        }
        .user {
            background-color: #6a0dad;
            color: white;
            margin-left: auto;
        }
        .bot {
            background-color: #2a2a2a;
            color: #c2bfff;
            margin-right: auto;
        }
        .header {
            font-size: 28px;
            font-weight: bold;
            text-align: center;
            margin-top: 20px;
            color: #c2bfff;
        }
        input {
            background-color: #292929 !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='header'>🧠 Tech Chatbot - LangChain x HuggingFace</div>", unsafe_allow_html=True)

input_text = st.text_input("💬 Ask me anything technical")


prompt = ChatPromptTemplate.from_messages([
    ("system","You are a friendly technical guide who explains things with depth and uses analogies when needed. Break down each answer into simple steps and ask follow-up questions if the user seems unsure."
),
    ("user", "Question: {question}")
])
llm = HuggingFaceEndpoint(
    repo_id="HuggingFaceH4/zephyr-7b-beta",
    huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_TOKEN"),
    temperature=0.5, max_new_tokens=500
)
parser = StrOutputParser()
chain = prompt | llm | parser

if input_text:
    with st.container():
        st.markdown(f"<div class='message user'>🧑‍💻 {input_text}</div>", unsafe_allow_html=True)
        response = chain.invoke({"question": input_text})
        st.markdown(f"<div class='message bot'>🤖 {response}</div>", unsafe_allow_html=True)
