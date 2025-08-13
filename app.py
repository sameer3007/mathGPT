import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="Math Solver & Search Assistant", page_icon="🧮")
st.title("🧮 Text to Math Problem Solver using Gemma 2")

# Initialize the language model
llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

# === Tool 1: Wikipedia Search ===
wikipedia_tool = Tool(
    name="Wikipedia",
    func=WikipediaAPIWrapper().run,
    description="Use to find general topic information from Wikipedia"
)

# === Tool 2: Math Calculator ===
math_chain = LLMMathChain.from_llm(llm=llm)
calculator_tool = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Use for math expressions and calculations"
)

# === Tool 3: Logical Reasoning ===
prompt = """
You are an agent solving math questions. Think step-by-step and explain clearly:
Question: {question}
Answer:
"""
reasoning_tool = Tool(
    name="Reasoning Tool",
    func=LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["question"], template=prompt)).run,
    description="Use for step-by-step math and logic-based explanations"
)

# === Combine tools into an agent ===
agent = initialize_agent(
    tools=[wikipedia_tool, calculator_tool, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose=False
)

# === Session state to store messages ===
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I'm a Math assistant. Ask me any math or logic question!"}
    ]

# Show past conversation
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# === Input box for user question ===
question = st.text_area("Enter your math question:")

if st.button("Get Answer"):
    if question:
        # Add user's message to session
        st.session_state.messages.append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        # Process response
        with st.spinner("Thinking..."):
            callback_handler = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
            response = agent.run(st.session_state.messages, callbacks=[callback_handler])

        # Add assistant's message to session and display
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)
    else:
        st.warning("Please enter a question.")











