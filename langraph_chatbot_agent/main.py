import os
import threading
import requests
import streamlit as st
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import uvicorn

# Load API keys from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Model names
MODEL_NAMES = ["llama3-70b-8192", "mixtral-8x7b-32768"]

# Initialize Tavily Search tool
tool_tavity = TavilySearchResults(max_results=2)
tools = [tool_tavity]

# FastAPI app
app = FastAPI(title="LangGraph AI Agent")

# Define request schema
class RequestState(BaseModel):
    model_name: str
    system_prompt: str
    messages: list  # Corrected to accept a list of messages

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with LangGraph AI Agent.
    """
    if request.model_name not in MODEL_NAMES:
        return {"error": "Invalid model name. Please select a valid model."}

    # Initialize the LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name=request.model_name)

    # Create the agent
    agent = create_react_agent(llm, tools=tools, system_message=request.system_prompt)

    # Create state for processing
    state = {"messages": request.messages}

    # Process the response
    result = agent.invoke(state)

    return {"message": result}

# Function to run FastAPI server in the background
def run_api():
    uvicorn.run(app, host="127.0.0.1", port=8000)

# Start FastAPI server in a separate thread
threading.Thread(target=run_api, daemon=True).start()

# Streamlit UI
st.set_page_config(page_title="LangGraph AI Agent", layout="centered")
st.title("LangGraph ChatBot")
st.write("Interact with LangGraph using this interface.")

given_system_prompt = st.text_area("Define your AI Agent:", height=100, placeholder="Type your System Prompt Here....")
selected_model = st.selectbox("Select Model:", MODEL_NAMES)
user_input = st.text_area("Enter your Message:", height=100, placeholder="Type your Message Here...")

if st.button("Submit"):
    if user_input.strip():
        try:
            API_URL = "http://127.0.0.1:8000/chat"
            payload = {"messages": [user_input], "model_name": selected_model, "system_prompt": given_system_prompt}
            response = requests.post(API_URL, json=payload)

            if response.status_code == 200:
                response_data = response.json()
                st.subheader("Agent Response:")
                st.markdown(f"**Final Response:** {response_data.get('message', '')}")
            else:
                st.error(f"Request failed with status code {response.status_code}. Response: {response.text}")
        except Exception as e:
            st.error(f"An Error Occurred: {e}")
    else:
        st.warning("Please enter a message before clicking 'Submit'.")
