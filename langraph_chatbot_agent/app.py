from fastapi import FastAPI # fastapi is used for creating web applicationthat is to get responses and send back
from pydantic import BaseModel
from typing import List
from langchain_community.tools.tavily_search import TavilySearchResults
import os # os module for environment variable handling The os module in Python provides functions to interact with the operating system. It helps in file handling, environment variables, process management, and system operations.
from langgraph.prebuilt import create_react_agent  # Function to create a ReAct agent
from langchain_groq import ChatGroq
from dotenv import load_dotenv



# Load environment variables from .env file
load_dotenv()
# Access the API keys
groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv('TAVILY_API_KEY')

# give model Names
MODEL_NAMES = [
    "llama3-70b-8192",  # Model 1: Llama 3 with specific configuration
    "mixtral-8x7b-32768"  # Model 2: Mixtral with specific configuration
]


#initialize tavily search so it can search the internet and get the response back
tool_tavity=TavilySearchResults(max_results=2)#fetch top 2 results from web

tools=[tool_tavity, ]

#fastapi Application with title
app=FastAPI(title='LangGraph AI Agent')
#defining the request schema for structed output usinng pydantic's base model
class RequestState(BaseModel):
    model_name: str#like gpt4 or gpt3
    system_prompt: str#you are a chatbot
    messages: list[str]# List of messages in the task

@app.post("/chat")
def chat_endpoint(request: RequestState):
    """
    API endpoint to interact with the chatbot using LangGraph and tools.
    Dynamically selects the model specified in the request.
    """
    if request.model_name not in MODEL_NAMES:
        # Return an error response if the model name is invalid
        return {"error": "Invalid model name. Please select a valid model."}
#Initialize the LLm with Selected model Name
    llm=ChatGroq(groq_api_key=groq_api_key, model_name=request.model_name)

#create an react Agent
    agent = create_react_agent(llm, tools=tools, state_modifier=request.system_prompt)


# Create the initial state for processing
    state = {"messages": request.messages}

# Process the state using the agent
    result = agent.invoke(state)  # Invoke the agent (can be async or sync based on implementation)

# Return the result as the response
    return result

# Run the application if executed as the main script
if __name__ == '__main__':
    import uvicorn  # Import Uvicorn server for running the FastAPI app
    uvicorn.run(app, host='127.0.0.1', port=8000)  # Start the app on localhost with port 8000

#When you run this script, it will check if the script is being executed directly.
#If yes, it will start a Uvicorn server on 127.0.0.1:8000, serving the FastAPI app.

