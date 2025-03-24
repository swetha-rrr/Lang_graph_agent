import streamlit as st
import requests
#stramlit configuration 
st.set_page_config(page_title="LangGraph AI Agent.UI", layout="centered")
#define the api end point
API_URL = "http://127.0.0.1:8000/chat"

MODEL_NAMES = [
    "llama3-70b-8192",  # Model 1: Llama 3 with specific configuration
    "mixtral-8x7b-32768"  # Model 2: Mixtral with specific configuration
]

#streamlit ui elements
st.title("LangGraph ChatBot")
st.write("Interact with LangGragh using this Interface.")

given_system_prompt=st.text_area("Define your AI Agent:", height=150, placeholder="Type your System Prompt Here....")

#dropdown for selecting the model
selected_model=st.selectbox("select model:",MODEL_NAMES)

#user text area
user_input=st.text_area("Enter your Message:", height=150, placeholder="Type your Message Here...")

#button to send query

if st.button("Submit"):
    if user_input.strip():
        try:
            payload = {"messages": user_input, "model_name": selected_model, "system_prompt": given_system_prompt}
            response = requests.post(API_URL, json=payload)

            # Debugging: Print the response
            st.write("Response Status:", response.status_code)
            st.write("Response Content:", response.text)

            if response.status_code == 200:
                response_data = response.json()
                st.subheader("Agent Response:")
                st.markdown(f"**Final Response:** {response_data.get('message', '')}")
            else:
                st.error(f"Request failed with status code {response.status_code}. Response: {response.text}")
        except Exception as e:
            st.error(f"An Error Occurred: {e}")
    else:
        st.warning("Please enter a message before clicking 'Send Query'.")

