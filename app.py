import streamlit as st
import os
import re
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

HF_TOKEN = os.getenv("HF_TOKEN")

# Initialize LLM (Language Model) from HuggingFace
llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    temperature=0.7
)

# Prompt template for generating responses
prompt_medical = PromptTemplate.from_template(
    """
    ### USER INPUT:
    {user_input}
    
    ### INSTRUCTION:
    You are MedicalChatBot, an AI-powered medical assistant providing general health advice. 
    Respond with helpful and informative guidance based on general knowledge. 
    If the question is serious, suggest consulting a medical professional. 
    Keep responses professional, clear, and supportive. Avoid diagnoses or prescriptions. 
    Do not include any country-specific information or emergency numbers in your responses.
    
    ### RESPONSE:
    """
)

# Streamlit page config
st.set_page_config(page_title="Medical ChatBot - AI Health Assistant", page_icon="🩺", layout="centered")

# Streamlit UI elements
st.title("🩺 Medical ChatBot - AI Health Assistant")
st.write("💬 Ask any health-related question, and Medical ChatBot will assist you!")

# Disclaimer: Remind users it's not a replacement for medical advice
st.write("⚠️ **Disclaimer:** This tool provides general health information. It is not intended to diagnose, treat, or cure any medical conditions. Please consult a healthcare professional for serious health concerns.")

# User input area
user_input = st.text_area("Enter your question:")

# Button to submit the query
st.write("⚠️ For emergencies, please consult a healthcare professional.")

# Content moderation function to filter harmful or inappropriate queries
def filter_content(input_text):
    # Example: simple filter to avoid self-harm or inappropriate content
    harmful_keywords = ["self-harm", "suicide", "kill", "die", "harm"]
    if any(keyword in input_text.lower() for keyword in harmful_keywords):
        return "This query seems to mention sensitive topics. If you are in immediate distress, please seek help from a professional."
    return input_text

# Button to get the chatbot response
if st.button("Get Advice"):
    if user_input.strip():
        # Apply content moderation filter
        filtered_input = filter_content(user_input)
        
        # If input is flagged by the filter, display a warning
        if filtered_input != user_input:
            st.warning(filtered_input)
        else:
            with st.spinner("Generating response..."):
                # Invoke the model with the cleaned user input
                response = (prompt_medical | llm).invoke({"user_input": user_input})
            
            # Display the response from the chatbot
            st.subheader("Medical ChatBot's Response:")
            st.write(response.replace("**", ""))
    else:
        st.warning("Please enter a question!")
