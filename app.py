import streamlit as st
import os
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

HF_TOKEN = os.getenv("HF_TOKEN")

llm = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct",
    huggingfacehub_api_token=HF_TOKEN,
    task="text-generation",
    temperature=0.7
)

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

st.set_page_config(page_title="Medical ChatBot - AI Health Assistant", page_icon="🩺", layout="centered")

st.title("🩺 Medical ChatBot - AI Health Assistant")
st.write("💬 Ask any health-related question, and Medical ChatBot will assist you!")

st.write("⚠️ **Disclaimer:** This tool provides general health information. It is not intended to diagnose, treat, or cure any medical conditions. Please consult a healthcare professional for serious health concerns.")

user_input = st.text_area("Enter your question:")

st.write("⚠️ For emergencies, please consult a healthcare professional.")

def filter_content(input_text):
    harmful_keywords = ["self-harm", "suicide", "kill", "die", "harm"]
    if any(keyword in input_text.lower() for keyword in harmful_keywords):
        return "This query seems to mention sensitive topics. If you are in immediate distress, please seek help from a professional."
    return input_text

if st.button("Get Advice"):
    if user_input.strip():
        filtered_input = filter_content(user_input)
        
        if filtered_input != user_input:
            st.warning(filtered_input)
        else:
            with st.spinner("Generating response..."):
                response = (prompt_medical | llm).invoke({"user_input": user_input})
            
            st.subheader("Medical ChatBot's Response:")
            st.write(response.replace("**", ""))
    else:
        st.warning("Please enter a question!")
