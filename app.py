import streamlit as st
from transformers import pipeline
from datetime import datetime

# Load the text generation model
@st.cache_resource
def load_model():
    return pipeline('text-generation', model="distilgpt2")

model = load_model()

# Initialize mood tracker dictionary
mood_tracker = {}

def generate_response(user_input):
    # Analyzing input to detect distress keywords and give appropriate responses
    distress_keywords = ["anxious", "stressed", "sad", "lonely", "help", "angry", "upset", "annoyed", "don't feel like talking"]
    
    if any(word in user_input.lower() for word in distress_keywords):
        response_text = (
            "I'm here to listen. It sounds like you're going through a tough time. "
            "Try taking a few slow, deep breaths: breathe in for 4 seconds, hold for 4 seconds, and breathe out slowly. "
            "This can help you feel a little calmer. \n\n"
            "Remember, it's okay to feel this way, and I'm here to support you. If it helps, you might want to try writing down "
            "what’s bothering you, or simply take a break from things for a bit. \n\n"
            "Would you like information about a support group or a mental health helpline? Just let me know."
        )
    else:
        # Generate response from the model for general or positive inputs
        response = model(user_input, max_length=50, num_return_sequences=1)
        response_text = response[0]['generated_text']
        # Append a positive affirmation
        response_text += (
            "\n\nRemember, you're doing great, and even small steps can make a big difference. "
            "If there's anything else on your mind, feel free to share it with me."
        )
    return response_text

# Streamlit App Interface
st.title("Mental Health Companion for Teens")
st.markdown("### Your safe space for mental wellness support 🌈")

# Mood Selection
st.subheader("How are you feeling right now?")
mood = st.selectbox("Select your mood", ["Happy", "Calm", "Anxious", "Stressed", "Sad", "Lonely"])

# Save mood with timestamp
if mood:
    mood_tracker[datetime.now().strftime("%Y-%m-%d %H:%M:%S")] = mood
    st.write("Thank you for sharing. We're here to support you.")

# Display mood trend over time
if st.button("Show Mood Trend"):
    st.write("Mood Tracker:")
    st.write(mood_tracker)

# Chatbot Interaction
user_input = st.text_input("What's on your mind?")
if user_input:
    response = generate_response(user_input)
    st.write("Chatbot:", response)

    # Suggest resources if distress is detected
    if any(word in user_input.lower() for word in ["help", "support", "alone", "distress", "need someone"]):
        st.write("Consider talking to a support group or helpline if you feel comfortable. Reaching out can make a big difference.")
