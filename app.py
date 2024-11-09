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
    # Generate response from the model
    response = model(user_input, max_length=50, num_return_sequences=1)
    if any(word in user_input.lower() for word in ["anxious", "stressed", "sad", "lonely", "help"]):
        response_text = "I'm here to help! Try this exercise: Breathe in slowly for 4 seconds, hold, then exhale slowly. Repeat as needed."
    else:
        response_text = response[0]['generated_text']
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
    if "help" in user_input.lower():
        st.write("Consider talking to a support group or helpline.")
