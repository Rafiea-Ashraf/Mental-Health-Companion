import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load pre-trained model and tokenizer (DialoGPT-small for better performance)
model_name = "microsoft/DialoGPT-small"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Make sure the model runs on CPU
model.to("cpu")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Function to generate detailed response
def generate_detailed_response(user_input):
    # Append user input to chat history
    st.session_state.chat_history.append(f"User: {user_input}")
    
    # Join all previous chat history to form a context for the model
    bot_input = "\n".join(st.session_state.chat_history)

    # Encode the input
    bot_input_ids = tokenizer.encode(bot_input + tokenizer.eos_token, return_tensors='pt')

    # Generate response
    output = model.generate(bot_input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)

    # Decode the response
    bot_response = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

    # Append bot's response to chat history
    st.session_state.chat_history.append(f"Bot: {bot_response}")

    # Return the bot's response
    return bot_response

# Streamlit Interface
st.title("Mental Health Companion for Teens")

# Prompt for user mood and input
mood = st.selectbox("Select your mood", ["Sad", "Stressed", "Anxious", "Lonely", "Angry", "Other"])

user_input = st.text_area("What's on your mind?", "")

# Button to submit user input
if st.button("Submit"):
    if user_input:
        response = generate_detailed_response(user_input)
        st.write("Bot:", response)
    else:
        st.write("Please share what's on your mind.")

# Display the chat history (for context in the chat)
if st.session_state.chat_history:
    for message in st.session_state.chat_history:
        st.write(message)
