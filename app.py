import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the smaller model (distilgpt2) to save memory
try:
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    st.write("Model and Tokenizer loaded successfully.")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

# Initialize session state variables if not already present
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Limit the number of messages in the chat history to the most recent 5 exchanges (user + bot)
MAX_HISTORY_LENGTH = 5

# Function to generate a basic response
def generate_detailed_response(user_input):
    try:
        # Append user input to chat history
        st.session_state.chat_history.append(f"User: {user_input}")

        # Limit chat history to the most recent 5 messages (2 * MAX_HISTORY_LENGTH for user + bot messages)
        if len(st.session_state.chat_history) > MAX_HISTORY_LENGTH * 2:
            st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_LENGTH * 2:]

        # Join all previous chat history to form a context for the model
        bot_input = "\n".join(st.session_state.chat_history)

        # Encode the input
        bot_input_ids = tokenizer.encode(bot_input + tokenizer.eos_token, return_tensors='pt')

        # Ensure the tensor is of type long (this will avoid the FloatTensor error)
        bot_input_ids = bot_input_ids.to(torch.long)

        # Generate response (set a reasonable max length to prevent memory issues)
        output = model.generate(bot_input_ids, max_length=200, pad_token_id=tokenizer.eos_token_id)

        # Decode the response
        bot_response = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)

        # Append bot's response to chat history
        st.session_state.chat_history.append(f"Bot: {bot_response}")

        # Return the bot's response
        return bot_response
    except Exception as e:
        st.write(f"Error generating response: {e}")
        return "Sorry, there was an issue processing your request."

# Streamlit UI
st.title("Mental Health Companion Chatbot")
st.write("I'm here to listen and help. Feel free to share what's on your mind!")

# User input
user_input = st.text_input("What's on your mind?")

# Generate detailed response when the user submits input
if user_input:
    response = generate_detailed_response(user_input)
    st.write(response)

# Display the chat history (optional)
if st.button("Show Chat History"):
    st.write("\n".join(st.session_state.chat_history))

# Clear session state (reset the conversation)
if st.button("Reset Conversation"):
    st.session_state.chat_history = []
    st.write("Conversation reset. Start a new chat!")

# Option to provide tips, helplines, or breathing exercises
st.write("Here are some resources that may help:")
st.write("- Breathing exercises: Try taking slow, deep breaths to calm down.")
st.write("- Motivational affirmation: You are strong and capable of overcoming this!")
st.write("- Mental health helplines: If you need immediate assistance, please reach out to a mental health helpline.")
