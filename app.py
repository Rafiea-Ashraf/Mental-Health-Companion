import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datetime import datetime

# Load the DialoGPT model and tokenizer
@st.cache_resource
def load_model():
    model_name = "microsoft/DialoGPT-medium"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

tokenizer, model = load_model()

# Initialize conversation history for dynamic responses
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Generate detailed responses based on user input and conversational context
def generate_detailed_response(user_input):
    # Append user input to the conversation history
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')
    
    # Track conversation length and limit tokens
    bot_input_ids = torch.cat([torch.tensor(st.session_state.chat_history)] + [new_user_input_ids], dim=-1)
    chat_length = len(bot_input_ids[0])
    
    # Generate response
    output = model.generate(bot_input_ids, max_length=min(chat_length + 100, 1024), pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(output[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
    
    # Customize response with empathetic and actionable advice
    detailed_response = (
        f"{response_text}\n\n"
        "**I can sense you're feeling some intense emotions.** It’s completely okay to have days like this. "
        "Here are a few steps to help you manage:\n\n"
        
        "**1. Try a breathing exercise**:\nTake a deep breath, hold for a few seconds, and release slowly. "
        "Repeat this to help reduce tension.\n\n"
        
        "**2. Reflect on any potential triggers**:\nThink back on any recent changes or challenges. "
        "Understanding what might be causing your feelings can be a big step toward managing them.\n\n"
        
        "**3. Consider talking to someone you trust or a professional**:\nSometimes, simply sharing your thoughts "
        "can lighten the load. There are also support groups and mental health professionals who can help guide you."
        
        "\n\n**If you'd like more support resources, let me know, and I’ll provide links to helplines or articles that might help.**"
    )
    
    # Update chat history
    st.session_state.chat_history.append(new_user_input_ids)
    st.session_state.chat_history.append(output)
    
    return detailed_response

# Streamlit App Interface
st.title("Mental Health Companion for Teens")
st.markdown("### Your safe space for mental wellness support 🌈")

# Get user input and display response
user_input = st.text_input("What's on your mind?", placeholder="Share how you're feeling...")
if user_input:
    response = generate_detailed_response(user_input)
    st.write("Chatbot:", response)

    # Display options for more resources if distress is detected
    if any(keyword in user_input.lower() for keyword in ["help", "support", "alone", "distress", "overwhelmed", "anxious"]):
        st.write(
            "It sounds like things might be tough. Would you like resources for mental health support? "
            "There are many who want to help, and I can provide links to trusted helplines and support networks."
        )

# Optional: Displaying chat history (for real-time feel)
st.markdown("### Conversation History")
for i in range(0, len(st.session_state.chat_history), 2):
    user_text = tokenizer.decode(st.session_state.chat_history[i], skip_special_tokens=True)
    bot_text = tokenizer.decode(st.session_state.chat_history[i+1], skip_special_tokens=True)
    st.write(f"**You:** {user_text}")
    st.write(f"**Chatbot:** {bot_text}")
