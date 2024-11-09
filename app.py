# Limit the number of messages in chat history to the most recent 5 exchanges
MAX_HISTORY_LENGTH = 5

# Function to generate detailed response
def generate_detailed_response(user_input):
    # Append user input to chat history
    st.session_state.chat_history.append(f"User: {user_input}")
    
    # Limit chat history to the most recent 5 messages
    if len(st.session_state.chat_history) > MAX_HISTORY_LENGTH * 2:  # 2 for user + bot messages
        st.session_state.chat_history = st.session_state.chat_history[-MAX_HISTORY_LENGTH * 2:]

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
