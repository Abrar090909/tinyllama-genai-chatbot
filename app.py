import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

@st.cache_resource
def load_model():
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return tokenizer, model

# Load model
tokenizer, model = load_model()

# Streamlit UI config
st.set_page_config(page_title="TinyLLaMA Chatbot", layout="centered")
st.title("🦙 TinyLLaMA Chatbot")
st.caption("Chat with TinyLLaMA running locally on your machine.")

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Form to take user input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask something...", key="user_input_box")
    submitted = st.form_submit_button("🚀 Enter")

# Display chat history
for msg in st.session_state.chat_history:
    role = "🧑‍💻 You" if msg["role"] == "user" else "🦙 TinyLLaMA"
    st.markdown(f"**{role}:** {msg['content']}")

# Process the input
if submitted and user_input.strip():
    # Add user input to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # Format using chat template
    prompt = tokenizer.apply_chat_template(
        st.session_state.chat_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with st.spinner("TinyLLaMA is thinking..."):
        output = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract only the assistant's last message
    if "<|assistant|>" in decoded:
        reply = decoded.split("<|assistant|>")[-1].strip()
    else:
        reply = decoded.strip()

    # Save response
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# Clear chat button
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
