import streamlit as st
import requests
import time

# Page Configuration
st.set_page_config(
    page_title="TinyStories GPT",
    layout="wide"
)

# Custom CSS for a cleaner look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        border-radius: 5px;
        height: 3em;
        transition: all 0.3s ease;
    }
    .stTextArea textarea {
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("TinyStories GPT Story Generator")
st.caption("Generate structured creative narratives using a Transformer model trained on the TinyStories dataset.")

# API Health Check
def check_api_health():
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar Configuration
with st.sidebar:
    st.header("Generation Parameters")
    
    max_tokens = st.slider(
        "Token Limit",
        min_value=50,
        max_value=1000,
        value=600,
        step=50,
        help="The maximum length of the generated sequence."
    )
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.85,
        step=0.05,
        help="Higher values increase randomness; lower values focus on the most likely sequence."
    )
    
    top_k = st.slider(
        "Top-k Sampling",
        min_value=1,
        max_value=200,
        value=50,
        step=1,
        help="Limits the vocabulary choices to the top k most probable tokens."
    )
    
    st.divider()
    
    # System Status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("System Status: Online")
    else:
        st.error("System Status: Offline")
        st.info("Ensure the backend is running at localhost:8000")
        st.code("uvicorn api.main:app --reload", language="bash")

# Prompt Presets
st.subheader("Prompt Templates")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Narrative: The Rabbit", use_container_width=True):
        st.session_state.prompt = "Once upon a time, there was a cheerful little rabbit named Hopper who"

with col2:
    if st.button("Narrative: The Discovery", use_container_width=True):
        st.session_state.prompt = "One day, a curious girl found a magical"

with col3:
    if st.button("Narrative: Space Exploration", use_container_width=True):
        st.session_state.prompt = "In a distant galaxy, a brave astronaut"

# Input Area
prompt = st.text_area(
    "Input Prompt",
    value=st.session_state.get("prompt", "Once upon a time, there was a cheerful little rabbit named Hopper who"),
    height=150,
    placeholder="Enter the beginning of your story..."
)

# Execution
left_spacer, center_button, right_spacer = st.columns([1, 1, 1])
with center_button:
    generate_button = st.button("Generate Narrative", type="primary", use_container_width=True)

if generate_button:
    if not prompt.strip():
        st.warning("Input prompt is required for generation.")
    elif not api_healthy:
        st.error("Connection failed. Verify that the FastAPI backend is active.")
    else:
        st.divider()
        st.subheader("Output")
        
        placeholder = st.empty()
        generated = prompt
        
        try:
            with st.spinner("Processing..."):
                response = requests.post(
                    "http://localhost:8000/generate",
                    json={
                        "prompt": prompt,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "top_k": top_k,
                    },
                    stream=True,
                    timeout=60
                )
                
                if response.status_code == 200:
                    for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
                        if chunk:
                            generated += chunk
                            placeholder.markdown(generated + "█")
                            time.sleep(0.01)
                    
                    placeholder.markdown(generated)
                    
                    # Post-generation actions
                    st.download_button(
                        label="Download Text File",
                        data=generated,
                        file_name="story_output.txt",
                        mime="text/plain"
                    )
                else:
                    st.error(f"API Error: {response.status_code}")
                    
        except requests.exceptions.Timeout:
            st.error("The request timed out. Consider reducing the Token Limit.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #6c757d; font-size: 0.85em;'>
        Transformer Language Model Interface | Streamlit & FastAPI
    </div>
    """,
    unsafe_allow_html=True
)
