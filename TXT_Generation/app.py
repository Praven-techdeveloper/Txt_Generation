import streamlit as st
from transformers import pipeline, set_seed
import torch
import time
import gc

st.set_page_config(
    page_title="Creative Text Generator",
    page_icon="🧠",
    layout="wide"
)
@st.cache_resource(show_spinner=False)
def load_model(model_choice):
    """Load the selected model with appropriate settings"""
    if model_choice == "GPT-J-6B (High Quality)":
        try:
            generator = pipeline(
                "text-generation",
                model="EleutherAI/gpt-j-6B",
                device_map="auto",
                load_in_8bit=True,
                torch_dtype=torch.float16
            )
            st.success("Loaded GPT-J-6B (8-bit quantized)")
            return generator
        except Exception as e:
            st.error(f"Failed to load GPT-J: {str(e)}")
            st.info("Falling back to GPT-2 Medium")
            return pipeline("text-generation", model="gpt2-medium")
    else:
        return pipeline("text-generation", model="gpt2-medium")

with st.sidebar:
    st.header("⚙️ Model Settings")
    
    model_choice = st.selectbox(
        "Select Model",
        ["GPT-2 (Fast)", "GPT-J-6B (High Quality)"],
        index=0
    )
    
    max_length = st.slider("Max Length", 50, 500, 150)
    temperature = st.slider("Temperature (Creativity)", 0.1, 1.0, 0.8)
    top_k = st.slider("Top-K (Diversity)", 1, 100, 50)
    seed = st.number_input("Random Seed", value=42)
    
    st.divider()
    st.caption("ℹ️ GPT-J requires >8GB GPU VRAM")
    st.caption("ℹ️ GPT-2 works on CPU instantly")
    
    if st.button("Clear Memory Cache"):
        gc.collect()
        torch.cuda.empty_cache()
        st.success("Memory cleared!")
st.title("🧠 Creative Text Generator")
st.subheader("Generate poetry, code, scripts & more with AI")
col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.text_area(
        "Enter your prompt:",
        height=150,
        value="AI:",
        placeholder="Python function to calculate Fibonacci...\nMovie script about time travel..."
    )
with col2:
    st.write("Try presets:")
    if st.button("Poetry", use_container_width=True):
        prompt = "Write a Shakespearean sonnet about space exploration:"
    if st.button("Code", use_container_width=True):
        prompt = "Python function to reverse a string without using built-in functions:"
    if st.button("Script", use_container_width=True):
        prompt = "INT. SPACESHIP COCKPIT - DAY\nCaptain stares at an alien planet on the viewscreen:"
    if st.button("Song", use_container_width=True):
        prompt = "[G]Rock ballad about lost love\n[D]Verse 1:"
if st.button("✨ Generate Text", type="primary", use_container_width=True):
    if not prompt.strip():
        st.warning("Please enter a prompt")
        st.stop()
    with st.spinner(f"Loading model..."):
        generator = load_model(model_choice)
    with st.spinner("Generating content..."):
        set_seed(seed)
        start_time = time.time()
        output = generator(
            prompt,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            num_return_sequences=1
        )[0]['generated_text']
        gen_time = time.time() - start_time
    st.divider()
    st.subheader("Generated Output")
    if "python" in prompt.lower() or "function" in prompt.lower():
        st.code(output, language="python")
    elif "script" in prompt.lower() or "INT." in prompt:
        st.code(output, language="plaintext")
    elif "haiku" in prompt.lower() or "sonnet" in prompt.lower():
        st.write(output)
    elif "song" in prompt.lower() or "[" in prompt:
        st.text(output)
    else:
        st.text_area("Output", output, height=300)
    
    st.caption(f"⏱️ Generated in {gen_time:.2f} seconds | Model: {model_choice}")
    st.caption(f"🔢 Parameters: Temp={temperature}, Top-K={top_k}, Seed={seed}")

# Instructions
st.divider()
st.markdown("### 📚 How to Use")
st.markdown("""
1. **Enter a creative prompt** (e.g., "Write a poem about robots", "JavaScript function to validate email")
2. **Adjust parameters** in sidebar:
   - **Model**: GPT-2 (faster) or GPT-J (higher quality)
   - **Max Length**: Output length (150-500 tokens)
   - **Temperature**: Creativity (0.1 = strict, 1.0 = wild)
   - **Top-K**: Diversity of word choices
3. Click **Generate Text** button
""")
st.markdown("### 💻 System Requirements")
st.markdown("""
| Model       | RAM  | GPU VRAM | Recommendation |
|-------------|------|----------|----------------|
| GPT-2       | 2GB+ | Not required | Any computer |
| GPT-J-6B    | 12GB+ | 8GB+ | NVIDIA RTX 3080+ |
""")