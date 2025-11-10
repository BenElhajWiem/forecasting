from __future__ import annotations
import streamlit as st

# Import modules
from utils.model_registry import Registry, LLMClientAdapter
from agents.orchestration_agent import orchestration_agent

# --- PAGE CONFIG ---
st.set_page_config(page_title="A Retrieval-Augmented Multi-Agent LLMs for Time Series",
                   page_icon="🌸",
                   layout="centered")
# --- HEADER ---
st.markdown(
    """
    <style>
    .title {font-size: 2.2rem; font-weight: 700; text-align: center; color: #e91e63;}
    .subtitle {font-size: 1rem; text-align: center; color: #888; margin-bottom: 2rem;}
    .stButton>button {border-radius: 10px; background-color: #e91e63; color: white; font-weight: 600;}
    .stTextArea textarea {border-radius: 10px;}
    </style>
    <div class="title">Forecasting with Language</div>
    <div class="subtitle">Choose a model, Enter a query, and Visualize the result.</div>
    """,
    unsafe_allow_html=True,
)

st.title("Forecasting with Language")
st.caption("Choose a model, enter a query, , see the result.")

# --- MODEL SELECTION ---
registry = Registry()
MODEL_NAME_TO_KEY = {
    "DeepSeek": "deepseek-chat",
    "Gemini": "gemini-flash-native",
    "Claude": "claude-api",
    "OpenAI": "openai-mini",
}
available_models = {
    nice: key for nice, key in MODEL_NAME_TO_KEY.items() if key in registry.presets
}
if not available_models:
    st.error("No matching presets found in your Registry. Check `registry.presets` keys.")
    st.stop()

model_name = st.selectbox(label="Model",
                          options= ["-- Select a model --"]+list(available_models.keys()),
                          index=0,
                          help="Select the model to use for forecasting.",
                          )

# --- QUERY INPUT ---
query = st.text_area(
    "Query",
    placeholder="Forecast Here.",
    height=100,
    help="Enter your forecasting query here.",
)

# --- Run ---
if st.button("🚀 Run"):
    if model_name == "-- Select a model --":
        st.warning("Please select a model first.")
        st.stop()

    if not query.strip():
        st.warning("Please enter a query.")
        st.stop()

    preset_key = available_models[model_name] 
    try:
        spec = registry.presets[preset_key]
        adapter = LLMClientAdapter(spec)
    except Exception as e:
        st.error(f"Failed to initialize adapter: {e}")
        st.stop()

    with st.spinner("Running your agent..."):
        try:
            result = orchestration_agent(user_query=query, adapter=adapter)
            st.success("Done ✅")
            st.write(result) 
        except Exception as e:
            st.error("Error during execution.")
            st.exception(e)