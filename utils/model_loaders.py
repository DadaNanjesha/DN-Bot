# utils/model_loaders.py
import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_detector_model():
    """Load the roberta-base-openai-detector pipeline for AI text detection."""
    return pipeline("text-classification", model="roberta-base-openai-detector")

@st.cache_resource
def load_paraphrase_model():
    """Load the T5-based paraphrasing pipeline (e.g., google/flan-t5-base)."""
    return pipeline("text2text-generation", model="google/flan-t5-base")
