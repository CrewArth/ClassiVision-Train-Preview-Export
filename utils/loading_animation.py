import streamlit as st
import time

def show_loading_animation(total_epochs):
    """Shows a loading animation synchronized with model training."""
    progress_bar = st.progress(0)  # Initialize progress bar
    for i in range(total_epochs):
        time.sleep(0.1)  # Simulate work; replace with real-time update
        progress_bar.progress((i + 1) / total_epochs)
    progress_bar.empty()  # Remove progress bar after completion
