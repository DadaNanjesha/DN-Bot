# main.py
import streamlit as st
from pages.ai_detection import show_pdf_detection_page
from pages.humanize_text import show_humanize_page

def main():
    st.set_page_config(page_title="Multi-Page App: PDF & Text Humanizer", layout="wide")

    # Initialize the current page in session_state if not present
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "PDF Detection & Annotation"

    st.title("Page Selection via Buttons")

    # Two buttons to select the page
    col1, col2 = st.columns(2)
    with col1:
        if st.button("PDF Detection & Annotation"):
            st.session_state["current_page"] = "PDF Detection & Annotation"
    with col2:
        if st.button("Humanize AI Text"):
            st.session_state["current_page"] = "Humanize AI Text"

    # Display the chosen page
    if st.session_state["current_page"] == "PDF Detection & Annotation":
        show_pdf_detection_page()
    else:
        show_humanize_page()

if __name__ == "__main__":
    main()
