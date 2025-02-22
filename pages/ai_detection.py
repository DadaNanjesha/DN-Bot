# pages/pdf_detection.py
import streamlit as st
import pandas as pd
import altair as alt
from utils.pdf_utils import extract_text_from_pdf, generate_annotated_pdf, word_count
from utils.ai_detection_utils import classify_text_hf  # Defined in utils/ai_detection_utils.py
from io import BytesIO

def show_pdf_detection_page():
    st.title("PDF Detection & Annotation")
    st.write("Upload a PDF document, classify each sentence, and download an annotated PDF with color-coded highlights.")
    
    # Initialize session state keys if not present
    if "classification_map" not in st.session_state:
        st.session_state["classification_map"] = None
    if "percentages" not in st.session_state:
        st.session_state["percentages"] = None
    if "annotated_pdf" not in st.session_state:
        st.session_state["annotated_pdf"] = None
    if "original_pdf_text" not in st.session_state:
        st.session_state["original_pdf_text"] = ""
    
    uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])
    if uploaded_pdf:
        pdf_bytes = uploaded_pdf.read()
        with st.spinner("Extracting text from PDF..."):
            extracted = extract_text_from_pdf(pdf_bytes)
            st.session_state["original_pdf_text"] = extracted

        if not st.session_state["original_pdf_text"].strip():
            st.error("No text could be extracted from this PDF.")
            return

        with st.spinner("Classifying text..."):
            c_map, pcts = classify_text_hf(st.session_state["original_pdf_text"])
            st.session_state["classification_map"] = c_map
            st.session_state["percentages"] = pcts

        with st.spinner("Annotating PDF..."):
            annotated = generate_annotated_pdf(pdf_bytes, st.session_state["classification_map"])
            st.session_state["annotated_pdf"] = annotated

        # Display classification breakdown
        if st.session_state["percentages"]:
            st.subheader("Classification Breakdown")
            df = pd.DataFrame({
                "Category": list(st.session_state["percentages"].keys()),
                "Percentage": list(st.session_state["percentages"].values())
            })
            color_scale = alt.Scale(
                domain=["AI-generated", "AI-generated & AI-refined", "Human-written", "Human-written & AI-refined"],
                range=["#ff6666", "#ff9900", "#66CC99", "#6699FF"]
            )
            chart = (
                alt.Chart(df)
                .mark_bar()
                .encode(
                    y=alt.Y("Category:N", sort="-x"),
                    x=alt.X("Percentage:Q", title="Percentage (%)", scale=alt.Scale(domain=[0, 100])),
                    color=alt.Color("Category:N", scale=color_scale),
                    tooltip=["Category:N", "Percentage:Q"]
                )
                .properties(height=200, width=600)
            )
            st.altair_chart(chart, use_container_width=True)
            st.table(df.set_index("Category"))
        
        if st.session_state["annotated_pdf"]:
            st.subheader("Download Annotated PDF")
            st.download_button(
                "Download Annotated PDF",
                data=st.session_state["annotated_pdf"],
                file_name="annotated_output.pdf",
                mime="application/pdf"
            )
        
        with st.expander("View Extracted Text"):
            st.text_area("Extracted PDF Text", st.session_state["original_pdf_text"], height=200)
    else:
        st.info("Please upload a PDF to start.")
