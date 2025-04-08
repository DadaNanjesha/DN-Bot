import streamlit as st
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import pipeline


# Make sure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# CITATION_REGEX: attempts to match something like (Smith et al., 2023, pp. 10-12)
CITATION_REGEX = re.compile(
    r"\(\s*[A-Za-z&\-,\.\s]+(?:et al\.\s*)?,\s*\d{4}(?:,\s*(?:pp?\.\s*\d+(?:-\d+)?))?\s*\)"
)

@st.cache_resource
def load_t5_model():
    """
    Load T5-based text2text-generation model (e.g. google/flan-t5-base) once, for speed.
    """
    return pipeline("text2text-generation", model="google/flan-t5-base")

def extract_citations(text):
    """
    Replace APA-like references with placeholders [[REF_1]], [[REF_2]], etc.
    Returns replaced_text and placeholder_map.
    """
    refs = CITATION_REGEX.findall(text)
    placeholder_map = {}
    replaced_text = text
    for i, r in enumerate(refs, start=1):
        placeholder = f"[[REF_{i}]]"
        placeholder_map[placeholder] = r
        replaced_text = replaced_text.replace(r, placeholder, 1)
    return replaced_text, placeholder_map

def restore_citations(text, placeholder_map):
    """
    Put original references back into the text.
    """
    restored = text
    for placeholder, ref_text in placeholder_map.items():
        restored = restored.replace(placeholder, ref_text)
    return restored

def sentence_level_rewrite(text, t5_pipeline, min_len=0, max_len=512):
    """
    Splits text by sentences, rewrites each with T5, then rejoins.
    """
    sentences = sent_tokenize(text)
    out_sents = []
    for sent in sentences:
        if not sent.strip():
            continue
        prompt = (
            "Rewrite this sentence to sound more natural and human while preserving details.\n\n"
            f"Original: {sent}"
        )
        res = t5_pipeline(
            prompt,
            do_sample=False,       # beam search, deterministic
            num_beams=4,
            min_length=max(min_len, len(word_tokenize(sent))),
            max_length=max_len,
            max_new_tokens=max_len
        )
        new_sent = res[0]["generated_text"].strip()
        out_sents.append(new_sent)
    return " ".join(out_sents)

def minimal_humanize_text(text):
    """
    Minimal rewriting approach:
      1) Replace references with placeholders
      2) Rewrite each sentence with T5
      3) Restore references
    """
    # 1) placeholders
    replaced, placeholder_map = extract_citations(text)
    
    # 2) rewrite each sentence
    t5 = load_t5_model()
    rewritten = sentence_level_rewrite(replaced, t5)
    
    # 3) restore references
    final = restore_citations(rewritten, placeholder_map)
    return final

def count_words(text):
    return len(word_tokenize(text))

def count_sentences(text):
    return len(sent_tokenize(text))

###############################################
# Streamlit App
###############################################
def main():
    st.title("Minimal, Fast T5 Humanizer")
    st.write("A simpler approach: references → placeholders → T5 rewriting sentence by sentence → restore references.")

    input_text = st.text_area("Enter text", height=200)
    if st.button("Rewrite"):
        if not input_text.strip():
            st.warning("Please enter some text.")
            return
        
        original_wordcount = count_words(input_text)
        original_sentcount = count_sentences(input_text)
        
        with st.spinner("Rewriting text..."):
            out_text = minimal_humanize_text(input_text)

        new_wordcount = count_words(out_text)
        new_sentcount = count_sentences(out_text)

        st.subheader("Rewritten Output")
        st.text_area("Humanized Text", out_text, height=200)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Original Word Count:** {original_wordcount}")
            st.markdown(f"**Original Sentence Count:** {original_sentcount}")
        with col2:
            st.markdown(f"**Rewritten Word Count:** {new_wordcount}")
            st.markdown(f"**Rewritten Sentence Count:** {new_sentcount}")

if __name__ == '__main__':
    main()
