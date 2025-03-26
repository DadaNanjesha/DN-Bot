import random
import re
import ssl
import warnings
import nltk
import spacy
import streamlit as st
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize, word_tokenize

warnings.filterwarnings("ignore", category=FutureWarning)

########################################
# Download needed NLTK resources
########################################
def download_nltk_resources():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    resources = ["punkt", "averaged_perceptron_tagger", "wordnet"]
    for r in resources:
        nltk.download(r, quiet=True)

download_nltk_resources()

########################################
# Prepare spaCy pipeline
########################################
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.warning("spaCy en_core_web_sm model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

########################################
# Citation Regex
########################################
CITATION_REGEX = re.compile(
    r"\(\s*[A-Za-z&\-,\.\s]+(?:et al\.\s*)?,\s*\d{4}(?:,\s*(?:pp?\.\s*\d+(?:-\d+)?))?\s*\)"
)

########################################
# Helper: Word & Sentence Counts
########################################
def count_words(text):
    return len(word_tokenize(text))

def count_sentences(text):
    return len(sent_tokenize(text))

########################################
# Step 1: Extract & Restore Citations
########################################
def extract_citations(text):
    refs = CITATION_REGEX.findall(text)
    placeholder_map = {}
    replaced_text = text
    for i, r in enumerate(refs, start=1):
        placeholder = f"[[REF_{i}]]"
        placeholder_map[placeholder] = r
        replaced_text = replaced_text.replace(r, placeholder, 1)
    return replaced_text, placeholder_map

PLACEHOLDER_REGEX = re.compile(r"\[\s*\[\s*REF_(\d+)\s*\]\s*\]")


def restore_citations(text, placeholder_map):

    def replace_placeholder(match):
        placeholder = match.group(0)
        return placeholder_map.get(placeholder, placeholder)

    restored = PLACEHOLDER_REGEX.sub(replace_placeholder, text)
    return restored


########################################
# Step 2: Expansions, Synonyms, & Transitions
########################################
contraction_map = {
    "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
    "'ve": " have", "'d": " would", "'m": " am"
}

ACADEMIC_TRANSITIONS = [
    "Moreover,",
    "Additionally,",
    "Furthermore,",
    "Hence,",
    "Therefore,",
    "Consequently,",
    "Nonetheless,",
    "Nevertheless,",
    "In contrast,",
    "On the other hand,",
    "In addition,",
    "As a result,",
]

def expand_contractions(sentence):
    tokens = word_tokenize(sentence)
    expanded = []
    for t in tokens:
        replaced = False
        lower_t = t.lower()
        for contr, expansion in contraction_map.items():
            if contr in lower_t and lower_t.endswith(contr):
                new_t = lower_t.replace(contr, expansion)
                if t[0].isupper():
                    new_t = new_t.capitalize()
                expanded.append(new_t)
                replaced = True
                break
        if not replaced:
            expanded.append(t)
    return " ".join(expanded)

def replace_synonyms(sentence, p_syn=0.2):
    if not nlp:
        return sentence

    doc = nlp(sentence)
    new_tokens = []
    for token in doc:
        if "[[REF_" in token.text:
            new_tokens.append(token.text)
            continue
        if token.pos_ in ["ADJ", "NOUN", "VERB", "ADV"] and wordnet.synsets(token.text):
            if random.random() < p_syn:
                synonyms = get_synonyms(token.text, token.pos_)
                if synonyms:
                    new_tokens.append(random.choice(synonyms))
                else:
                    new_tokens.append(token.text)
            else:
                new_tokens.append(token.text)
        else:
            new_tokens.append(token.text)
    return " ".join(new_tokens)


def add_academic_transition(sentence, p_transition=0.2):
    if random.random() < p_transition:
        transition = random.choice(ACADEMIC_TRANSITIONS)
        return f"{transition} {sentence}"
    return sentence


def get_synonyms(word, pos):
    wn_pos = None
    if pos.startswith("ADJ"):
        wn_pos = wordnet.ADJ
    elif pos.startswith("NOUN"):
        wn_pos = wordnet.NOUN
    elif pos.startswith("ADV"):
        wn_pos = wordnet.ADV
    elif pos.startswith("VERB"):
        wn_pos = wordnet.VERB

    synonyms = set()
    if wn_pos:
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace("_", " ")
                if lemma_name.lower() != word.lower():
                    synonyms.add(lemma_name)
    return list(synonyms)


########################################
# Step 3: Minimal "Humanize" line-by-line
########################################
def minimal_humanize_line(line, p_syn=0.2, p_trans=0.2):
    line = expand_contractions(line)
    line = replace_synonyms(line, p_syn=p_syn)
    line = add_academic_transition(line, p_transition=p_trans)
    return line


def minimal_rewriting(text, p_syn=0.2, p_trans=0.2):
    lines = sent_tokenize(text)
    out_lines = [
        minimal_humanize_line(ln, p_syn=p_syn, p_trans=p_trans) for ln in lines
    ]
    return " ".join(out_lines)


########################################
# Final: Show Humanize Page
########################################
def show_humanize_page():
    st.title("Humanize AI Text (No T5, faster, keeps references, minimal changes)")
    st.write(
        "This approach only expands contractions, optionally replaces synonyms, and inserts academic transitions. "
        "It preserves your APA citations exactly, line by line, without T5 rewriting (which can lose data)."
    )

    input_text = st.text_area("Enter text to humanize", height=200)
    p_syn = st.slider("Synonym Replacement Probability", 0.0, 1.0, 0.2, 0.05)
    p_trans = st.slider("Academic Transition Probability", 0.0, 1.0, 0.2, 0.05)

    if st.button("Humanize"):
        if not input_text.strip():
            st.warning("Please enter some text first.")
            return

        orig_wc = count_words(input_text)
        orig_sc = count_sentences(input_text)

        with st.spinner("Rewriting text..."):
            no_refs_text, placeholders = extract_citations(input_text)
            partially_rewritten = minimal_rewriting(
                no_refs_text, p_syn=p_syn, p_trans=p_trans
            )
            final_text = restore_citations(partially_rewritten, placeholders)

        # Normalize spaces around punctuation
        final_text = re.sub(
            r"\s+([.,;:!?])", r"\1", final_text
        )  # Remove spaces before punctuation
        final_text = re.sub(
            r"(\()\s+", r"\1", final_text
        )  # Remove spaces after opening parenthesis
        final_text = re.sub(
            r"\s+(\))", r")", final_text
        )  # Remove spaces before closing parenthesis

        new_wc = count_words(final_text)
        new_sc = count_sentences(final_text)

        st.subheader("Humanized Output")
        st.text_area("Result", final_text, height=200)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"**Original Word Count:** {orig_wc}")
            st.markdown(f"**Original Sentence Count:** {orig_sc}")
        with col2:
            st.markdown(f"**Rewritten Word Count:** {new_wc}")
            st.markdown(f"**Rewritten Sentence Count:** {new_sc}")
    else:
        st.info("Adjust probabilities above and click the button to rewrite.")

# Run the app
if __name__ == "__main__":
    show_humanize_page()
