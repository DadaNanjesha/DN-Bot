# pages/humanize_text.py
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

    resources = [
        'punkt',
        'averaged_perceptron_tagger',
        'wordnet'
    ]
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
        # print(placeholder_map[placeholder], replaced_text)
    return replaced_text, placeholder_map
PLACEHOLDER_REGEX = re.compile(r"\[\s*\[\s*REF_(\d+)\s*\]\s*\]")
def restore_citations(text, placeholder_map):
    restored = text.replace("[ [", "[[").replace("] ]", "]]")
    pattern = re.compile(r"\[\[\s*(REF_\d+)\s*\]\]")
    restored = pattern.sub(lambda m: f"[[{m.group(1)}]]", restored)
    
    print("restored",restored)
    for placeholder, ref_text in placeholder_map.items():
        print("placeholder, ref_text" , placeholder, ref_text) 
        restored = restored.replace(placeholder,ref_text)
        
    return restored

########################################
# Step 2: Expansions, Synonyms, & Transitions
########################################
contraction_map = {
    "n't": " not", "'re": " are", "'s": " is", "'ll": " will",
    "'ve": " have", "'d": " would", "'m": " am"
}

ACADEMIC_TRANSITIONS = [
    "Moreover,", "Additionally,", "Furthermore,", "Hence,",
    "Therefore,", "Consequently,", "Nonetheless,", "Nevertheless,"
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
    """
    Replaces some words with synonyms outside of placeholders,
    guided by a random chance p_syn for each eligible word.
    """
    tokens = word_tokenize(sentence)
    pos_tags = nltk.pos_tag(tokens)
    new_tokens = []
    for (word, pos) in pos_tags:
        if "[[REF_" in word:
            # This is a placeholder; skip synonyms
            new_tokens.append(word)
            continue
        if pos.startswith(('J','N','V','R')) and wordnet.synsets(word):
            # chance of replacement
            if random.random() < p_syn:
                synonyms = get_synonyms(word, pos)
                if synonyms:
                    new_tokens.append(random.choice(synonyms))
                else:
                    new_tokens.append(word)
            else:
                new_tokens.append(word)
        else:
            new_tokens.append(word)
    return " ".join(new_tokens)

def add_academic_transition(sentence, p_transition=0.2):
    """
    Sometimes prepend a transition, e.g. "Moreover," or "Hence,"
    """
    if random.random() < p_transition:
        transition = random.choice(ACADEMIC_TRANSITIONS)
        return f"{transition} {sentence}"
    return sentence

def get_synonyms(word, pos):
    wn_pos = None
    if pos.startswith('J'):
        wn_pos = wordnet.ADJ
    elif pos.startswith('N'):
        wn_pos = wordnet.NOUN
    elif pos.startswith('R'):
        wn_pos = wordnet.ADV
    elif pos.startswith('V'):
        wn_pos = wordnet.VERB

    synonyms = set()
    if wn_pos:
        for syn in wordnet.synsets(word, pos=wn_pos):
            for lemma in syn.lemmas():
                lemma_name = lemma.name().replace('_',' ')
                if lemma_name.lower() != word.lower():
                    synonyms.add(lemma_name)
    return list(synonyms)

########################################
# Step 3: Minimal "Humanize" line-by-line
########################################
def minimal_humanize_line(line, p_syn=0.2, p_trans=0.2):
    """
    For a single line of text (placeholders included):
      - expand contractions
      - optionally replace synonyms
      - optionally add transitions
    """
    # Expand contractions
    line = expand_contractions(line)
    # Replace synonyms
    line = replace_synonyms(line, p_syn=p_syn)
    # Possibly add transitions
    line = add_academic_transition(line, p_transition=p_trans)
    return line

def minimal_rewriting(text, p_syn=0.2, p_trans=0.2):
    """
    Splits text by lines or sentences, transforms each minimally,
    and reassembles. 
    """
    lines = sent_tokenize(text)
    out_lines = []
    for ln in lines:
        out_lines.append(minimal_humanize_line(ln, p_syn=p_syn, p_trans=p_trans))
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

        # Original counts
        orig_wc = count_words(input_text)
        orig_sc = count_sentences(input_text)

        with st.spinner("Rewriting text..."):
            # 1) Extract references
            no_refs_text, placeholders = extract_citations(input_text)
            # 2) minimal rewriting
            partially_rewritten = minimal_rewriting(no_refs_text, p_syn=p_syn, p_trans=p_trans)
            # 3) restore references
            final_text = restore_citations(partially_rewritten, placeholders)

        # New counts
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
