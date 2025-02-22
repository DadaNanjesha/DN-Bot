# utils/citation_utils.py
import re
import nltk
from nltk.tokenize import sent_tokenize
from utils.model_loaders import load_paraphrase_model

nltk.download('punkt', quiet=True)

# A refined regex to match typical APA-like references (e.g., (Karaman & Frazzoli, 2011, pp. 83–86))
CITATION_PATTERN = re.compile(
    r"\(\s*[A-Za-z&\-,\.\s]+(?:et al\.\s*)?,\s*\d{4}(?:,\s*(?:pp?\.\s*\d+(?:-\d+)?))?\s*\)"
)

def extract_citations(sentence):
    """Replace APA-like references with placeholders [[REF_x]] and return mapping."""
    references = CITATION_PATTERN.findall(sentence)
    mapping = {}
    replaced = sentence
    for i, ref in enumerate(references, start=1):
        placeholder = f"[[REF_{i}]]"
        mapping[placeholder] = ref
        replaced = replaced.replace(ref, placeholder, 1)
    return replaced, mapping

def restore_citations(paraphrased, mapping):
    """Restore original citations from placeholders."""
    result = paraphrased
    for placeholder, ref in mapping.items():
        result = result.replace(placeholder, ref)
    return result

def rewrite_sentence_preserving_citations(sentence):
    """
    Rewrite a single sentence using a T5-based paraphraser while preserving APA citations.
    """
    replaced, mapping = extract_citations(sentence)
    if not replaced.strip():
        return sentence

    prompt = (
        "Rewrite the following sentence to be more natural while preserving all details and references exactly. "
        "Do NOT remove, alter, or reposition placeholders like [[REF_x]].\n\n"
        f"Original: {replaced}"
    )

    paraphraser = load_paraphrase_model()
    output = paraphraser(
        prompt,
        do_sample=True,
        temperature=0.9,
        top_p=0.95,
        max_length=256,
        min_length=len(replaced.split()),
        max_new_tokens=256
    )
    paraphrased = output[0]["generated_text"].strip()
    final_sentence = restore_citations(paraphrased, mapping)
    return final_sentence

def rewrite_text_preserving_citations(original_text):
    """Rewrite input text sentence-by-sentence, preserving APA citations."""
    sentences = sent_tokenize(original_text)
    output_sentences = []
    for s in sentences:
        new_s = rewrite_sentence_preserving_citations(s)
        output_sentences.append(new_s)
    return " ".join(output_sentences)
