import nltk
from nltk.tokenize import sent_tokenize
from utils.model_loaders import load_detector_model

nltk.download('punkt', quiet=True)

def classify_text_hf(text, threshold=0.8):
    """
    Splits text into sentences, uses roberta-base-openai-detector to classify each sentence
    as AI-generated or human-written, returning a map of {sentence: label} and overall percentages.
    """
    detector = load_detector_model()
    sentences = sent_tokenize(text)
    results = detector(sentences, truncation=True)

    classification_map = {}
    counts = {
        "AI-generated": 0,
        "AI-generated & AI-refined": 0,
        "Human-written": 0,
        "Human-written & AI-refined": 0
    }

    for sentence, result in zip(sentences, results):
        label = result['label'].upper()  # "FAKE" or "REAL"
        score = result['score']
        if label == "FAKE":
            if score >= threshold:
                new_label = "AI-generated"
            else:
                new_label = "AI-generated & AI-refined"
        elif label == "REAL":
            if score >= threshold:
                new_label = "Human-written"
            else:
                new_label = "Human-written & AI-refined"
        else:
            new_label = "Human-written"
        classification_map[sentence] = new_label
        counts[new_label] += 1

    total = sum(counts.values())
    percentages = {
        cat: round((count / total)*100, 2) if total > 0 else 0
        for cat, count in counts.items()
    }
    return classification_map, percentages
