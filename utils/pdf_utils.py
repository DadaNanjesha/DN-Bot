# utils/pdf_utils.py
import fitz
from io import BytesIO
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt', quiet=True)

def extract_text_from_pdf(pdf_bytes):
    """Extract text from all pages of a PDF."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    all_text = ""
    for page in doc:
        all_text += page.get_text("text") + "\n"
    doc.close()
    return all_text

def word_count(text):
    return len(word_tokenize(text))

def generate_annotated_pdf(pdf_bytes, classification_map):
    """Generate an annotated PDF with color-coded highlights for AI text."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    legend_text = (
        "Color Legend:\n"
        "• Red: AI-generated\n"
        "• Orange: AI-generated & AI-refined\n"
        "• Light Blue: Human-written & AI-refined\n\n"
        "Note: Sentences classified as 'Human-written' are not highlighted."
    )
    legend_page = doc.new_page(pno=0)
    legend_page.insert_text((72, 72), legend_text, fontsize=14, fontname="helv")

    def hex_to_rgb_float(hex_color):
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        return (r, g, b)

    COLOR_MAPPING = {
        "AI-generated": "#ffcccc",
        "AI-generated & AI-refined": "#ffe5cc",
        "Human-written & AI-refined": "#e6f2ff"
    }

    for sentence, label in classification_map.items():
        if label == "Human-written":
            continue
        color_hex = COLOR_MAPPING.get(label)
        if not color_hex:
            continue
        color = hex_to_rgb_float(color_hex)
        for page in doc:
            rects = page.search_for(sentence)
            for rect in rects:
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=color)
                annot.update()

    out_bytes = doc.write()
    doc.close()
    return BytesIO(out_bytes)
