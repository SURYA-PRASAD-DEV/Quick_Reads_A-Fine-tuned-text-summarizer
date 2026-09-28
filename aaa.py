import gradio as gr
import torch
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from gtts import gTTS  # Google Text-to-Speech
import os
from bs4 import BeautifulSoup  # For removing HTML tags

# Load fine-tuned T5 model and tokenizer
MODEL_PATH = r'mysummarization/results/checkpoint'
TOKENIZER_PATH = r'mysummarization/results'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)

# Load spaCy NLP model for concept understanding
nlp = spacy.load("en_core_web_sm")

# Preprocessing Function
def preprocess_text(text):
    """Cleans and prepares text for summarization."""
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\n+', ' ', text)  # Convert new lines to spaces
    return text.strip()

# Extract Key Points using TF-IDF & Named Entity Recognition (NER)
def extract_keypoints(text, top_n=5):
    """Extracts important keyphrases using TF-IDF and Named Entity Recognition (NER)."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    # Extract Named Entities (important people, places, events)
    named_entities = {ent.text for ent in doc.ents if ent.label_ in ["ORG", "GPE", "PERSON", "EVENT", "DATE", "MONEY", "CARDINAL"]}

    # Apply TF-IDF vectorization
    vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2))
    X = vectorizer.fit_transform(sentences)

    # Get top key phrases
    feature_names = vectorizer.get_feature_names_out()
    scores = X.sum(axis=0).A1
    keyphrases = [feature_names[i] for i in scores.argsort()[-top_n:][::-1]]

    # Combine NER key points with TF-IDF key phrases
    keypoints = list(named_entities) + keyphrases
    return list(set(keypoints))  # Remove duplicates

# Context-Aware Chunking for Better Coherence
def chunk_text(text, max_words=500, overlap=1):
    """Splits large documents into smaller chunks, ensuring better logical breaks and context overlap."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    chunks = []
    current_chunk = []
    word_count = 0
    
    for sent in sentences:
        words = sent.split()
        if word_count + len(words) > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = current_chunk[-overlap:]  # Keep overlap for context
                word_count = sum(len(sent.split()) for sent in current_chunk)
        current_chunk.append(sent)
        word_count += len(words)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# Post-processing Function for Logical Flow & Formatting
def format_summary(summary):
    """Formats the summary with colored numbers and underlined key entities for better readability."""
    summary = summary.strip()
    summary = re.sub(r'\s+', ' ', summary)

    # Highlight numbers with blue color
    summary = re.sub(r'(\d+(\.\d+)?)', r'<span style="color:blue;">\1</span>', summary)

    # Underline key entities (Organizations, People, Places, Events)
    doc = nlp(summary)
    for ent in doc.ents:
        if ent.label_ in ["ORG", "GPE", "PERSON", "EVENT", "DATE"]:
            summary = summary.replace(ent.text, f'<u>{ent.text}</u>')

    if not summary.endswith("."):
        summary += "."

    return summary

# Multi-Pass Summarization for Cross-Chunk Coherence
def abstractive_summarize(text, max_length=150, min_length=50, num_beams=4, top_p=0.8, temperature=0.7):
    """Generates a structured, coherent abstractive summary with improved logical flow."""
    if not text.strip():
        return "⚠️ Please enter valid text to summarize.", "", 0, 0

    clean_text = preprocess_text(text)
    input_word_count = len(clean_text.split())
    keypoints = extract_keypoints(clean_text)

    text_chunks = chunk_text(clean_text)
    chunk_summaries = []

    for chunk in text_chunks:
        inputs = tokenizer(
            "summarize: " + chunk,
            return_tensors='pt',
            max_length=512,
            truncation=True,
            padding="max_length"
        ).to(device)

        summary_ids = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            top_p=top_p,
            temperature=temperature,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            length_penalty=1.1,
            early_stopping=True
        )

        chunk_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        chunk_summaries.append(chunk_summary)

    # Final summarization of all chunk summaries
    combined_text = " ".join(chunk_summaries)
    final_summary = abstractive_summarize(combined_text, max_length=max_length, min_length=min_length, num_beams=num_beams, top_p=top_p, temperature=temperature)[0]
    
    return final_summary, ", ".join(keypoints), input_word_count, len(final_summary.split())

# Gradio Interface
interface = gr.Interface(
    fn=abstractive_summarize,
    inputs=[
        gr.Textbox(lines=10, placeholder='Enter the text here...', label='Input Text'),
        gr.Slider(minimum=50, maximum=300, value=150, step=10, label="Max Summary Length"),
        gr.Slider(minimum=20, maximum=100, value=50, step=5, label="Min Summary Length"),
        gr.Slider(minimum=1, maximum=10, value=4, step=1, label="Beam Search Width (num_beams)"),
        gr.Slider(minimum=0.5, maximum=1.0, value=0.8, step=0.05, label="Top-p Sampling (Diversity)"),
        gr.Slider(minimum=0.3, maximum=1.0, value=0.7, step=0.1, label="Temperature (Randomness)")
    ],
    outputs=[
        gr.HTML(label='Summarized Text'),
        gr.Textbox(label='Key Points'),
        gr.Textbox(label='Input Text Word Count'),
        gr.Textbox(label='Summary Word Count')
    ],
    title="Quick-Reads: A Text Summarizer"
)

# Launch Gradio app
interface.launch()
