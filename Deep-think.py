import time
import re
import logging
import requests
import os
from transformers import pipeline
from collections import Counter
from bs4 import BeautifulSoup
from textblob import TextBlob
from readability import Readability

# Set up logging
logging.basicConfig(filename="summarizer.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load pre-trained models for summarization
summarizers = {
    'bart': pipeline("summarization", model="facebook/bart-large-cnn"),
    't5': pipeline("summarization", model="t5-small"),
}

# Helper functions for cleaning and preprocessing text
def clean_text(text):
    """
    Clean the input text by removing unwanted characters and extra spaces.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove non-alphanumeric characters
    return text.strip()

def preprocess_text(text, remove_stopwords=False):
    """
    Preprocess text by cleaning and removing stopwords if specified.
    """
    text = clean_text(text)
    if remove_stopwords:
        stopwords = set(["the", "and", "is", "in", "to", "a", "of", "that", "on", "with", "for", "it", "as"])
        words = text.split()
        text = " ".join([word for word in words if word.lower() not in stopwords])
    return text

# Summarization function
def summarize_text(text, model='bart', min_len=25, max_len=150):
    """
    Summarize the text using the specified model.
    :param text: The text to be summarized.
    :param model: The model to use for summarization ('bart' or 't5').
    :param min_len: Minimum length of the summary.
    :param max_len: Maximum length of the summary.
    :return: Summarized text.
    """
    if not text.strip():
        return "Error: The provided text is empty. Please enter valid text."
    
    try:
        summary = summarizers[model](text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"

# Sentiment analysis function
def analyze_sentiment(text):
    """
    Analyze the sentiment of the text using TextBlob.
    """
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

# Readability analysis function
def analyze_readability(text):
    """
    Analyze the readability of the text using the Readability library.
    """
    r = Readability(text)
    flesch_score = r.flesch()
    return flesch_score.score

# Word frequency analysis function
def analyze_word_frequency(text):
    """
    Analyze the word frequency of the text.
    """
    words = text.split()
    word_count = Counter(words)
    return word_count.most_common(10)

# Web scraping function to extract text from a URL
def get_text_from_url(url):
    """
    Extract and clean text from a URL.
    """
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return clean_text(text)
    except Exception as e:
        return f"Error: {e}"

# Display functions
def display_summary(original_text, summary):
    """
    Display the original and summarized text.
    """
    print(f"Original Text:\n{original_text}\n")
    print(f"Summarized Text:\n{summary}\n")

def log_summary(original_text, summary, model_used):
    """
    Log the summary to a file.
    """
    logging.info(f"Model used: {model_used}")
    logging.info("Original Text: %s", original_text[:100] + '...')
    logging.info("Summary: %s", summary[:100] + '...')
    logging.info("="*100)

def save_summary_to_file(summary, filename):
    """
    Save the summary to a text file.
    """
    with open(filename, 'w') as file:
        file.write(summary)

def rate_summary(summary):
    """
    Ask the user to rate the summary.
    """
    rating = input("Rate the summary (1-5): ")
    return int(rating)

# File and folder management functions
def create_folder(folder_name):
    """
    Create a folder to store summaries if it does not exist.
    """
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_summary_in_folder(summary, folder_name, filename):
    """
    Save the summary in a specific folder.
    """
    create_folder(folder_name)
    file_path = os.path.join(folder_name, filename)
    save_summary_to_file(summary, file_path)

# CLI functions for interactive input
def main():
    print("Welcome to the Deep-Think Text Summarizer!\n")
    
    while True:
        print("Choose input method:")
        print("1. Enter text")
        print("2. Summarize from URL")
        print("3. Summarize from file")
        choice = input("Enter your choice (1/2/3): ")
        
        if choice == "1":
            text = input("\nEnter the text you want to summarize: ")
        elif choice == "2":
            url = input("\nEnter the URL: ")
            text = get_text_from_url(url)
        elif choice == "3":
            file_path = input("\nEnter the file path: ")
            with open(file_path, 'r') as file:
                text = file.read()
        else:
            print("Invalid choice. Try again.")
            continue

        print("\nChoose model for summarization:")
        print("1. BART (facebook/bart-large-cnn)")
        print("2. T5 (t5-small)")
        model_choice = input("Enter your choice (1/2): ")
        model = 'bart' if model_choice == "1" else 't5'

        print("\nProcessing...\n")
        start_time = time.time()

        processed_text = preprocess_text(text, remove_stopwords=True)
        summary = summarize_text(processed_text, model=model)
        
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        
        print(f"Summary generated in {time_taken} seconds.")
        
        display_summary(text, summary)
        log_summary(text, summary, model)
        
        # Save summary to file
        save_to_file = input("Do you want to save the summary to a file? (yes/no): ")
        if save_to_file.lower() == "yes":
            filename = input("Enter the file name (with .txt extension): ")
            save_summary_to_file(summary, filename)

        # Save summary in folder
        save_in_folder = input("Would you like to save this summary in a specific folder? (yes/no): ")
        if save_in_folder.lower() == "yes":
            folder_name = input("Enter the folder name: ")
            save_summary_in_folder(summary, folder_name, filename)

        # Analyze sentiment
        polarity, subjectivity = analyze_sentiment(summary)
        print(f"Sentiment - Polarity: {polarity}, Subjectivity: {subjectivity}")
        
        # Analyze readability
        readability_score = analyze_readability(summary)
        print(f"Readability Score (Flesch): {readability_score}")

        # Word frequency analysis
        word_freq = analyze_word_frequency(summary)
        print("Most common words in summary:", word_freq)
        
        # Rating
        rating = rate_summary(summary)
        print(f"Thank you for rating the summary: {rating}/5")

        # Option to continue
        again = input("\nWould you like to summarize more text? (yes/no): ")
        if again.lower() != "yes":
            print("Goodbye!")
            break

if __name__ == "__main__":
    main()
