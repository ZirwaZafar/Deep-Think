import time
import re
import logging
from transformers import pipeline
from collections import Counter
import requests
from bs4 import BeautifulSoup
import os

# Set up logging
logging.basicConfig(filename="summarizer.log", level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load pre-trained model for summarization
summarizer = pipeline("summarization")

def clean_text(text):
    """
    Clean the input text by removing unwanted characters and extra spaces.
    """
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespaces
    text = re.sub(r'[^\w\s.,!?]', '', text)  # Remove non-alphanumeric characters
    return text.strip()

def preprocess_text(text):
    """
    Preprocess text by cleaning and removing stopwords.
    """
    text = clean_text(text)
    # For the sake of this example, let's assume stop words are just basic filler words.
    stopwords = set(["the", "and", "is", "in", "to", "a", "of", "that", "on", "with"])
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(filtered_words)

def summarize_text(text, min_len=25, max_len=150):
    """
    Summarize the text using a transformer-based model.

    :param text: The text to be summarized.
    :param min_len: Minimum length of the summary.
    :param max_len: Maximum length of the summary.
    :return: Summary of the text.
    """
    if not text.strip():
        return "Error: The provided text is empty. Please enter valid text."
    
    try:
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Error: {e}"

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

def display_summary(original_text, summary):
    """
    Display the original and summarized text.
    """
    print(f"Original Text:\n{original_text}\n")
    print(f"Summarized Text:\n{summary}\n")

def log_summary(original_text, summary):
    """
    Log the summary to a file.
    """
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

def summarize_from_file(file_path):
    """
    Summarize text from a file.
    """
    if not os.path.exists(file_path):
        return "Error: File does not exist."

    with open(file_path, 'r') as file:
        text = file.read()
    text = clean_text(text)
    return summarize_text(text)

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
            text = summarize_from_file(file_path)
        else:
            print("Invalid choice. Try again.")
            continue

        print("\nProcessing...\n")
        start_time = time.time()
        
        processed_text = preprocess_text(text)
        summary = summarize_text(processed_text)
        
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        
        print(f"Summary generated in {time_taken} seconds.")
        
        display_summary(text, summary)
        log_summary(text, summary)
        
        # Save summary to file
        save_to_file = input("Do you want to save the summary to a file? (yes/no): ")
        if save_to_file.lower() == "yes":
            filename = input("Enter the file name (with .txt extension): ")
            save_summary_to_file(summary, filename)

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
