# Task 1: Text Summarization Tool
# Instructions: Create a tool that summarizes lengthy articles using Natural Language Processing (NLP).
# Library used: Hugging Face Transformers

# NOTE: Before running, you must install the library by typing this in the terminal:
# pip install transformers torch

from transformers import pipeline

def summarize_article(text):
    """
    Summarizes a given text using a pre-trained BART model.
    """
    print("Loading summarization model... (this may take a moment)")
    # Initialize the summarization pipeline
    # 'facebook/bart-large-cnn' is a powerful model for summarizing text
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Generate summary
    # max_length: limits how long the summary can be
    # min_length: ensures the summary isn't too short
    summary_result = summarizer(text, max_length=150, min_length=40, do_sample=False)

    return summary_result[0]['summary_text']

if __name__ == "__main__":
    # This is a sample text to test the tool
    article = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural intelligence displayed by animals including humans. 
    AI research has been defined as the field of study of intelligent agents, which refers to any system that perceives its environment and takes actions 
    that maximize its chance of achieving its goals. The term "artificial intelligence" had previously been used to describe machines that mimic and 
    display "human" cognitive skills that are associated with the human mind, such as "learning" and "problem-solving". This definition has since been 
    rejected by major AI researchers who now describe AI in terms of rationality and acting rationally, which does not limit how intelligence can be articulated.
    """
    
    print("\n--- Original Text ---")
    print(article)
    print("\n--- Generating Summary ---")
    
    # Call the function
    summary = summarize_article(article)
    
    print("\n--- Summary Result ---")
    print(summary)