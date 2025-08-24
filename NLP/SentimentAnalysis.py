from openai import OpenAI
from dotenv import load_dotenv
import os
# import openai


# Initialize OpenAI client
client = OpenAI(
    
    api_key='sk-proj--9pfuP7zINTJOSerqmgAyxQBuiY1TJ5diU73R0m0yGPvsSeID-Db91Nu38MjRGIVCxD5pjywrT3BlbkFJkcKW64sBIMgyoQ7kKmGSRQihyTzRpyMMiwlEI1DMqEl7VyoXpQHwAzprkjc_DzU7Gi41MwLyUA'
)

def analyze_sentiment(text):
    """
    Analyzes the sentiment of the provided text using OpenAI API.
    
    Args:
        text (str): The text to analyze.
        
    Returns:
        str: Sentiment result (Positive, Negative, Neutral).
    """
    try:
        # Define the prompt for sentiment analysis
        prompt = (
            "Analyze the sentiment of the following text and classify it as Positive, Negative, or Neutral:\n\n"
            f"Text: {text}\n\n"
            "Sentiment:"
        )
        
        # Call OpenAI's completion endpoint
        response = client.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=10,  # Short response for classification
            temperature=0,  # Ensure deterministic output
        )
        
        # Extract and return the sentiment result
        sentiment = response.choices[0].text.strip()
        return sentiment

    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == "__main__":
    # Example usage
    example_text = "I love how user-friendly and powerful OpenAI's tools are!"
    
    # Analyze the sentiment
    sentiment_result = analyze_sentiment(example_text)
    print(f"Text: {example_text}")
    print(f"Sentiment: {sentiment_result}")
