import ollama


def analyze_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of the provided text using Qwen2.5 via Ollama.

    Args:
        text (str): The text to analyze.

    Returns:
        str: Sentiment result (Positive, Negative, Neutral).
    """
    try:
        prompt = (
            "Analyze the sentiment of the following text and classify it strictly as "
            "Positive, Negative, or Neutral:\n\n"
            f"Text: {text}\n\n"
            "Sentiment:"
        )

        response = ollama.chat(
            model="qwen2.5:latest", messages=[{"role": "user", "content": prompt}]
        )

        sentiment = response["message"]["content"].strip()
        return sentiment

    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    example_text = "I love how user-friendly and powerful OpenAI's tools are!"
    sentiment_result = analyze_sentiment(example_text)
    print(f"Text: {example_text}")
    print(f"Sentiment: {sentiment_result}")
