import ollama


def extract_topics(text: str, num_topics: int = 3) -> str:
    """
    Extracts main topics from the given text using Qwen2.5 via Ollama.

    Args:
        text (str): Input text.
        num_topics (int): Number of topics to extract.

    Returns:
        str: Extracted topics as a simple list.
    """
    prompt = f"""
    Extract {num_topics} main topics from the following text. 
    Return the result as a simple list of topics:

    Text: \"\"\"{text}\"\"\"
    """

    response = ollama.chat(
        model="qwen2.5:latest",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who extracts key topics.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return response["message"]["content"].strip()


# Example usage
if __name__ == "__main__":
    sample_text = """
    OpenAI has released several groundbreaking models in recent years, including GPT-3 and GPT-4. 
    These models are used in a variety of applications such as chatbots, document summarization, 
    code generation, and customer support. Researchers are exploring ways to reduce hallucinations, 
    improve factual accuracy, and enhance fine-tuning methods.
    """

    topics = extract_topics(sample_text)
    print("Extracted Topics:")
    print(topics)
