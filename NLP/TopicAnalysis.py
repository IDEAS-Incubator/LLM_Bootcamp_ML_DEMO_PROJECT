from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

your_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=your_api_key)


def extract_topics(text, num_topics=3):
    prompt = f"""
    Extract {num_topics} main topics from the following text. 
    Return the result as a simple list of topics:

    Text: \"\"\"{text}\"\"\"
    """

    response = client.chat.completions.create(
        model="gpt-4o",  # or "gpt-4"
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant who extracts key topics.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
    )

    return response.choices[0].message.content


# Example usage
sample_text = """
OpenAI has released several groundbreaking models in recent years, including GPT-3 and GPT-4. These models are used 
in a variety of applications such as chatbots, document summarization, code generation, and customer support. 
Researchers are exploring ways to reduce hallucinations, improve factual accuracy, and enhance fine-tuning methods.
"""

topics = extract_topics(sample_text)
print("Extracted Topics:")
print(topics)
