import openai

# Set your OpenAI API key
openai.api_key = "your-api-key"

def extract_topics(text, num_topics=3):
    prompt = f"""
    Extract {num_topics} main topics from the following text. 
    Return the result as a simple list of topics:

    Text: \"\"\"{text}\"\"\"
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # or "gpt-4"
        messages=[
            {"role": "system", "content": "You are a helpful assistant who extracts key topics."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response['choices'][0]['message']['content']

# Example usage
sample_text = """
OpenAI has released several groundbreaking models in recent years, including GPT-3 and GPT-4. These models are used 
in a variety of applications such as chatbots, document summarization, code generation, and customer support. 
Researchers are exploring ways to reduce hallucinations, improve factual accuracy, and enhance fine-tuning methods.
"""

topics = extract_topics(sample_text)
print("Extracted Topics:")
print(topics)
