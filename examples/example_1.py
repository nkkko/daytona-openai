from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI

load_dotenv()

client = DaytonaOpenAI()

print("Example 1: Standard request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    messages=[
        {"role": "user", "content": "Explain quantum computing basics in 3 sentences"}
    ]
)
print(response.choices[0].message.content)
print("\n")