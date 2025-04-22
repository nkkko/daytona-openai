from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI

load_dotenv()

client = DaytonaOpenAI()

print("Example 4: Complex computational request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    prompt="How many letters 'r' are in the word 'strawberry'?",
    compute=True
)
print(response.choices[0].message.content)
print("\n")