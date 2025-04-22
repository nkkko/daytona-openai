from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI

load_dotenv()

client = DaytonaOpenAI()

print("Example 3: Complex computational request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-mini",
    prompt="Calculate me all the 2-digit Fibonacci numbers. Then sum them up.",
    compute=True
)
print(response.choices[0].message.content)
print("\n")