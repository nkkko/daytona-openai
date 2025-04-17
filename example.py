from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI

load_dotenv()

client = DaytonaOpenAI()

print("Example 1: Standard request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Explain quantum computing basics in 3 sentences"}
    ]
)
print(response.choices[0].message.content)
print("\n")

print("Example 2: Compute-enabled request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Generate the first 10 prime numbers."}
    ],
    compute=True
)
print(response.choices[0].message.content)
print("\n")

print("Example 3: Complex computational request")
print("-" * 50)
response = client.completions.create(
    model="gpt-4o",
    prompt="Calculate me all the 2-digit Fibonacci numbers.",
    compute=True
)
print(response.choices[0].message.content)