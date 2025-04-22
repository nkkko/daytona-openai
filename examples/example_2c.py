from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI
import logging

# Configure root logger to show debug messages
# logging.basicConfig(level=logging.DEBUG,
#                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

client = DaytonaOpenAI()

print("Example 1: Request without compute")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": "Calculate 1.348911 * 1.348912"}
    ],
)
print(response.choices[0].message.content)
print("\n")

print("Example 2: Compute-enabled request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": "Calculate 1.348911 * 1.348912"}
    ],
    compute=True
)
print(response.choices[0].message.content)
print("\n")