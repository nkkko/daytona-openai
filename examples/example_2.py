from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI
import logging

# Configure root logger to show debug messages
# logging.basicConfig(level=logging.DEBUG,
#                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

client = DaytonaOpenAI()

print("Example 1: Request without compute")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": "There is a 3x3 grid. In the top-left cell, there is a red square. In the center cell, there is a blue circle. In the bottom-right cell, there is a green triangle. If you rotate the grid 90 degrees clockwise, and 180 degrees counter clockwise, where will the red square be located?"}
    ],
)
print(response.choices[0].message.content)
print("\n")

print("Example 2: Compute-enabled request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": "There is a 3x3 grid. In the top-left cell, there is a red square. In the center cell, there is a blue circle. In the bottom-right cell, there is a green triangle. If you rotate the grid 90 degrees clockwise, and 180 degrees counterclockwise, where will the red square be located?"}
    ],
    compute=True
)
print(response.choices[0].message.content)
print("\n")