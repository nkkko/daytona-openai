from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI

load_dotenv()

client = DaytonaOpenAI()

print("Example 5: Complex computational request")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": """
Here is the graph to operate on:
The graph has the following edges:
cfcd208495 -> cfcd208495
cfcd208495 -> 1679091c5a
cfcd208495 -> c81e728d9d
cfcd208495 -> c4ca4238a0
c4ca4238a0 -> c9f0f895fb
c4ca4238a0 -> 45c48cce2e
c4ca4238a0 -> eccbc87e4b
c4ca4238a0 -> c9f0f895fb
c81e728d9d -> 45c48cce2e
c81e728d9d -> eccbc87e4b
c81e728d9d -> eccbc87e4b
c81e728d9d -> c9f0f895fb
eccbc87e4b -> d3d9446802
eccbc87e4b -> d3d9446802
eccbc87e4b -> a87ff679a2
eccbc87e4b -> c4ca4238a0
a87ff679a2 -> 1679091c5a
a87ff679a2 -> eccbc87e4b
a87ff679a2 -> cfcd208495
a87ff679a2 -> e4da3b7fbb
e4da3b7fbb -> c4ca4238a0
e4da3b7fbb -> 1679091c5a
e4da3b7fbb -> 1679091c5a
e4da3b7fbb -> 45c48cce2e
1679091c5a -> 8f14e45fce
1679091c5a -> 8f14e45fce
1679091c5a -> e4da3b7fbb
1679091c5a -> a87ff679a2
8f14e45fce -> 45c48cce2e
8f14e45fce -> e4da3b7fbb
8f14e45fce -> 8f14e45fce
8f14e45fce -> cfcd208495
c9f0f895fb -> eccbc87e4b
c9f0f895fb -> cfcd208495
c9f0f895fb -> eccbc87e4b
c9f0f895fb -> 8f14e45fce
45c48cce2e -> cfcd208495
45c48cce2e -> 1679091c5a
45c48cce2e -> a87ff679a2
45c48cce2e -> a87ff679a2
d3d9446802 -> 8f14e45fce
d3d9446802 -> d3d9446802
d3d9446802 -> 45c48cce2e
d3d9446802 -> e4da3b7fbb


Operation:
Find the parents of node 8f14e45fce.

You should immediately return the set of nodes that the operation results in, with no additional text. Return your final answer as a list of nodes in the very last line of your response. For example, if the operation returns the set of nodes [node1, node2, node3], your response should be:
Final Answer: [node1, node2, node3]
If the operation returns the empty set, your response should be:
Final Answer: []
"""}
    ],
)
print(response.choices[0].message.content)
print("\n")