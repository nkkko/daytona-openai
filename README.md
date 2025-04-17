# Daytona OpenAI

Enhanced OpenAI client with Daytona sandbox execution capabilities.

## Overview

The `daytona-openai-demo` package extends the standard OpenAI client to add compute capabilities that execute code in a Daytona sandbox environment. This allows you to automatically generate and execute code based on natural language prompts.

## Installation

```bash
pip install daytona-openai-demo
```

All required dependencies (`openai` and `daytona-sdk`) will be automatically installed.

## Usage

```python
from daytona_openai_demo import DaytonaOpenAI

# Initialize the client
client = DaytonaOpenAI()

# Standard OpenAI request
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Explain quantum computing basics in 3 sentences"}
    ]
)
print(response.choices[0].message.content)

# Compute-enabled request (automatically generates and executes code)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": "Generate the first 10 prime numbers."}
    ],
    compute=True
)
print(response.choices[0].message.content)
```

## Features

- Seamless integration with the OpenAI Python client
- Automatic code generation and execution in a secure sandbox
- Support for both chat completions and legacy completions APIs
- All standard OpenAI parameters are supported

## Requirements

- Python 3.10 or higher
- OpenAI API key
- Daytona API key
