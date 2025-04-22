from dotenv import load_dotenv
from daytona_openai_demo import DaytonaOpenAI
import logging
from typing import Any

# Configure root logger to show debug messages
logging.basicConfig(level=logging.DEBUG, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

load_dotenv()

class DaytonaOpenAIWithDebug(DaytonaOpenAI):
    """Extension of DaytonaOpenAI that exposes additional debugging information."""
    
    def _compute_request(self, prompt: str, model: str) -> Any:
        """Override to expose intermediate results."""
        # Original function steps with added outputs
        print("\n===== COMPUTE DEBUGGING =====")
        print(f"Original prompt: {prompt}")
        
        # 1. Generate code from the prompt
        code_generation_prompt = self._create_code_generation_prompt(prompt)
        print(f"\nCode generation prompt:\n{code_generation_prompt}")
        
        code_response = self.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are an expert Python programmer."},
                {"role": "user", "content": code_generation_prompt}
            ]
        )
        
        # 2. Extract code from the response
        generated_code = code_response.choices[0].message.content
        clean_code = self._extract_code(generated_code)
        print(f"\nGenerated code:\n```python\n{clean_code}\n```")
        
        if not clean_code:
            print("Failed to generate valid executable code")
            return "Failed to generate valid executable code for your request."
        
        # 3. Execute the code in Daytona sandbox
        print("\nExecuting code in Daytona sandbox...")
        execution_result = self._run_in_sandbox(clean_code)
        print(f"\nExecution result:\n{execution_result}")
        
        # 4. Format results and get final response
        final_prompt = f"""
        Generated code to solve this problem: "{prompt}"

        Here's the code:
        ```python
        {clean_code}
        ```

        When executed, it produced this output:
        ```
        {execution_result}
        ```

        Please give me a final solution to the original problem while including the output, but dont include or focus on the code if that is not specificly asked from you.
        """
        print(f"\nFinal prompt (summarized):\n{final_prompt[:100]}...")
        
        final_response = self.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": final_prompt}
            ]
        )
        
        print("\n===== END COMPUTE DEBUGGING =====")
        return final_response


# Create client with debug features
client = DaytonaOpenAIWithDebug()

print("Example: Compute-enabled request with debugging")
print("-" * 50)
response = client.chat.completions.create(
    model="gpt-4.1-nano",
    messages=[
        {"role": "user", "content": "Calculate 1.348911 * 1.348912"}
    ],
    compute=True
)
print("\nFinal response:")
print("-" * 50)
print(response.choices[0].message.content)