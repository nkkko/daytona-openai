import re
from daytona_sdk import DaytonaConfig
import httpx
from typing import Any, TypeVar, Union, Mapping, Optional


from openai import OpenAI
from openai._types import Timeout, NotGiven, NOT_GIVEN

T = TypeVar('T')

class DaytonaOpenAI(OpenAI):
    """Enhanced OpenAI client with Daytona sandbox execution capabilities.

    This client extends the standard OpenAI client to add compute capabilities
    that execute code in a Daytona sandbox environment.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        base_url: Optional[Union[str, httpx.URL]] = None,
        websocket_base_url: Optional[Union[str, httpx.URL]] = None,
        timeout: Union[float, Timeout, None, NotGiven] = NOT_GIVEN,
        max_retries: int = 2,
        default_headers: Optional[Mapping[str, str]] = None,
        default_query: Optional[Mapping[str, object]] = None,
        http_client: Optional[httpx.Client] = None,
        daytona_config: DaytonaConfig = None,
        **kwargs
    ):
        """Construct a new synchronous OpenAI client instance with Daytona.

        This automatically infers the following arguments from their corresponding environment variables if they are not provided:
        - `api_key` from `OPENAI_API_KEY`
        - `organization` from `OPENAI_ORG_ID`
        - `project` from `OPENAI_PROJECT_ID`
        - `daytona_config` from `DAYTONA_API_KEY`, `DAYTONA_API_URL` and `DAYTONA_TARGET` environment variables
        """
        super().__init__(
            api_key=api_key,
            organization=organization,
            project=project,
            base_url=base_url,
            websocket_base_url=websocket_base_url,
            timeout=timeout,
            max_retries=max_retries,
            default_headers=default_headers,
            default_query=default_query,
            http_client=http_client,
            **kwargs
        )

        from daytona_sdk import Daytona
        self.daytona = Daytona(daytona_config)

        # Wrap methods that return model responses
        self._wrap_chat_completions()
        self._wrap_completions()

    def _wrap_chat_completions(self) -> None:
        """Wrap the chat.completions.create method to add compute functionality.

        This adds a 'compute' parameter that, when set to True, will execute the generated code
        in a Daytona sandbox and return the results.
        """
        original_create = self.chat.completions.create

        def wrapped_create(*args, compute: bool = False, **kwargs):
            """Enhanced chat.completions.create with compute capability.

            Args:
                *args: Positional arguments to pass to the original method.
                compute: When True, executes the generated code in a Daytona sandbox.
                **kwargs: Keyword arguments to pass to the original method.

            Returns:
                The completion response or the compute response if compute=True.
            """

            if not compute:
                return original_create(*args, **kwargs)
            else:
                # Extract prompt from messages
                messages = kwargs.get('messages', [])
                if not messages and args:
                    for arg in args:
                        if isinstance(arg, (list, tuple)) and all(isinstance(m, dict) for m in arg):
                            messages = arg
                            break

                if not messages:
                    raise ValueError("No messages provided for compute request")

                # Extract the user's message(s)
                user_messages = [msg for msg in messages if msg.get('role') == 'user']
                if not user_messages:
                    raise ValueError("No user messages found for compute request")

                # Use the last user message as the prompt
                last_user_message = user_messages[-1]
                prompt_content = last_user_message.get('content', '')

                if isinstance(prompt_content, list):
                    text_parts = []
                    for part in prompt_content:
                        if isinstance(part, dict) and part.get('type') == 'text':
                            text_parts.append(part.get('text', ''))
                    prompt = ' '.join(text_parts)
                else:
                    prompt = prompt_content

                model = kwargs.get('model', 'gpt-4')

                return self._compute_request(prompt, model)

        self.chat.completions.create = wrapped_create

    def _wrap_completions(self) -> None:
        """Wrap the completions.create method to add compute functionality.

        This adds a 'compute' parameter that, when set to True, will execute the generated code
        in a Daytona sandbox and return the results.
        """
        if hasattr(self, 'completions') and hasattr(self.completions, 'create'):
            original_create = self.completions.create

            def wrapped_create(*args, compute: bool = False, **kwargs):
                """Enhanced completions.create with compute capability.

                Args:
                    *args: Positional arguments to pass to the original method.
                    compute: When True, executes the generated code in a Daytona sandbox.
                    **kwargs: Keyword arguments to pass to the original method.

                Returns:
                    The completion response or the compute response if compute=True.
                """

                if not compute:
                    return original_create(*args, **kwargs)
                else:
                    prompt = kwargs.get('prompt', '')
                    if not prompt and args:
                        for arg in args:
                            if isinstance(arg, str):
                                prompt = arg
                                break

                    if not prompt:
                        raise ValueError("No prompt provided for compute request")

                    model = kwargs.get('model', 'gpt-4')

                    return self._compute_request(prompt, model)

            self.completions.create = wrapped_create

    def _compute_request(self, prompt: str, model: str) -> Any:
        """Handle request with code generation and execution.

        This method takes a prompt, generates code to solve the problem described in the prompt,
        executes the code in a Daytona sandbox, and returns a final response that includes
        both the code and its execution results.

        Args:
            prompt: The user's prompt describing the problem to solve.
            model: The OpenAI model to use for code generation and final response.

        Returns:
            The final response from the model after code execution.
        """
        # 1. Generate code from the prompt
        code_generation_prompt = self._create_code_generation_prompt(prompt)
        code_response = self.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": code_generation_prompt}
            ]
        )

        # 2. Extract code from the response
        generated_code = code_response.choices[0].message.content
        clean_code = self._extract_code(generated_code)

        if not clean_code:
            return "Failed to generate valid executable code for your request."

        # 3. Execute the code in Daytona sandbox
        execution_result = self._run_in_sandbox(clean_code)

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

        final_response = self.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": final_prompt}
            ]
        )

        return final_response

    def _create_code_generation_prompt(self, original_prompt: str) -> str:
        """Create a prompt specifically designed to generate executable code.

        Args:
            original_prompt: The user's original prompt.

        Returns:
            A modified prompt that instructs the model to generate executable Python code.
        """
        return f"""Write Python code to solve this: {original_prompt}.
        Requirements:
        - Output MUST be executable Python 3.10+ code
        - Include print() statements for results
        - No markdown formatting
        - Validate inputs/outputs
        - Code must be complete and self-contained"""

    def _extract_code(self, response: str) -> str:
        """Extract executable Python code from model response.

        Args:
            response: The model's response containing code.

        Returns:
            The extracted Python code ready for execution.
        """
        # Look for code blocks with ```python...``` format
        code_blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', response, re.DOTALL)

        if code_blocks:
            return code_blocks[0].strip()

        # If no code blocks found, try to extract just Python code from the imports down
        # This is a fallback for when the model doesn't use markdown formatting
        # lines = response.split('\n')
        # code_lines = []
        # in_code = False

        # for line in lines:
        #     if line.strip().startswith('import ') or line.strip().startswith('from '):
        #         in_code = True

        #     if in_code:
        #         code_lines.append(line)

        # if code_lines:
        #     return '\n'.join(code_lines)

        # If still no code found, return the original response
        # The sandbox will likely fail, but we'll handle that
        return response

    def _run_in_sandbox(self, code: str) -> str:
        """Execute code in Daytona sandbox and return the output.

        Args:
            code: The Python code to execute.

        Returns:
            The output from executing the code, or an error message if execution failed.
        """
        try:
            sandbox = self.daytona.create()
            response = sandbox.process.code_run(code)

            if response.exit_code != 0:
                raise Exception(f"Code execution failed with exit code {response.exit_code}")

            self.daytona.remove(sandbox)

            return response.result
        except Exception as e:
            return f"Error executing code: {str(e)}"