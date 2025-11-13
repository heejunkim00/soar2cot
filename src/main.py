from pydantic import BaseModel, Field

# Import logging_config first to apply patches
import src.logging_config  # noqa: F401
from src.llms.messages import (
    get_next_message_anthropic,
    get_next_message_deepseek,
    get_next_message_gemini,
    get_next_message_openai,
    get_next_message_openrouter,
)
from src.llms.models import Model
from src.llms.structured import get_next_structure
from src.log import log
from src.models import GRID, Challenge, Example, Input


class InstructionsResponse(BaseModel):
    """Given the input / output examples, provide step by step instructions for how to transform the input grids into output grids."""

    instructions: str = Field(
        ...,
        description="Step by step instructions for how to transform the input grids into output grids.",
    )


class StructuredInstructionsResponse(BaseModel):
    """Given the input / output examples, provide analysis and instructions in two separate parts."""

    input_output_analysis: str = Field(
        ...,
        description="Detailed analysis of patterns observed in the input grids and output grids. Describe shapes, colors, positions, symmetries, relationships, and what changes between input and output.",
    )

    transformation_method: str = Field(
        ...,
        description="Step-by-step instructions for how to transform any input grid into the correct output grid. Must be clear, general, and applicable to all training examples.",
    )


class ReviseInstructionsResponse(BaseModel):
    """Given the input / output examples and the failed attempts following the previous instructions, provide revised step by step instructions for how to transform the input grids into output grids."""

    reasoning_for_why_old_instructions_are_wrong: str = Field(
        ...,
        description="Give the reasoning for why the old instructions were not correct and how you will improve them to make the revised instructions correct.",
    )
    revised_instructions: str = Field(
        ...,
        description="Revised step by step instructions for how to transform the input grids into output grids taking into account the failed attempts.",
    )


class GridResponse(BaseModel):
    """Using the instructions given, produce the correct output grid from the test input grid."""

    grid: list[list[int]] = Field(
        ...,
        description="The output grid which is the transform instructions applied to the test input grid.",
    )


# Import all prompts from centralized prompts.py
from src.prompts import (
    AGENT_FOLLOW_INSTRUCTIONS_PROMPT,
    INTUITIVE_PROMPT,
    INTUITIVE_PROMPT_WITH_CODE,
)

func_to_llm = {
    get_next_message_openai: {
        Model.gpt_4_5,
        Model.o3_mini,
        Model.gpt_4_o,
        Model.o3_mini_high,
        Model.o4_mini_high,
        Model.o4_mini,
        Model.gpt_5,
        Model.gpt_5_pro,
    },
    get_next_message_anthropic: {Model.sonnet_3_7, Model.sonnet_3_5, Model.sonnet_4_5},
    get_next_message_gemini: {Model.gemini_2_5},
    get_next_message_deepseek: {Model.deepseek_chat, Model.deepseek_reasoner},
    get_next_message_openrouter: {
        Model.openrouter_sonnet_3_7_thinking,
        Model.openrouter_gemini_2_5_free,
        Model.openrouter_deepseek_3_free,
        Model.openrouter_gemini_2_5,
        Model.openrouter_deepseek_r1,
        Model.openrouter_grok_v3,
        Model.openrouter_quasar_alpha,
        Model.openrouter_sonnet_3_7,
        Model.openrouter_optimus_alpha,
        Model.openrouter_deepseek_r1_free,
    },
}

# now reverse the map
llm_to_func = {}
for f, _llms in func_to_llm.items():
    for _llm in _llms:
        llm_to_func[_llm] = f


class PromptResponse(BaseModel):
    create_instructions: str
    follow_instructions: str


def contents_from_grid(grid: GRID, grid_label: str, include_base64: bool) -> list[dict]:
    contents = [
        {
            "type": "input_text",
            "text": f"{grid_label}:\n{Challenge.grid_to_str(grid=grid)}",
        },
    ]
    if include_base64:
        try:
            contents.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{Challenge.grid_to_base64(grid=grid)}",
                    "detail": "high",
                }
            )
        except (
            ValueError,
            TypeError,
        ) as e:
            log.error("Error generating base64 image", error=str(e))
            pass

    return contents


def contents_from_example(
    example: Example,
    attempt: GRID | None,
    example_number: int,
    include_base64: bool,
    use_diffs: bool,
) -> list[dict]:
    if not attempt:
        attempt_messages = []
    else:
        if attempt != example.output:
            failed_str = "Failed Output Attempt"
        else:
            failed_str = "Successful Output Attempt"
        attempt_messages = contents_from_grid(
            grid=attempt, grid_label=failed_str, include_base64=include_base64
        )

        # Add diff notation if use_diffs is True and attempt failed
        if use_diffs and attempt != example.output:
            from src.run import generate_grid_diff

            diff_text = generate_grid_diff(
                expected_grid=example.output, actual_grid=attempt
            )
            attempt_messages.append(
                {
                    "type": "input_text",
                    "text": f"\nDifference notation (actual→expected):\n{diff_text}",
                }
            )

    return [
        {"type": "input_text", "text": f"Training Example {example_number}\n"},
        *contents_from_grid(
            grid=example.input, grid_label="Input", include_base64=include_base64
        ),
        *contents_from_grid(
            grid=example.output, grid_label="Output", include_base64=include_base64
        ),
        *attempt_messages,
    ]


def contents_from_challenge(
    training_examples: list[Example],
    training_example_attempts: list[GRID] | None,
    test_inputs: list[Input],
    include_base64: bool,
    use_diffs: bool,
) -> list[dict]:
    contents: list[dict] = [{"type": "input_text", "text": "--Training Examples--"}]
    for i, example in enumerate(training_examples):
        attempt = (
            None if not training_example_attempts else training_example_attempts[i]
        )
        contents.extend(
            contents_from_example(
                example=example,
                attempt=attempt,
                example_number=i + 1,
                include_base64=include_base64,
                use_diffs=use_diffs,
            )
        )
    if len(test_inputs) == 1:
        test_inputs_str = "Test Input"
    else:
        test_inputs_str = "Test Inputs"
    contents.append(
        {
            "type": "input_text",
            "text": f"--End of Training Examples--\n\n--{test_inputs_str}--",
        }
    )
    for i, test in enumerate(test_inputs):
        contents.extend(
            contents_from_grid(
                grid=test.input,
                grid_label=f"Test Input {i + 1}",
                include_base64=include_base64,
            )
        )
    contents.append({"type": "input_text", "text": f"--End of {test_inputs_str}--"})
    return contents


def contents_from_python_code(python_code: str) -> list[dict]:
    """
    Convert Python code to prompt content format.

    Args:
        python_code: The Python code string (SOAR transform function)

    Returns:
        List of content dictionaries with the formatted Python code section
    """
    return [
        {
            "type": "input_text",
            "text": "\n--Reference Python Solution--",
        },
        {
            "type": "input_text",
            "text": f"```python\n{python_code}\n```",
        },
        {
            "type": "input_text",
            "text": "--End of Reference Solution--",
        },
    ]


def contents_from_challenge_with_code(
    training_examples: list[Example],
    training_example_attempts: list[GRID] | None,
    test_inputs: list[Input],
    include_base64: bool,
    use_diffs: bool,
    python_code: str,
) -> list[dict]:
    """
    Generate prompt contents from challenge data WITH Python code reference (for SOAR).

    This extends the standard contents_from_challenge by adding a Python code section
    after the test inputs.

    Args:
        training_examples: List of training input-output pairs
        training_example_attempts: Optional predicted outputs for training examples
        test_inputs: List of test inputs
        include_base64: Whether to include base64 image representations
        use_diffs: Whether to include grid diffs
        python_code: The reference Python implementation (SOAR code)

    Returns:
        List of content dictionaries including training examples, test inputs, and Python code
    """
    # Get standard challenge contents
    contents = contents_from_challenge(
        training_examples=training_examples,
        training_example_attempts=training_example_attempts,
        test_inputs=test_inputs,
        include_base64=include_base64,
        use_diffs=use_diffs,
    )

    # Add Python code section
    contents.extend(contents_from_python_code(python_code))

    return contents


PERFECT_PROMPT = """
These instructions are a guide to help you get the correct output grid.
If you think there is an error with the instructions that would cause you to get the wrong output grid, ignore that part of the instructions.
What is most important is that you get the exact correct output grid given the general pattern you observe.
""".strip()


async def output_grid_from_instructions(
    instructions: str,
    training_examples: list[Example],
    test_input_grid: GRID,
    model: Model,
    include_base64: bool,
    use_diffs: bool,
    is_perfect: bool,
) -> GRID:
    contents_from_examples: list[dict] = []
    for i, example in enumerate(training_examples):
        contents_from_examples.extend(
            contents_from_example(
                example=example,
                example_number=i + 1,
                include_base64=include_base64,
                attempt=None,
                use_diffs=use_diffs,
            )
        )
    perfect_messages = []
    if not is_perfect:
        perfect_messages = [{"type": "input_text", "text": PERFECT_PROMPT}]
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": AGENT_FOLLOW_INSTRUCTIONS_PROMPT,
                },
                {"type": "input_text", "text": f"\nInstructions:\n{instructions}"},
                *perfect_messages,
                *contents_from_examples,
                *contents_from_grid(
                    grid=test_input_grid,
                    grid_label="Test Input Grid",
                    include_base64=include_base64,
                ),
                {"type": "input_text", "text": "Test Output Grid:"},
            ],
        }
    ]
    grid_response, is_truncated = await get_next_structure(
        structure=GridResponse, messages=messages, model=model
    )
    return grid_response.grid, is_truncated
