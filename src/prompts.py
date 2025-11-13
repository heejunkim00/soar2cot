"""
ARC Puzzle Solving Prompts
모든 프롬프트를 중앙에서 관리합니다.
"""

# Example instructions (optional)
USE_EXAMPLE_INSTRUCTIONS = False

EXAMPLE_INSTRUCTIONS = """
Here are example instructions that have worked in the past:
<example instructions>
1. Find the background color and ignore that.
2. Take the biggest contiguous shape general shape. This is the outline for the output grid.
3. Fill that largest shape like Tetris with the smaller shapes in ways that they fit together. You will not need to rotate any of the smaller shapes. They each will slot in so that the inner shape is filled and they will also fill in the gaps on the outside of the inner shape.
4. You will be left with a rectangle the height and width of the biggest starting shape.
</example instructions>
"""

EXAMPLE_INSTRUCTIONS_STR = (
    "" if not USE_EXAMPLE_INSTRUCTIONS else f"\n\n{EXAMPLE_INSTRUCTIONS}\n\n"
)


# Main instruction generation prompt (original, without code)
INTUITIVE_PROMPT = f"""
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Find the common pattern that transforms each input grid into its corresponding output grid, based on the training examples below.

Your task is to write clear instructions that describe this transformation pattern. These instructions must:
- Apply consistently to ALL training examples (the same rule works for every input→output pair)
- Be general enough to work on new test cases
- Be intuitive and easy to understand
- Describe the pattern without referencing specific example numbers or positions

The transformation pattern should be simple and logical - these puzzles are designed to have elegant, intuitive solutions that humans can readily grasp.

Write your instructions as a clear, step-by-step process that someone could follow to transform any input grid into the correct output grid.
{EXAMPLE_INSTRUCTIONS_STR}
Here are the training examples and test input grids:
"""


# Main instruction generation prompt WITH Python code reference (for SOAR data)
INTUITIVE_PROMPT_WITH_CODE = f"""
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Find the common pattern that transforms each input grid into its corresponding output grid, based on the training examples below.

Your task is to write clear instructions that describe this transformation pattern. These instructions must:
- Apply consistently to ALL training examples (the same rule works for every input→output pair)
- Be general enough to work on new test cases
- Be intuitive and easy to understand
- Describe the pattern without referencing specific example numbers or positions

The transformation pattern should be simple and logical - these puzzles are designed to have elegant, intuitive solutions that humans can readily grasp.

Additionally, you will be provided with a reference Python implementation that solves this task.
Use this code to better understand the transformation pattern, but write your instructions in natural language that a human can follow.

Write your instructions as a clear, step-by-step process that someone could follow to transform any input grid into the correct output grid.
{EXAMPLE_INSTRUCTIONS_STR}
Here are the training examples and test input grids:
"""


# Structured instruction generation prompt WITH Python code reference
# This version asks for separate analysis and transformation sections
STRUCTURED_PROMPT_WITH_CODE = f"""
You are participating in a puzzle solving competition. You are an expert at solving puzzles.

Your task is to analyze the transformation pattern in TWO distinct parts.

IMPORTANT: You will provide your answer in two separate fields:
1. "input_output_analysis" - Your detailed analysis of the patterns
2. "transformation_method" - Your step-by-step transformation instructions

---

## PART 1: INPUT/OUTPUT ANALYSIS
(This goes in the "input_output_analysis" field)

Carefully examine all the training examples and provide a detailed analysis:
- What patterns do you observe in the INPUT grids? (shapes, colors, positions, symmetries, sizes)
- What patterns do you observe in the OUTPUT grids? (structure, organization, size, arrangement)
- What relationships exist between inputs and outputs? (scaling, rotation, filtering, combining)
- What stays the same between input and output? What changes?
- Are there any special cases or edge conditions?

Be thorough and specific in your observations. This analysis will guide your transformation method.

---

## PART 2: TRANSFORMATION METHOD
(This goes in the "transformation_method" field)

Based on your analysis above, write clear step-by-step instructions that describe how to transform ANY input grid into the correct output grid:
- The instructions must apply consistently to ALL training examples (the same rule works for every input→output pair)
- Be general enough to work on new test cases you haven't seen
- Be intuitive and easy to follow for a human
- Describe the exact process without referencing specific example numbers
- Number your steps clearly (1. 2. 3. etc.)

The transformation pattern should be simple and logical - these puzzles are designed to have elegant, intuitive solutions that humans can readily grasp.

---

Additionally, you will be provided with a reference Python implementation that solves this task.
Use this code to better understand the transformation pattern, but write your instructions in natural language that a human can follow.
{EXAMPLE_INSTRUCTIONS_STR}
Here are the training examples and test input grids:
"""


# Instruction revision prompt
REVISION_PROMPT = """
Your previous instructions were applied to the training input grids, but they did not produce the correct output grids.

Below you'll see what outputs were generated when following your instructions. Compare these incorrect outputs with the correct outputs to identify where your instructions went wrong.

Based on this feedback, provide updated instructions that correctly describe the transformation pattern. Your revised instructions must:
- Fix the specific errors you observe
- Still work correctly for ALL training examples
- Remain clear, intuitive, and general

Analyze the differences between the incorrect outputs and the correct outputs to understand the true pattern, then write improved instructions.
""".strip()


# Multiple instruction synthesis prompt
SYNTHESIS_PROMPT = """
Multiple expert puzzle solvers have attempted to describe the transformation pattern for these grids. Each attempt captured some aspects correctly but failed in other ways.

Below you'll find:
- Each set of proposed instructions
- The outputs produced when following those instructions
- How those outputs differ from the correct answers

Your task is to analyze why each approach partially failed and synthesize a complete, correct set of instructions.

By examining multiple flawed attempts, you can:
- Identify what each approach got right
- Understand what each approach missed
- Recognize common misconceptions about the pattern
- Build comprehensive instructions that avoid all these pitfalls

Study the patterns of success and failure across all attempts, then write instructions that correctly describe the complete transformation rule that works for ALL training examples.

Your final instructions should be clear, intuitive, and capture the true underlying pattern.
""".strip()


# Instruction following prompt
AGENT_FOLLOW_INSTRUCTIONS_PROMPT = """
You are an expert puzzle solver in a competition.

You will receive:
1. Step-by-step instructions for transforming input grids into output grids
2. Training examples showing these instructions applied correctly
3. A test input grid to solve

Your task: Apply the given instructions precisely to transform the test input grid into its output grid.

The training examples demonstrate how the instructions work - use them to understand the pattern, then follow the exact same process for the test input.
""".strip()
