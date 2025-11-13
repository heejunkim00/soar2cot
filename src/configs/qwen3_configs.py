from src.configs.models import Model, RunConfig, Step, StepRevisionPool

# Local Qwen3-32B model
qwen3_model = Model.local_qwen3_32b

# Qwen3-32B configuration (similar to GPT-OSS but optimized)
local_qwen3_32b_config = RunConfig(
    final_follow_model=qwen3_model,
    final_follow_times=3,
    max_concurrent_tasks=4,  # Same as GPT-OSS
    steps=[
        Step(
            instruction_model=qwen3_model,
            follow_model=qwen3_model,
            times=2,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=qwen3_model,
            follow_model=qwen3_model,
            times=2,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=qwen3_model,
            follow_model=qwen3_model,
            times=3,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=3,
            instruction_model=qwen3_model,
            follow_model=qwen3_model,
            times=2,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
