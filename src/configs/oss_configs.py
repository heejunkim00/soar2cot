from src.configs.models import Model, RunConfig, Step, StepRevision, StepRevisionPool

oss_model = Model.openrouter_gpt_oss_120b

oss_config = RunConfig(
    final_follow_model=oss_model,
    final_follow_times=5,
    max_concurrent_tasks=40,
    steps=[
        Step(
            instruction_model=oss_model,
            follow_model=oss_model,
            times=10,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=oss_model,
            follow_model=oss_model,
            times=20,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=5,
            instruction_model=oss_model,
            follow_model=oss_model,
            times=5,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)

# Local GPT-OSS 20B configuration
local_gpt_oss_20b_model = Model.local_gpt_oss_20b

local_gpt_oss_20b_config = RunConfig(
    final_follow_model=local_gpt_oss_20b_model,
    final_follow_times=3,
    max_concurrent_tasks=4,  # Reduced to 4 to prevent vLLM overload
    steps=[
        Step(
            instruction_model=local_gpt_oss_20b_model,
            follow_model=local_gpt_oss_20b_model,
            times=2,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=local_gpt_oss_20b_model,
            follow_model=local_gpt_oss_20b_model,
            times=2,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        Step(
            instruction_model=local_gpt_oss_20b_model,
            follow_model=local_gpt_oss_20b_model,
            times=3,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
        StepRevisionPool(
            top_scores_used=3,
            instruction_model=local_gpt_oss_20b_model,
            follow_model=local_gpt_oss_20b_model,
            times=2,
            timeout_secs=300,
            include_base64=False,
            use_diffs=True,
        ),
    ],
)
