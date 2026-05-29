PIPELINE_DEFAULTS = {
    "semantic_breakpoint_percentile": {
        "value": 85,
        "used_in": "src/chunking/semantic_chunker.py",
        "reason": "Used as the default threshold for detecting large semantic distance spikes.",
        "status": "tunable_default",
    },
    "semantic_breakpoint_std_multiplier": {
        "value": 1,
        "used_in": "src/chunking/semantic_chunker.py",
        "reason": "In standard_deviation mode, threshold is mean + (1 × std) of cosine distances.",
        "status": "tunable_default",
    },
    "llm_system_prompt_token_estimate": {
        "value": 100,
        "used_in": "src/metrics/metrics.py",
        "reason": "Used only for approximate API token usage estimation.",
        "status": "estimate",
    },
    "tiktoken_fallback_encoding": {
        "value": "cl100k_base",
        "used_in": "src/metrics/metrics.py",
        "reason": "Fallback encoding when the selected model is unknown to tiktoken.",
        "status": "estimate",
    },
    "clod_io_model_mapping": {
        "value": "src/config/model_config.py MODEL_OPTIONS",
        "used_in": "src/config/model_config.py",
        "reason": "Maps UI display names to Clod.io /v1 model IDs (see provider /v1/models).",
        "status": "configured",
    },
}


def get_pipeline_defaults():
    return PIPELINE_DEFAULTS
