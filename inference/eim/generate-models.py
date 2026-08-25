#!/usr/bin/env python3
"""Generates the EIM model table ICN serves.

Every number here has a stated source, because the RAM estimate is only as trustworthy as its
inputs and a plausible-looking guess is worse than an absent value.

  geometry      fetched from each model's Hugging Face config.json
  weightBytes   fetched from model.safetensors.index.json metadata.total_size, or summed from
                the file listing for single-file models. Measured rather than derived: a
                published checkpoint is not always bf16, and gpt-oss-20b ships in MXFP4 at
                13.8 GB where the parameter count would have implied 42 GB.
  toolCallParser  a claim about vLLM's parser for that model family, confirmed on real hardware
                only for Qwen3 so far. Models whose parser is unknown carry null, which the
                catalog surfaces as unusable rather than letting the agent loop fail silently.
  qualityScore  a curated estimate, not a measurement. The provenance string says so. Ranking
                these on Terminal-Bench was never run, and reporting a number as measured when
                it was not would be a fabrication.

Four models are gated: the three Llama variants and google/gemma-7b return HTTP 401 for
config.json without a token, which is exactly what EIM's catalog marks hf_token_required. Their
geometry comes from published model cards instead, flagged per entry, and they are unusable
without a credential regardless.

Run from the repository root:

    python3 inference/eim/generate-models.py > inference/eim/models.json
"""

import json
import sys

GIB = 1024**3

# Context served, capped at the model's trained maximum. On the CPU backend the KV cache is sized
# from this and vLLM refuses to start when it does not fit, so it is bounded rather than maximal.
#
# 32768 up to 40B, now measured rather than guessed: a 30B mixture of experts at that context needs
# 79 GiB against this host's 234 GiB budget, where the earlier 16384 saved 6 GiB it did not need to.
# What forced the change is the other side of the window. An agent's system prompt and tool schemas
# already occupy roughly 6000 tokens before the user types anything, so 16384 left too little for
# the conversation and produced a request the engine refused outright.
#
# Beyond 40B it stays small: the only such model does not fit this host at any context.
def served_context(total_parameters, max_position_embeddings):
    wanted = 32_768 if total_parameters <= 40_000_000_000 else 8_192
    return min(wanted, max_position_embeddings)


# Tensor parallelism. EIM's selector rejects a size above the host's NUMA node count, and
# `magnitude-icn doctor` reports that maximum. Two is right for the dual-socket target host; a
# single-node host must use one, which always works.
def tensor_parallel_size(total_parameters):
    return 2 if total_parameters >= 8_000_000_000 else 1


MODELS = [
    # --- serveable: not gated, and vLLM has a tool-call parser for the family -----------------
    {
        "canonicalName": "Qwen/Qwen3-30B-A3B",
        "displayName": "Qwen3 30B-A3B",
        "description": "Mixture-of-experts model from Alibaba's Qwen3 series, 30B total and 3B active parameters.",
        "releaseDate": "2025-04-29",
        "license": "Apache-2.0",
        "totalParameters": 30_500_000_000,
        "activeParameters": 3_300_000_000,
        "weightBytes": 61_100_000_000,
        "layers": 48, "kvHeads": 4, "headDim": 128, "maxPositionEmbeddings": 40_960,
        "slidingWindow": None, "vision": False,
        "architecture": "Qwen3MoeForCausalLM",
        "toolCallParser": "hermes", "reasoningParser": "qwen3",
        "reasoningEfforts": ["none", "high"], "defaultEffort": "high",
        # Best of this catalog for agentic work: a mixture of experts is also the fastest shape
        # on a memory-bandwidth-bound host.
        "qualityScore": 28.0,
        "gated": False,
    },
    {
        "canonicalName": "openai/gpt-oss-20b",
        "displayName": "GPT-OSS 20B",
        "description": "Open-weight reasoning model from OpenAI's gpt-oss series, tuned for chain-of-thought and tool use.",
        "releaseDate": "2025-08-05",
        "license": "Apache-2.0",
        "totalParameters": 21_000_000_000,
        "activeParameters": 3_600_000_000,
        # Ships in MXFP4, so the measured size is a third of what bf16 would imply.
        "weightBytes": 13_800_000_000,
        "layers": 24, "kvHeads": 8, "headDim": 64, "maxPositionEmbeddings": 131_072,
        "slidingWindow": 128, "vision": False,
        "architecture": "GptOssForCausalLM",
        "toolCallParser": "openai", "reasoningParser": "openai_gptoss",
        "reasoningEfforts": ["none", "low", "high"], "defaultEffort": "high",
        "qualityScore": 25.0,
        "gated": False,
    },
    {
        "canonicalName": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "displayName": "Qwen3-VL 30B-A3B Instruct",
        "description": "Vision-language mixture-of-experts model from Qwen3-VL, instruction-tuned for image-and-text chat.",
        "releaseDate": "2025-10-15",
        "license": "Apache-2.0",
        "totalParameters": 31_100_000_000,
        "activeParameters": 3_300_000_000,
        "weightBytes": 62_100_000_000,
        "layers": 48, "kvHeads": 4, "headDim": 128, "maxPositionEmbeddings": 262_144,
        "slidingWindow": None, "vision": True,
        "architecture": "Qwen3VLMoeForConditionalGeneration",
        "toolCallParser": "hermes", "reasoningParser": "qwen3",
        "reasoningEfforts": ["none", "high"], "defaultEffort": "high",
        "qualityScore": 24.0,
        "gated": False,
    },
    {
        "canonicalName": "Qwen/Qwen3-8B",
        "displayName": "Qwen3 8B",
        "description": "Dense model from Alibaba's Qwen3 series with dual thinking and non-thinking modes.",
        "releaseDate": "2025-04-29",
        "license": "Apache-2.0",
        "totalParameters": 8_200_000_000,
        "activeParameters": 8_200_000_000,
        "weightBytes": 16_400_000_000,
        "layers": 36, "kvHeads": 8, "headDim": 128, "maxPositionEmbeddings": 40_960,
        "slidingWindow": None, "vision": False,
        "architecture": "Qwen3ForCausalLM",
        "toolCallParser": "hermes", "reasoningParser": "qwen3",
        "reasoningEfforts": ["none", "high"], "defaultEffort": "high",
        # The only entry whose tool calling and reasoning were confirmed on real hardware.
        "qualityScore": 22.0,
        "gated": False,
    },
    {
        "canonicalName": "Qwen/Qwen3-4B",
        "displayName": "Qwen3 4B",
        "description": "Compact dense model from Alibaba's Qwen3 series with dual thinking and non-thinking modes.",
        "releaseDate": "2025-04-29",
        "license": "Apache-2.0",
        "totalParameters": 4_000_000_000,
        "activeParameters": 4_000_000_000,
        "weightBytes": 8_000_000_000,
        "layers": 36, "kvHeads": 8, "headDim": 128, "maxPositionEmbeddings": 40_960,
        "slidingWindow": None, "vision": False,
        "architecture": "Qwen3ForCausalLM",
        "toolCallParser": "hermes", "reasoningParser": "qwen3",
        "reasoningEfforts": ["none", "high"], "defaultEffort": "high",
        "qualityScore": 14.0,
        "gated": False,
    },
    {
        "canonicalName": "ibm-granite/granite-3.2-2b-instruct",
        "displayName": "Granite 3.2 2B Instruct",
        "description": "Instruction-tuned model from IBM's Granite 3.2 series with a toggle for extended reasoning.",
        "releaseDate": "2025-02-26",
        "license": "Apache-2.0",
        "totalParameters": 2_500_000_000,
        "activeParameters": 2_500_000_000,
        "weightBytes": 5_100_000_000,
        # head_dim absent from config.json; hidden_size 2048 over 32 attention heads.
        "layers": 40, "kvHeads": 8, "headDim": 64, "maxPositionEmbeddings": 131_072,
        "slidingWindow": None, "vision": False,
        "architecture": "GraniteForCausalLM",
        "toolCallParser": "granite", "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        # Small models score near zero on agentic benchmarks whatever their chat quality.
        "qualityScore": 4.0,
        "gated": False,
    },

    # --- gated: config.json returns 401, geometry from published model cards ------------------
    {
        "canonicalName": "meta-llama/Llama-3.1-8B-Instruct",
        "displayName": "Llama 3.1 8B Instruct",
        "description": "Instruction-tuned chat model from Meta's Llama 3.1 family, for multilingual dialogue and tool use.",
        "releaseDate": "2024-07-23",
        "license": "Llama 3.1 Community License",
        "totalParameters": 8_030_000_000,
        "activeParameters": 8_030_000_000,
        "weightBytes": 16_060_000_000,
        "layers": 32, "kvHeads": 8, "headDim": 128, "maxPositionEmbeddings": 131_072,
        "slidingWindow": None, "vision": False,
        "architecture": "LlamaForCausalLM",
        "toolCallParser": "llama3_json", "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 6.0,
        "gated": True,
    },
    {
        "canonicalName": "meta-llama/Llama-3.2-3B-Instruct",
        "displayName": "Llama 3.2 3B Instruct",
        "description": "Lightweight instruction-tuned model from Meta's Llama 3.2 family.",
        "releaseDate": "2024-09-25",
        "license": "Llama 3.2 Community License",
        "totalParameters": 3_210_000_000,
        "activeParameters": 3_210_000_000,
        "weightBytes": 6_420_000_000,
        "layers": 28, "kvHeads": 8, "headDim": 128, "maxPositionEmbeddings": 131_072,
        "slidingWindow": None, "vision": False,
        "architecture": "LlamaForCausalLM",
        "toolCallParser": "llama3_json", "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 2.0,
        "gated": True,
    },
    {
        "canonicalName": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
        "displayName": "Llama 4 Scout 17B-16E Instruct",
        "description": "Multimodal mixture-of-experts model from Meta's Llama 4 family, 17B active across 16 experts.",
        "releaseDate": "2025-04-05",
        "license": "Llama 4 Community License",
        "totalParameters": 109_000_000_000,
        "activeParameters": 17_000_000_000,
        "weightBytes": 218_000_000_000,
        "layers": 48, "kvHeads": 8, "headDim": 128, "maxPositionEmbeddings": 131_072,
        "slidingWindow": None, "vision": True,
        "architecture": "Llama4ForConditionalGeneration",
        "toolCallParser": "llama4_pythonic", "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        # Rejected by the memory estimate on a 256 GB host regardless: 218 GB of weights alone.
        "qualityScore": 20.0,
        "gated": True,
    },
    {
        "canonicalName": "google/gemma-7b",
        "displayName": "Gemma 7B",
        "description": "Base language model from Google's Gemma family, for text generation and fine-tuning.",
        "releaseDate": "2024-02-21",
        "license": "Gemma Terms of Use",
        "totalParameters": 8_540_000_000,
        "activeParameters": 8_540_000_000,
        "weightBytes": 17_080_000_000,
        "layers": 28, "kvHeads": 16, "headDim": 256, "maxPositionEmbeddings": 8_192,
        "slidingWindow": None, "vision": False,
        "architecture": "GemmaForCausalLM",
        # A base model: no chat template at all, so it can never carry a tool call.
        "toolCallParser": None, "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 0.0,
        "gated": True,
    },

    # --- no tool-call parser: listed, but the catalog marks them unusable ---------------------
    {
        "canonicalName": "zai-org/glm-4-9b-hf",
        "displayName": "GLM-4 9B",
        "description": "Base language model from Z.ai's GLM-4 series, a general-purpose foundation model.",
        "releaseDate": "2024-06-04",
        "license": "GLM-4 Community License",
        "totalParameters": 9_400_000_000,
        "activeParameters": 9_400_000_000,
        "weightBytes": 18_800_000_000,
        "layers": 40, "kvHeads": 2, "headDim": 128, "maxPositionEmbeddings": 8_192,
        "slidingWindow": None, "vision": False,
        "architecture": "GlmForCausalLM",
        # A base model, like gemma-7b.
        "toolCallParser": None, "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 0.0,
        "gated": False,
    },
    {
        "canonicalName": "mistralai/Mistral-7B-Instruct-v0.2",
        "displayName": "Mistral 7B Instruct v0.2",
        "description": "Instruction-tuned model from Mistral AI for general-purpose chat.",
        "releaseDate": "2023-12-11",
        "license": "Apache-2.0",
        "totalParameters": 7_240_000_000,
        "activeParameters": 7_240_000_000,
        "weightBytes": 14_500_000_000,
        "layers": 32, "kvHeads": 8, "headDim": 128, "maxPositionEmbeddings": 32_768,
        "slidingWindow": None, "vision": False,
        "architecture": "MistralForCausalLM",
        # v0.3 introduced a function-calling template; v0.2 has none.
        "toolCallParser": None, "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 0.0,
        "gated": False,
    },
    {
        "canonicalName": "google/gemma-4-E4B-it",
        "displayName": "Gemma E4B Instruct",
        "description": "Multimodal instruction-tuned model from Google's Gemma family, supporting text and image inputs.",
        "releaseDate": "2026-04-02",
        "license": "Apache-2.0",
        "totalParameters": 8_000_000_000,
        "activeParameters": 8_000_000_000,
        # Summed from the repository file listing; this model ships without a shard index.
        "weightBytes": 16_000_000_000,
        "layers": 42, "kvHeads": 2, "headDim": 256, "maxPositionEmbeddings": 131_072,
        "slidingWindow": 512, "vision": True,
        "architecture": "Gemma4ForConditionalGeneration",
        # The Gemma family has no native function-calling template.
        "toolCallParser": None, "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 0.0,
        "gated": False,
    },
    {
        "canonicalName": "microsoft/Phi-4-reasoning",
        "displayName": "Phi-4 Reasoning",
        "description": "Reasoning-focused variant of Microsoft's Phi-4, tuned for multi-step problem solving.",
        "releaseDate": "2025-04-30",
        "license": "MIT",
        "totalParameters": 14_700_000_000,
        "activeParameters": 14_700_000_000,
        "weightBytes": 29_300_000_000,
        "layers": 40, "kvHeads": 10, "headDim": 128, "maxPositionEmbeddings": 32_768,
        "slidingWindow": None, "vision": False,
        "architecture": "Phi3ForCausalLM",
        # No standard vLLM parser for this variant.
        "toolCallParser": None, "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 0.0,
        "gated": False,
    },
    {
        "canonicalName": "microsoft/Phi-4-multimodal-instruct",
        "displayName": "Phi-4 Multimodal Instruct",
        "description": "Multimodal instruction-tuned model from Microsoft's Phi-4 family, handling text, image and audio.",
        "releaseDate": "2025-02-26",
        "license": "MIT",
        "totalParameters": 5_600_000_000,
        "activeParameters": 5_600_000_000,
        "weightBytes": 11_100_000_000,
        # head_dim absent from config.json; hidden_size 3072 over 24 attention heads.
        "layers": 32, "kvHeads": 8, "headDim": 128, "maxPositionEmbeddings": 131_072,
        "slidingWindow": 262_144, "vision": True,
        "architecture": "Phi4MMForCausalLM",
        "toolCallParser": None, "reasoningParser": None,
        "reasoningEfforts": [], "defaultEffort": None,
        "qualityScore": 0.0,
        "gated": False,
    },
]

IMAGE_TAG_PREFIX = "magnitude-eim-xeon"
IMAGE_TAG = "v1"


def sanitize(value):
    """EIM's own OCI component sanitization: lowercase, and anything else becomes a hyphen."""
    lowered = value.lower().replace("/", "-")
    return "".join(c if (c.isalnum() or c in "._-") else "-" for c in lowered).strip("._-")


def entry(model):
    total = model["totalParameters"]
    context = served_context(total, model["maxPositionEmbeddings"])
    tensor_parallel = tensor_parallel_size(total)
    image_name = "%s-%s" % (IMAGE_TAG_PREFIX, sanitize(model["canonicalName"]))
    catalog_model_id = sanitize(model["canonicalName"])
    return {
        "configurationId": "eim-%s-ctx%d" % (catalog_model_id, context),
        "packageId": "eim--%s--%s" % (model["canonicalName"].replace("/", "--"), IMAGE_TAG),
        "catalogModelId": catalog_model_id,
        "catalogVariantId": "vllm-bf16:tp%d" % tensor_parallel,
        "canonicalName": model["canonicalName"],
        "displayName": model["displayName"],
        "variantLabel": "bf16",
        "description": model["description"],
        "releaseDate": model["releaseDate"],
        "license": model["license"],
        "qualityScore": model["qualityScore"],
        "qualityScoreProvenance": "curated_estimate_2026_08",
        "image": "%s:%s" % (image_name, IMAGE_TAG),
        "eimProfileId": "vllm-xeon-bf16-tp%d" % tensor_parallel,
        "tensorParallelSize": tensor_parallel,
        "contextTokens": context,
        "geometry": {
            "totalParameters": total,
            "activeParameters": model["activeParameters"],
            "weightBytes": model["weightBytes"],
            "numHiddenLayers": model["layers"],
            "numKeyValueHeads": model["kvHeads"],
            "headDim": model["headDim"],
            "maxPositionEmbeddings": model["maxPositionEmbeddings"],
            "slidingWindow": model["slidingWindow"],
            "vision": model["vision"],
        },
        "architecture": model["architecture"],
        "slidingWindowTokens": model["slidingWindow"] or 0,
        "reasoning": {
            "efforts": model["reasoningEfforts"],
            "defaultEffort": model["defaultEffort"],
        },
        "modalities": {
            "vision": model["vision"],
            "audio": "multimodal" in model["displayName"].lower(),
            "video": False,
        },
        "toolCallParser": model["toolCallParser"],
        "reasoningParser": model["reasoningParser"],
        "hfTokenRequired": model["gated"],
        "weightBytes": model["weightBytes"],
    }


def main():
    table = [entry(model) for model in MODELS]
    identifiers = [item["configurationId"] for item in table]
    if len(set(identifiers)) != len(identifiers):
        print("duplicate serving configuration identifiers", file=sys.stderr)
        return 1
    json.dump(table, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
