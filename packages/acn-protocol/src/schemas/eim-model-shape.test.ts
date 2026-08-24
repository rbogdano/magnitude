/**
 * Stage 0 of the EIM container-backend fork: prove that a container-backed,
 * safetensors/vLLM model can be represented in the existing local-model
 * protocol without changing any schema.
 *
 * Every value here is what `icn-eim` will synthesize at runtime:
 *   - `sha256` is a Docker image content digest, not a GGUF file digest
 *   - `path` is the in-container weight cache directory
 *   - the assessment comes from the RAM formula, not a llama.cpp FitReport
 *   - `performance` is a memory-bandwidth roofline estimate, not a measurement
 *
 * If any of these decode failures reappear, the fork's premise that the
 * TypeScript layer needs no schema change is false and the plan must change.
 */
import { describe, expect, it } from "vitest"
import { Option, Schema } from "effect"
import { installedLocalModels } from "./local-model-projection"
import {
  FitsModelAssessmentSchema,
  LocalModelsStateSchema,
  LocalModelSchema,
  MemoryAssessmentSchema,
  ModelCapabilitiesSchema,
  ProviderModelCatalogEntrySchema,
} from "./model-state"

const GIB = 1024 ** 3

/** Xeon 6767P / 256 GB, the target host. */
const PHYSICAL_CAPACITY_BYTES = 256 * 1000 ** 3
/** `icn_hardware::system_memory_thresholds`: reserve = max(10% of physical, 2 GiB). */
const ASSESS_RESERVE_BYTES = Math.max(Math.floor(PHYSICAL_CAPACITY_BYTES / 10), 2 * GIB)

/** A Docker image ID (`docker image inspect --format '{{.Id}}'`) with the `sha256:` prefix stripped. */
const IMAGE_DIGEST = "d0508be469accfdd25a1a97b5a50ece67db5c5ca23031cbed45fdbf8d0a7bb89"

/**
 * `assess_load_candidates` produces one candidate per parallel-sequence count and
 * `select_load_allocation` takes the largest that fits. `remainingBytes` must equal
 * `capacity - reserve - required` exactly, or `ModelAssessment::is_valid_for` rejects it
 * on the Rust side before it ever reaches the wire.
 */
const systemMemoryAssessment = (requiredBytes: number) => ({
  memoryDomainId: "system",
  capacityBytes: PHYSICAL_CAPACITY_BYTES,
  requiredBytes,
  compatibilityReserveBytes: ASSESS_RESERVE_BYTES,
  remainingBytes: PHYSICAL_CAPACITY_BYTES - ASSESS_RESERVE_BYTES - requiredBytes,
})

/**
 * Qwen3-30B-A3B: the fork's target model. Mixture of experts, so the RAM formula
 * charges *total* parameters (all experts stay resident) while the roofline uses
 * *active* parameters.
 */
const QWEN3_30B_A3B = {
  canonicalName: "Qwen/Qwen3-30B-A3B",
  catalogModelId: "qwen-qwen3-30b-a3b",
  catalogVariantId: "vllm-bf16:tp1",
  packageId: "eim--Qwen--Qwen3-30B-A3B--magnitude-eim-xeon-v1",
  totalParameters: 30_500_000_000,
  activeParameters: 3_300_000_000,
  contextLength: 32_768,
  weightBytes: 61_000_000_000,
  requiredBytes: 72_200_000_000,
  installedBytes: 61_000_000_000,
} as const

/** Wire (encoded) shape — what `icn-eim` will serialize from the catalog overlay. */
const capabilitiesInput = () => ({
  vision: false,
  tools: true,
  structuredOutput: true,
  // Qwen3 is dual-mode: `enable_thinking` in chat_template_kwargs.
  reasoning: { supported: true, efforts: ["none", "high"], defaultEffort: "high" },
})

const capabilities = () => Schema.decodeUnknownSync(ModelCapabilitiesSchema)(capabilitiesInput())

const standaloneBundle = () => ({
  _tag: "Standalone" as const,
  package: {
    id: QWEN3_30B_A3B.packageId,
    // vLLM downloads from the Hub; the fork prefetches the same repository.
    source: { _tag: "HuggingFace", repository: QWEN3_30B_A3B.canonicalName, revision: "main" },
    files: [{
      id: `${QWEN3_30B_A3B.packageId}/image`,
      // Relative to the mounted weight cache, in EIM's Local Directory layout.
      path: "Qwen/Qwen3-30B-A3B",
      role: "weights",
      sizeBytes: QWEN3_30B_A3B.weightBytes,
      tensorStorageBytes: QWEN3_30B_A3B.totalParameters * 2,
      sha256: IMAGE_DIGEST,
    }],
    relationships: [],
    properties: {
      format: "safetensors",
      quantization: "bf16",
      quantizationName: "BF16",
      architecture: "Qwen3MoeForCausalLM",
      maximumContextLength: 40_960,
      intrinsicModelId: QWEN3_30B_A3B.canonicalName,
      intrinsicQualityId: "bf16",
    },
  },
})

const fitsAssessment = (requiredBytes: number) => ({
  _tag: "Fits" as const,
  assessmentId: `eim-assessment-${IMAGE_DIGEST.slice(0, 16)}`,
  environmentId: "xeon-vllm-cpu",
  profile: { contextLength: QWEN3_30B_A3B.contextLength },
  memory: {
    domains: [systemMemoryAssessment(requiredBytes)],
    totalRequiredBytes: requiredBytes,
    requiredSystemMemoryBytes: requiredBytes,
    systemUseState: {
      _tag: "WithinRecommendedHeadroom",
      recommendedHeadroomBytes: ASSESS_RESERVE_BYTES,
      predictedHeadroomBytes: PHYSICAL_CAPACITY_BYTES - ASSESS_RESERVE_BYTES - requiredBytes,
    },
    currentHeadroomState: { _tag: "NotObserved" },
  },
  // Roofline over 12-channel DDR5; the grid must end exactly at the serving context.
  performance: [
    { contextTokens: 512, lowerTokensPerSecond: 32, estimatedTokensPerSecond: 51, upperTokensPerSecond: 82, confidence: "low" },
    { contextTokens: 4_096, lowerTokensPerSecond: 30, estimatedTokensPerSecond: 48, upperTokensPerSecond: 77, confidence: "low" },
    { contextTokens: 16_384, lowerTokensPerSecond: 25, estimatedTokensPerSecond: 40, upperTokensPerSecond: 64, confidence: "low" },
    { contextTokens: QWEN3_30B_A3B.contextLength, lowerTokensPerSecond: 20, estimatedTokensPerSecond: 33, upperTokensPerSecond: 53, confidence: "low" },
  ],
})

const eimLocalModel = () => ({
  bundle: standaloneBundle(),
  presentation: {
    displayName: "Qwen3 30B-A3B",
    variantLabel: "bf16",
    description:
      "Mixture-of-experts language model from Alibaba's Qwen3 series (30B total / 3B active parameters).",
    license: "Apache-2.0",
  },
  downloadBytes: QWEN3_30B_A3B.weightBytes,
  catalogMembershipState: {
    _tag: "InCatalog",
    catalogData: {
      modelId: QWEN3_30B_A3B.catalogModelId,
      variantId: QWEN3_30B_A3B.catalogVariantId,
      releaseDate: "2025-04-29",
      parameterization: {
        architecture: "mixtureOfExperts",
        totalParameters: QWEN3_30B_A3B.totalParameters,
        activeParameters: QWEN3_30B_A3B.activeParameters,
      },
      intelligenceScore: 31.4,
      intelligenceScoreSource: "curated_public_benchmarks_2026_08",
      // Every EIM profile is bf16, so there is no fidelity axis to rank on.
      fidelityRank: 100,
      quantizationAware: false,
      qualityNotes: [
        "Served by vLLM on Intel Xeon via an EIM container.",
        "Tool calling uses vLLM's hermes parser.",
      ],
    },
  },
  // "Installed" means the image is present AND the weight cache holds a complete manifest.
  acquisitionState: {
    _tag: "Installed",
    installedBytes: QWEN3_30B_A3B.installedBytes,
    packages: [{
      packageId: QWEN3_30B_A3B.packageId,
      path: "/var/lib/magnitude/eim/model-cache/Qwen/Qwen3-30B-A3B",
      origin: "Magnitude",
    }],
  },
  upgradeState: { _tag: "NotApplicable" },
  servingState: {
    _tag: "Assessed",
    configuration: {
      id: `eim-${IMAGE_DIGEST.slice(0, 16)}-ctx32768`,
      bundle: standaloneBundle(),
      profile: { contextLength: QWEN3_30B_A3B.contextLength },
    },
    capabilities: capabilitiesInput(),
    assessment: fitsAssessment(QWEN3_30B_A3B.requiredBytes),
    availabilityState: { _tag: "Selectable", providerModelId: `eim-${IMAGE_DIGEST.slice(0, 16)}-ctx32768` },
    recommendations: [{ id: "balanced-qwen3-30b-a3b", intent: "balanced", explanation: "Best overall fit for this host." }],
  },
})

describe("EIM container-backed model in the local-model protocol", () => {
  it("decodes a container-backed model as an ordinary LocalModel", () => {
    const model = Schema.decodeUnknownSync(LocalModelSchema)(eimLocalModel())

    expect(model.bundle._tag).toBe("Standalone")
    expect(model.acquisitionState._tag).toBe("Installed")
    expect(model.servingState._tag).toBe("Assessed")
  })

  it("accepts a Docker image digest as the package file sha256", () => {
    const model = Schema.decodeUnknownSync(LocalModelSchema)(eimLocalModel())
    const bundle = model.bundle

    expect(bundle._tag === "Standalone" && bundle.package.files[0]!.sha256).toBe(IMAGE_DIGEST)
  })

  it("carries container-sourced quality notes through the catalog projection", () => {
    const model = Schema.decodeUnknownSync(LocalModelSchema)(eimLocalModel())
    const membership = model.catalogMembershipState

    expect(membership._tag === "InCatalog" && membership.catalogData.qualityNotes).toEqual([
      "Served by vLLM on Intel Xeon via an EIM container.",
      "Tool calling uses vLLM's hermes parser.",
    ])
  })

  it("satisfies the installed-bundle package correspondence filter", () => {
    // The filter compares bundle package IDs against installed package IDs. A
    // mismatch is the most likely way the fork's synthesis goes wrong silently.
    const mismatched = eimLocalModel()
    mismatched.acquisitionState.packages[0]!.packageId = "eim--some--other--package"

    expect(() => Schema.decodeUnknownSync(LocalModelSchema)(mismatched)).toThrow()
  })

  it("decodes into LocalModelsState with a unique target-package identity", () => {
    const state = Schema.decodeUnknownSync(LocalModelsStateSchema)({
      inventoryState: { _tag: "Ready" },
      models: [eimLocalModel()],
      discoveryState: { _tag: "Ready", progress: [] },
    })

    expect(state.models).toHaveLength(1)
  })
})

describe("RAM-formula assessment shapes", () => {
  it("requires remainingBytes to equal capacity minus reserve minus required", () => {
    // Mirrors `ModelAssessment::is_valid_for`, which the Rust side enforces before
    // serializing. The TS schema only requires an integer, so this test documents
    // the arithmetic the synthesis must honor rather than a decode failure.
    const assessment = Schema.decodeUnknownSync(MemoryAssessmentSchema)(
      systemMemoryAssessment(QWEN3_30B_A3B.requiredBytes),
    )

    expect(assessment.remainingBytes).toBe(
      assessment.capacityBytes - assessment.compatibilityReserveBytes - assessment.requiredBytes,
    )
    expect(assessment.remainingBytes).toBeGreaterThan(0)
  })

  it("requires the roofline grid to end exactly at the serving context length", () => {
    const wireAssessment = {
      _tag: "Fits" as const,
      profile: { contextLength: QWEN3_30B_A3B.contextLength },
      configurationId: `eim-${IMAGE_DIGEST.slice(0, 16)}-ctx32768`,
      assessmentId: "eim-assessment-1",
      environmentId: "xeon-vllm-cpu",
      memory: [systemMemoryAssessment(QWEN3_30B_A3B.requiredBytes)],
      performance: fitsAssessment(QWEN3_30B_A3B.requiredBytes).performance,
    }

    expect(Schema.decodeUnknownSync(FitsModelAssessmentSchema)(wireAssessment).performance).toHaveLength(4)

    const truncated = { ...wireAssessment, performance: wireAssessment.performance.slice(0, 3) }
    expect(() => Schema.decodeUnknownSync(FitsModelAssessmentSchema)(truncated)).toThrow()
  })

  it("rejects a roofline grid that is not strictly ascending by context", () => {
    const unordered = {
      _tag: "Fits" as const,
      profile: { contextLength: QWEN3_30B_A3B.contextLength },
      configurationId: "eim-cfg",
      assessmentId: "eim-assessment-1",
      environmentId: "xeon-vllm-cpu",
      memory: [systemMemoryAssessment(QWEN3_30B_A3B.requiredBytes)],
      performance: [
        { contextTokens: 4_096, lowerTokensPerSecond: 30, estimatedTokensPerSecond: 48, upperTokensPerSecond: 77, confidence: "low" },
        { contextTokens: 512, lowerTokensPerSecond: 32, estimatedTokensPerSecond: 51, upperTokensPerSecond: 82, confidence: "low" },
        { contextTokens: QWEN3_30B_A3B.contextLength, lowerTokensPerSecond: 20, estimatedTokensPerSecond: 33, upperTokensPerSecond: 53, confidence: "low" },
      ],
    }

    expect(() => Schema.decodeUnknownSync(FitsModelAssessmentSchema)(unordered)).toThrow()
  })
})

describe("reasoning capabilities for EIM models", () => {
  it("accepts a dual-mode reasoning model declared from the catalog overlay", () => {
    expect(capabilities().reasoning.supported).toBe(true)
  })

  it("accepts a non-reasoning model with an empty effort set", () => {
    const nonReasoning = Schema.decodeUnknownSync(ModelCapabilitiesSchema)({
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: { supported: false, efforts: [] },
    })

    expect(nonReasoning.reasoning.defaultEffort).toStrictEqual(Option.none())
  })

  it("rejects a reasoning model whose default effort is outside its effort set", () => {
    expect(() => Schema.decodeUnknownSync(ModelCapabilitiesSchema)({
      vision: false,
      tools: true,
      structuredOutput: true,
      reasoning: { supported: true, efforts: ["none", "low"], defaultEffort: "high" },
    })).toThrow()
  })
})

/**
 * The requirement is that all 15 EIM models stay visible, with a badge and a note on
 * the ones the fork cannot serve. These tests pin the CURRENT behavior, which drops
 * them instead. Three projections filter on `assessment._tag === "Fits"`:
 *
 *   1. `installedLocalModels`                    (this package, asserted below)
 *   2. `localModelOptions`                       (packages/client-common/src/local-models/options.ts:56,84)
 *   3. `catalogLocalModels`                      (cli/src/features/model-menus/container.tsx:411)
 *
 * `buildModelsMenuEntries` (container.tsx:387) is the exception and shows the way: it
 * already tags a non-fitting *installed* model as `LocalStatus` rather than `Local`, so
 * a "visible but not selectable" row is an existing concept. It still returns nothing
 * for a model that was never installed, which is the state every gated or
 * tool-callless EIM model will be in.
 *
 * When the fork surfaces unavailable models, these expectations invert.
 */
describe("current projections drop models the fork must still display", () => {
  const doesNotFitModel = () => {
    const model = eimLocalModel()
    return {
      ...model,
      servingState: {
        ...model.servingState,
        // Llama-4-Scout on this host: 218 GB of bf16 weights against a ~230 GB budget.
        assessment: {
          _tag: "DoesNotFit",
          assessmentId: "eim-assessment-scout",
          environmentId: "xeon-vllm-cpu",
          memoryDomains: [systemMemoryAssessment(PHYSICAL_CAPACITY_BYTES)],
          totalRequiredBytes: 244_900_000_000,
          deficitBytes: 244_900_000_000 - (PHYSICAL_CAPACITY_BYTES - ASSESS_RESERVE_BYTES),
          limitingResource: "system memory",
        },
        availabilityState: {
          _tag: "Unavailable",
          failure: {
            code: "insufficient_resources",
            message: "Requires 245 GB; this host has 230 GB available.",
            retryable: false,
          },
        },
        recommendations: [],
      },
    }
  }

  it("still decodes a DoesNotFit model as a valid LocalModel", () => {
    const model = Schema.decodeUnknownSync(LocalModelSchema)(doesNotFitModel())

    expect(model.servingState._tag === "Assessed" && model.servingState.assessment._tag)
      .toBe("DoesNotFit")
  })

  it("omits it from installedLocalModels even though it is installed", () => {
    const state = Schema.decodeUnknownSync(LocalModelsStateSchema)({
      inventoryState: { _tag: "Ready" },
      models: [doesNotFitModel()],
      discoveryState: { _tag: "Ready", progress: [] },
    })

    // The model is Installed, so this omission is purely the `Fits` requirement.
    expect(state.models[0]!.acquisitionState._tag).toBe("Installed")
    expect(installedLocalModels(state)).toHaveLength(0)
  })

  it("includes it once the assessment fits", () => {
    const state = Schema.decodeUnknownSync(LocalModelsStateSchema)({
      inventoryState: { _tag: "Ready" },
      models: [eimLocalModel()],
      discoveryState: { _tag: "Ready", progress: [] },
    })

    expect(installedLocalModels(state)).toHaveLength(1)
  })
})

describe("catalog badges for models the fork cannot serve", () => {
  const disabledEntry = (reason: string, overrides: Record<string, unknown> = {}) => ({
    providerId: "local",
    providerModelId: "eim-gemma-7b-ctx8192",
    displayName: "Gemma 7B",
    supportedSlots: ["primary"],
    contextWindow: 8_192,
    maxOutputTokens: 4_096,
    capabilities: {
      vision: false,
      tools: false,
      structuredOutput: false,
      reasoning: { supported: false, efforts: [] },
    },
    availability: { _tag: "Disabled", reason },
    ...overrides,
  })

  // One row per way an EIM model can be unusable. All four reasons already exist and
  // already render a label in cli/src/features/model-menus/container.tsx.
  it.each([
    ["incompatible_runtime", "base model with no chat or tool template"],
    ["invalid_configuration", "gated repository with no HF_TOKEN configured"],
    ["insufficient_resources", "Llama-4-Scout: 218 GB of bf16 weights on a 230 GB budget"],
    ["installation_unavailable", "image built but weight cache is empty"],
  ])("represents %s (%s) without a schema change", (reason) => {
    const entry = Schema.decodeUnknownSync(ProviderModelCatalogEntrySchema)(disabledEntry(reason))

    expect(entry.availability).toEqual({ _tag: "Disabled", reason })
  })

  it("keeps memory accounting absent for a model that was never assessed", () => {
    const entry = Schema.decodeUnknownSync(ProviderModelCatalogEntrySchema)(
      disabledEntry("incompatible_runtime"),
    )

    expect(entry.memory).toStrictEqual(Option.none())
  })

  it("exposes memory accounting for an assessed, serveable model", () => {
    const entry = Schema.decodeUnknownSync(ProviderModelCatalogEntrySchema)(disabledEntry("model_unavailable", {
      availability: { _tag: "Available" },
      providerModelId: `eim-${IMAGE_DIGEST.slice(0, 16)}-ctx32768`,
      displayName: "Qwen3 30B-A3B",
      contextWindow: QWEN3_30B_A3B.contextLength,
      maxOutputTokens: 8_192,
      memory: [systemMemoryAssessment(QWEN3_30B_A3B.requiredBytes)],
      capabilities: capabilitiesInput(),
    }))

    expect(Option.isSome(entry.memory)).toBe(true)
  })
})
