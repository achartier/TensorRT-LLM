<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native FP8 routed-expert MoE LoRA validation

## Verdict

The prototype is **ready for a production PR**. Eager execution and full-engine
CUDA-graph replay are functionally and numerically validated on Qwen3-30B-A3B for
both BF16 and native E4M3 adapters. Compute Sanitizer reports zero errors after the
fix. The measured FP8 overhead is material and warrants follow-up optimization, but
it is characterized here and is not a correctness blocker.

## Environment and source

- Validation date: 2026-08-28
- GPU: NVIDIA H100 PCIe, 81,559 MiB, driver 595.58.03
- Container: `gitlab-master.nvidia.com:5005/achartier/tekit:latest`
- Model: BF16 Qwen3-30B-A3B, 48 layers, 128 routed experts, top-k 8
- Validated branch: `fp8-lora-moe-native`
- Rebase parent / `origin/main`: `5767bed0251ee3539b260bd316487aed57ecae4a`
- Build: clean native-SM90 Release build followed by an incremental rebuild and
  wheel installation after the descriptor fix

## Adapter construction

The harness generates paired BF16 and E4M3 adapters from the same E4M3-rounded
values. It adapts gate, up, and down projections for every expert in layers 46 and
47 at rank 16 with `alpha=rank`.

- 1,536 tensors
- 34,603,008 total E4M3 values
- 34,064,592 nonzero values (98.4440%)
- BF16 reference values are exact E4M3 dequantizations, isolating compute/cache
  precision from adapter-generation differences

## Eager end-to-end result

Both engines loaded all 1,016 base-model tensors. PEFT insertion cached task 0 with
six nonzero expert module ranks (three projections across two layers). The BF16 cache
reported `torch.bfloat16`; the native cache reported `torch.float8_e4m3fn`.

The adapter changes output:

| Measurement | BF16 adapter | Native FP8 adapter |
| --- | ---: | ---: |
| First-token logit delta, max abs | 2.109375 | 2.093750 |
| First-token logit delta, mean abs | 0.334096 | 0.326491 |
| First generated token | 3555 | 358 |
| 16-token decode differs from base | yes | yes |
| Repeated decode deterministic | yes | yes |

The FP8 and BF16 logit deltas pass `atol=0.25, rtol=0.10`:

- allclose: true
- maximum absolute delta error: 0.15625
- mean absolute delta error: 0.031414
- cosine similarity: 0.995705

## Workspace and performance

The conversion scratch consists of one E4M3 input and one E4M3 result buffer per
active MoE invocation:

| Shape | Input | Result | Total |
| --- | ---: | ---: | ---: |
| Decode batch 1 (8 expanded rows) | 16 KiB | 16 KiB | 32 KiB |
| Decode batch 4 (32 expanded rows) | 64 KiB | 64 KiB | 128 KiB |
| Configured 256 tokens (2,048 expanded rows) | 4 MiB | 4 MiB | 8 MiB |

The first FP8 adapter request added no persistent `torch.cuda.memory_allocated()`
bytes after engine/cache construction; the BF16 reference added 32 KiB.

Batch-4, 16-token CUDA-graph measurements use three interleaved base/LoRA iterations:

| Cache dtype | Base latency | LoRA latency | Latency overhead | Base tok/s | LoRA tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| BF16 | 0.1481 s | 0.1940 s | 31.02% | 432.26 | 329.92 |
| FP8 | 0.1487 s | 0.2136 s | 43.60% | 430.32 | 299.66 |

These are basic single-GPU measurements, not a production performance study. The
FP8 conversion overhead is material at this small batch/sequence shape and needs
broader rank, batch, prefill, and decode characterization as follow-up work.

## Defects found during full-model validation

1. Expert-only adapters did not select the PEFT cache dtype. The loader now infers a
   homogeneous dtype across all LoRA modules, including routed experts.
2. Native FP8 was explicitly disabled for MoE weights in the loader. The implementation
   now retains supported E4M3 expert weights.
3. `Qwen3MoEDecoderLayer` dropped `lora_params` before `Qwen3MoE` and its expert
   backend. This made full-model expert LoRA a silent no-op; propagation is now
   explicit and eager output changes.
4. MoE allocated one pinned maximum-problem descriptor, while the grouped-GEMM FP8
   validator and CUTLASS interface consume one per problem. Multi-token execution
   read beyond that allocation. The scratch now owns and initializes the full
   capture-stable descriptor array.
5. CUDA-graph warmup reserved only `max_batch_size * max_tokens_per_seq` rows. A
   later eager prefill could grow the shared runner after capture. Reservation now
   covers the engine-wide `max_num_tokens` capacity.
6. CUDA-graph slot metadata used one adapter-wide rank for every layer and module.
   A partial adapter therefore paired rank 16 with null weight pointers in unadapted
   layers. Expert pointer expansion added the expert byte offset to null, producing
   invalid addresses such as `0x2d0000`. Slot ranks are now stored per layer/module,
   so an absent module has rank 0 and its grouped GEMMs are no-ops.

## CUDA-graph validation and root cause

Before the per-module rank fix, BF16 routed-expert LoRA succeeded in eager prefill
and failed on the first full-engine LoRA graph replay with
`cudaErrorIllegalAddress`. Compute Sanitizer identified 32 invalid 16-byte reads in
the CUTLASS grouped GEMM. The failing addresses were expert-sized offsets from null,
which led directly to the global-rank/partial-adapter mismatch above.

After the fix, a two-layer adapter completes the full 16-token BF16 and FP8 paths
with 16 CUDA-graph replays each. Both adapters change decoded output relative to the
base model. The paired numerical comparison passes `atol=0.25, rtol=0.10` with
maximum absolute error 0.15625, mean absolute error 0.031414, and cosine similarity
0.995705.

The same full-model BF16 and FP8 graph sequence under Compute Sanitizer completes
with `ERROR SUMMARY: 0 errors`. Focused graph coverage also includes multiple calls
sharing one cached runner, FP8 replay, multiple adapters, slot reassignment, growing
captured batch shapes after worst-case reservation, and capture with an empty slot
followed by activation on replay.

## Tests executed

- Focused matrix (`test_moe_lora_op.py`, `test_moe_lora_grouped_gemm.py`,
  `test_moe_lora_cuda_graph_params.py`, and `test_lora_manager.py`): 58 passed,
  3 subtests passed
- Eager full-model harness: passed and emitted `report-eager.json`
- CUDA-graph full-model harness: passed and emitted `report-cudagraph.json`
- Full-model post-fix Compute Sanitizer: `ERROR SUMMARY: 0 errors`

The validation harness is
`tests/integration/defs/llmapi/fp8_moe_lora_validation.py` and intentionally is not
named `test_*.py` pending a production CI plan.
