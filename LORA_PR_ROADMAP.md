# LoRA Adapter Support — PR Roadmap

Findings from LoRA validation testing across dense, MoE, and hybrid models
with bf16 and fp8 adapter weights. Each section is a self-contained PR.

**Jira**: TRTLLM-15314 tracks the complete LoRA investigation and follow-up work in this roadmap.

## Current Status (2026-08-27)

### Landed upstream

- Qwen3 dense and MoE LoRA propagation fixes landed in #12785 (`ae86b91e12`).
- FP8 adapter files load through the BF16/FP16 upcast path from #12848 (`de1212718f`).
- LoRA weights were removed from PyTorch request broadcast in #12959 (`3db10d9179`), so the pickle replacement proposed in PR 5 is no longer needed for this path.
- Dense FP8 LoRA support landed in #16810 (`d247d1454a`). The merged scope keeps adapter weights and activations in E4M3 through the homogeneous PEFT cache and native grouped GEMM, supports eager and CUDA-graph execution, and casts the LoRA delta back to the model activation dtype before accumulation.
- B200 native FP8 LoRA support landed in #17521 (`aaaf65998b`). The merged scope adds dedicated SM100 eager and CUDA-graph grouped-GEMM dispatch, runtime kernel-capability detection, B200 regression coverage, and architecture support documentation while preserving compute-dtype fallback when native kernels are unavailable.
- H100 L0 coverage landed in #18114 (`95af10a8f1`). It adds `tests/unittest/others/test_lora_manager.py` to `l0_h100.yml` so shared LoRA loading, FP8 conversion, and native SM90 capability paths run in H100 CI.
- Exact PEFT-cache byte-budget accounting landed in #18200 (`ad84866c81`). It recalculates host and device page capacity from the first adapter's homogeneous dtype, preserves explicit logical capacity, and fixes percentage-based device sizing to reuse one derived byte budget.

### Post-merge state

- Production native FP8 adapter support now covers dense LoRA on SM90 and SM100. SM103/SM107 and SM120/SM121 remain disabled and use the compute-dtype fallback path.
- Routed-expert MoE FP8 LoRA remains unsupported. FP8 expert adapters are rejected instead of being passed to the compute-dtype fused-MoE path.
- The final PR CI twice timed out in `TestNemotron3Super120B::test_auto_dtype[mtp_nextn=0-block_reuse=False-use_py_transceiver=False]` on different B200 nodes. This exact disaggregated-serving test was previously tracked by NVBug 6465993 and fixed by #16505 (`cf44a1ccee`). The test disables LoRA and runs outside the SM90-only FP8 path, so no code-path relationship to #16810 was found. The PR subsequently merged.
- The final #17521 pre-merge pipeline (`L0_MergeRequest_PR` #55954) passed on reviewed tip `a4483f2`; the PR then merged as `aaaf65998b`.
- #18114 was validated from a clean native-SM90 Release build and wheel installation on an H100 PCIe GPU. The complete LoRA manager module passed with 18 tests and 3 subtests.
- SM120/SM121 support, routed-expert MoE support, and broader performance characterization remain follow-up work.

### Next steps

- Treat SM120/SM121 as a separate kernel-enablement project: validate the required CUTLASS collective and launch constraints, add architecture-specific instantiations if needed, and keep runtime capability detection authoritative.
- Run full-model routed-expert MoE FP8 LoRA E2E validation before considering the option-3 prototype for production. Include numerical comparison, CUDA-graph replay, workspace cost, and throughput/latency measurements.
- Characterize dense native FP8 LoRA performance on both SM90 and SM100 against the compute-dtype fallback, including eager and CUDA-graph execution across representative ranks and batch shapes.

### Branch roles

- `lora-analysis` is the umbrella investigation branch. It retains the validation tests, historical root-cause notes, and alternative implementation work for reference.
- `test-lora-manager-h100-ci` was the review branch for merged PR #18114. Upstream contains the squashed implementation at `95af10a8f1`.
- `fp8-lora-peft-cache-capacity` was the review branch for merged PR #18200. Upstream contains the squashed implementation at `ad84866c81`.
- `fp8-lora-b200` was the review branch for merged PR #17521. Its reviewed tip was `a4483f2`; upstream contains the squashed implementation at `aaaf65998b`, and the local worktree/branch were removed after merge.
- `fp8-lora-grouped-gemm` is the broad implementation branch. It retains the native Hopper FP8 grouped GEMM, regression fixes, PEFT cache design notes, the homogeneous FP8 cache prototype, and the activation conversion required by block-scaled FP8 models.
- `fp8-lora-dense-minimal` is the historical review branch for merged PR #16810. Its final reviewed tip is `f8cbd4128f`; upstream contains the squashed implementation at `d247d1454a`.
- `fp8-lora-moe-native` is the option-3 experiment: native FP8 routed-expert LoRA using FP8 activation/result scratch around the existing single-dtype grouped GEMMs. Its signed and H100-validated tip is `b1c01ed285`.

### Homogeneous FP8 PEFT cache

The dense implementation now lets the first supplied adapter select one cache dtype. Host and device page managers are reinitialized while empty, later adapters must use the same dtype, and source/page plus host/device dtype mismatches are rejected. The selected dtype is exposed through nanobind and propagated to LoRA execution. Because the grouped GEMM is single-dtype, `LoraLayer` saturates FP16, BF16, or FP32 activations to the E4M3 finite range and casts them to E4M3 when the cache is FP8. It casts the LoRA result back to the original activation dtype before the caller accumulates it. Other activation/cache dtype mismatches remain errors. This adds no public configuration surface.

#18200 keeps the provisional model-dtype allocation so cache-capacity APIs remain available before the first adapter arrives. When the first adapter selects the homogeneous dtype, the manager recalculates both page configurations from the normalized cache settings. Direct host byte limits and the fixed byte budget derived from `deviceCachePercent` therefore gain the expected FP8 capacity, while explicit `numHostModuleLayer` and `numDeviceModuleLayer` logical capacities remain unchanged. Matching dtype/page configurations retain the original allocation; differing empty page managers are replaced together.

### H100 validation results

Validation ran on an H100 PCIe with an SM90-only clean build. The build exposed and fixed stale `nvinfer1::DataType::kFP8` comparisons in LoRA and grouped-GEMM sources; all affected parameters use `tensorrt_llm::DataType`. The initial clean build and editable wheel installation completed successfully; both extracted branches subsequently completed incremental Release rebuilds against the same SM90 tree.

- Passed for #18114: all 18 tests and 3 subtests in `tests/unittest/others/test_lora_manager.py` after a clean native-SM90 Release build and wheel installation.
- Passed: `PeftCacheManagerTest.adapterSelectsHomogeneousCacheDataType` after generating the standard ignored TP2 LoRA fixtures.
- Passed: all 16 tests in `test_fp8_lora_grouped_gemm_regressions.py`, including eager and CUDA-graph-mode activation conversion checks plus FP8-delta/BF16-output accumulation. A temporary test-only workaround was required because current `origin/main` has an unrelated import-time mismatch: `MTPEagleDynamicTreeWorker.forward` conflicts with the new `SpecWorkerBase.__init_subclass__` invariant.
- Passed on `fp8-lora-dense-minimal`: `TestQwen3LoRA.test_qwen3_fp8_lora` against `Qwen3-0.6B-FP8` with generated nonzero rank-16 FP8 adapters. The run covered model loading, cache construction, context execution, and decode, and verified that LoRA changes the output (`1 passed` in 97.95 seconds).
- Passed control on the same minimal branch: `TestQwen3LoRA.test_qwen3_bf16_lora` against the same checkpoint (`1 passed` in 11.97 seconds with session reuse).
- Passed on `fp8-lora-moe-native`: all 14 extraction and CUDA-graph parameter tests, all 22 fused-MoE LoRA op tests, and all 6 grouped-GEMM eager/CUDA-graph tests. New coverage verifies FP8 adapter deltas in both per-request and slot-indexed schemas and bit-exact FP8 CUDA-graph replay. The Release build and editable wheel install completed. A full MoE-model E2E run remains outstanding because the staged model set only contained dense Qwen3-0.6B.

Qwen3 dynamic 128x128 block-scaled FP8 linear accepts BF16 activations and quantizes them internally for the base GEMM. `Linear.apply_linear` sends the original BF16 activation to LoRA, so limiting support to base-weight dtype equal to LoRA dtype was insufficient. The activation conversion at the `LoraLayer` boundary now satisfies the single-dtype kernel contract without changing the base linear path.

The merged production implementation keeps native-FP8 support dense-only and deliberately narrow: one homogeneous cache dtype per cache lifetime, E4M3 adapters on SM90, LoRA rank and hidden/output dimensions divisible by 16, and no mixed activation/weight kernel.

The option-3 experiment resolves the routed-expert dtype/stride boundary without changing base-MoE precision. Python propagates the PEFT cache dtype into the fused op; C++ allocates capture-stable FP8 input/result scratch, converts BF16/FP16 activations into E4M3 for the existing grouped GEMMs, and restores their output before activation and finalization. Eager and CUDA-graph op-level tests pass. This remains separate from the minimal dense patch until full MoE-model E2E and performance data justify the added workspace and conversion cost.

Full MoE-model E2E, broader numerical comparison, and dense/MoE performance measurement remain follow-up work.

The PR sections below preserve the original decomposition; this status block supersedes their older dependency and blocker statements.

---

## PR 1: Fix Qwen3 dense LoRA (GatedMLP missing layer_idx)

**Priority**: High — blocks all LoRA usage on Qwen3 dense models
**Complexity**: Trivial (1 line)

**Problem**: `Qwen3DecoderLayer.__init__` creates `GatedMLP` without passing
`layer_idx`, causing `AssertionError: layer_idx is required for lora` during
CUDA graph capture.

**Fix**: `tensorrt_llm/_torch/models/modeling_qwen3.py` — add `layer_idx=layer_idx`
to the `GatedMLP` constructor call.

**Test**: `test_lora_fixes.py::TestQwen3DenseLoRA::test_qwen3_dense_bf16_lora`
— generates dummy bf16 LoRA weights for Qwen3-0.6B and verifies outputs differ
with vs without LoRA. Fails without fix (AssertionError), passes with fix.

**Scope**: Affects all Qwen3 dense models (0.6B, 1.7B, 4B, 8B, 14B, 32B).
Likely also affects other models using `GatedMLP` without `layer_idx` if LoRA
targets MLP modules.

---

## PR 2: Fix Qwen3 MoE LoRA (missing kwargs propagation)

**Priority**: High — LoRA silently does nothing on Qwen3 MoE models
**Complexity**: Trivial (1 line)

**Problem**: `Qwen3MoEModel.forward()` does not pass `**kwargs` to decoder layer
calls, so `lora_params` never reaches the attention modules. LoRA loads without
error but produces identical output to the base model.

**Fix**: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py` — add `**kwargs` to
the decoder layer call in `Qwen3MoEModel.forward()`.

**Test**: `test_lora_fixes.py::TestQwen3MoELoRA::test_qwen3_moe_bf16_lora`
— generates dummy bf16 attention LoRA for Qwen3-30B-A3B and verifies output
changes. Fails without fix (identical output), passes with fix.

**Note**: The dense Qwen3 model (`modeling_qwen3.py`) already correctly passes
`**kwargs` at the equivalent location. This is a copy-paste omission in the MoE
variant.

**Scope**: Affects all Qwen3 MoE models (30B-A3B, 235B-A22B).

---

## PR 3: Support loading FP8 LoRA weight files

**Priority**: Medium — enables fp8 adapter files to be used
**Complexity**: Small (5 lines)

**Problem**: When LoRA adapter weights are stored in `torch.float8_e4m3fn`
format, `lora_manager.py` fails at `t_out = t_out * scale` with
`NotImplementedError: "mul_cuda" not implemented for 'Float8_e4m3fn'`.

**Fix**: `tensorrt_llm/lora_manager.py` — cast weights to model dtype (bf16/fp16)
before the scale multiply. Weights are upcast during loading; the fp8 GEMM
kernel benefits from fp8 **activations** when the base model runs in fp8 mode.

**Test**: `test_lora_fixes.py::TestQwen3DenseLoRA::test_qwen3_dense_fp8_lora`
and `TestQwen3MoELoRA::test_qwen3_moe_fp8_lora`. Both fail without fix
(NotImplementedError), pass with fix.

**Historical note**: The original investigation was also blocked by FP8 tensor
pickle serialization in the request communicator. Upstream #12959 removed LoRA
weights from that broadcast path. The remaining blocker is homogeneous FP8 PEFT
cache support.

**Scope**: All models, all GPU architectures.

---

## PR 4: Native FP8 lora_grouped_gemm kernel (CUTLASS 3.x)

**Priority**: Medium — performance optimization for Hopper+ with fp8 models
**Complexity**: Large (800+ lines C++)
**Depends on**: PR 3

**Problem**: The `lora_grouped_gemm` C++ kernel only supports fp16/bf16. When
the base model runs in fp8, LoRA activations must be upcast to bf16 for the
GEMM, losing the fp8 tensor core throughput advantage.

**Fix**: Add fp8 dispatch to all three grouped GEMM variants using the CUTLASS
3.x collective API (`KernelPtrArrayTmaWarpSpecializedCooperativeFP8FastAccum`):
- `cpp/tensorrt_llm/kernels/groupGemm.cu` — eager path
- `cpp/tensorrt_llm/kernels/splitkGroupGemm.cu` — split-K path
- `cpp/tensorrt_llm/kernels/cuda_graph_grouped_gemm.cu` — CUDA graph path
- `cpp/tensorrt_llm/kernels/lora/lora.cpp` — force grouped path for fp8
- `cpp/tensorrt_llm/thop/loraOp.cpp` — add `kFP8` dtype switch

All guarded by `#ifdef ENABLE_FP8` and requires sm90+ (Hopper).

**Test**: `test_lora_plugin_vs_lora_op.py::TestLoraGroupedGemmFp8` — compares
fp8 kernel output against bf16 kernel on the same (fp8-round-tripped) data.
Passes on H100 with atol=0.05.

**Key constraints**:
- LoRA rank, hidden size, and output size must be multiples of 16 for the selected TMA kernel
- LoRA rank >= 16 is required
- Uses tile shape `<16, 32, 64>` for both isLoraIn and !isLoraIn paths
- CUTLASS 4.2.1 (in `3rdparty/cutlass/`) has full support

**Scope**: Hopper+ GPUs only. No effect on pre-Hopper (guarded).

---

## PR 5: Fix FP8 tensor pickle serialization in communicator

**Status**: Superseded by upstream #12959 (`3db10d9179`), which removed LoRA tensors from the PyTorch request broadcast. Do not implement the pickle replacement for this use case.

**Priority**: None — resolved upstream

**Problem**: `communicator.py:safe_broadcast()` uses `pickle.dumps`/`pickle.loads`
to serialize LoRA data between executor and worker processes. PyTorch 2.11's
pickle `__reduce_ex__` for fp8 tensors goes through `_legacy_load` which hits
`UntypedStorage.dtype` — an attribute that doesn't exist because fp8 has no
typed storage class.

**Historical proposed fix**: Replace `pickle.dumps`/`pickle.loads` with `torch.save`/`torch.load`
in `safe_broadcast()` (the non-legacy torch serialization path handles fp8).

**Test**: Unit test that pickles an fp8 CUDA tensor through `safe_broadcast`
and verifies round-trip correctness.

**Note**: This is a broader fix that would benefit any future dtype additions,
not just fp8 LoRA.

See `FP8_LORA_WEIGHT_SERIALIZATION.md` for full analysis and alternative approaches.

---

## PR 6: Keep FP8 LoRA weights end-to-end

**Status**: Landed upstream in #16810 (`d247d1454a`) after H100 validation.

**Priority**: Low — performance optimization
**Depends on**: PR 4, upstream #12959, homogeneous PEFT cache support

With request broadcast resolved upstream, `lora_manager.py` keeps fp8 weights in fp8 on SM90 GPUs:
- Bake `alpha/rank` scale into B weights via bf16 round-trip:
  `t_out = (t_out.to(bf16) * scale).to(fp8)`
- Skip the full dtype cast for `t_in`
- Both the GEMM inputs (activations) and weights would be in fp8

This gives the full benefit of the fp8 `lora_grouped_gemm` kernel.

---

## Summary Matrix

| PR | Files | Lines | Risk | Blocks |
|----|-------|-------|------|--------|
| 1. Qwen3 dense layer_idx | 1 Python | +1 | Low | — |
| 2. Qwen3 MoE kwargs | 1 Python | +2/-1 | Low | — |
| 3. FP8 weight loading | 1 Python | +5/-3 | Low | — |
| 4. FP8 GEMM kernel | 7 C++ | +800 | Medium | — |
| 5. Communicator pickle fix | 1 Python | ~20 | Medium | — |
| 6. E2E fp8 weights | 1 Python | ~10 | Low | PR 4, 5 |

PRs 1-4 and PR 6 have landed upstream, as has removal of LoRA weights from
request broadcast. The homogeneous cache and dense Qwen3 FP8 path have C++ and
H100 end-to-end validation. Native MoE is separately proven at op and
CUDA-graph level on `fp8-lora-moe-native`, with full-model E2E, numerical, and
performance coverage remaining. PR 5 is no longer planned.
