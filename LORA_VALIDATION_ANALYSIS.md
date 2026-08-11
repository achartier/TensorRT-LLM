# LoRA Adapter Validation Analysis

**Date**: 2026-04-05
**Environment**: H100 PCIe 80GB (computelab), TRT-LLM 1.3.0rc11, PyTorch 2.11.0a0+eb65b36914.nv26.02

## Test Matrix

| Model | Type | bf16 LoRA | fp8 LoRA |
|-------|------|-----------|----------|
| Qwen3-0.6B | Dense | PASS (after fix) | FAIL |
| Qwen3-30B-A3B | MoE | PASS (after fix) | FAIL |
| Nemotron-H-8B-Reasoning-128K | Hybrid (Mamba+Attn) | PASS | FAIL |

## Bugs Found

### Bug 1: Missing `layer_idx` in Qwen3 dense GatedMLP

**File**: `tensorrt_llm/_torch/models/modeling_qwen3.py:117`

The `GatedMLP` module requires `layer_idx` for LoRA to work, but `Qwen3DecoderLayer.__init__` doesn't pass it:

```python
# Before (broken):
self.mlp = GatedMLP(
    hidden_size=config.hidden_size,
    intermediate_size=config.intermediate_size,
    bias=config.mlp_bias if hasattr(config, "mlp_bias") else False,
    dtype=config.torch_dtype,
    overridden_tp_size=1 if self.enable_attention_dp else None,
    config=model_config,
)

# After (fixed):
self.mlp = GatedMLP(
    ...,
    config=model_config,
    layer_idx=layer_idx,  # <-- added
)
```

**Error**: `AssertionError: layer_idx is required for lora` during CUDA graph capture.

**Impact**: Blocks ALL LoRA usage on Qwen3 dense models (and likely other models using GatedMLP without layer_idx).

### Bug 2: Missing `**kwargs` propagation in Qwen3 MoE model

**File**: `tensorrt_llm/_torch/models/modeling_qwen3_moe.py:393-401`

`Qwen3MoEModel.forward()` does not pass `**kwargs` to decoder layers, so `lora_params` never reaches the attention modules:

```python
# Before (broken):
for decoder_layer in self.layers:
    hidden_states, residual = decoder_layer(
        position_ids=position_ids,
        hidden_states=hidden_states,
        attn_metadata=attn_metadata,
        residual=residual,
        spec_metadata=spec_metadata,
        mrope_config=mrope_config,
        deepstack_embeds=deepstack_embeds)  # No **kwargs!

# After (fixed):
for decoder_layer in self.layers:
    hidden_states, residual = decoder_layer(
        ...,
        deepstack_embeds=deepstack_embeds,
        **kwargs)  # <-- added
```

**Impact**: LoRA silently does nothing on Qwen3 MoE models. No error, just identical output with/without LoRA.

Note: The dense Qwen3 model (`modeling_qwen3.py`) correctly passes `**kwargs` at the equivalent location (line 262).

## FP8 LoRA Support Gap Analysis

FP8 LoRA weights fail at two distinct levels. Both must be addressed.

### Blocker 1: Weight Loading — PyTorch `mul_cuda` not implemented for fp8

**Location**: `tensorrt_llm/lora_manager.py:1157`
**Operation**: `t_out = t_out * scale` where `t_out` is `torch.float8_e4m3fn`
**Error**: `NotImplementedError: "mul_cuda" not implemented for 'Float8_e4m3fn'`

This is a **native PyTorch limitation**. Verified experimentally:

```python
import torch
x = torch.randn(4, 4, dtype=torch.bfloat16).to(torch.float8_e4m3fn).cuda()
x * 2.0  # NotImplementedError: "mul_cuda" not implemented for 'Float8_e4m3fn'
x.to(torch.bfloat16) * 2.0  # OK
```

PyTorch 2.11 does not support element-wise arithmetic on fp8 tensors. The `torch.mul`, `*` operator, and `torch.Tensor.__mul__` all fail.

**Fix**: Cast to model dtype (bf16/fp16) *before* scaling. Lines 1157-1159 currently do:
```python
t_out = t_out * scale            # FAIL: fp8 doesn't support mul
t_in = t_in.to(model_dtype)      # cast after
t_out = t_out.to(model_dtype)    # cast after
```

Should be:
```python
t_in = t_in.to(model_dtype)      # cast first
t_out = t_out.to(model_dtype)    # cast first
t_out = t_out * scale            # now works (bf16/fp16)
```

### Blocker 2: LoRA GEMM Kernel — TRT-LLM extension only supports fp16/bf16

**Location**: `cpp/tensorrt_llm/thop/loraOp.cpp:157-159` (eager path) and `:307-309` (CUDA graph path)

```cpp
case torch::kFloat16: loraRuntimeDataType = nvinfer1::DataType::kHALF; break;
case torch::kBFloat16: loraRuntimeDataType = nvinfer1::DataType::kBF16; break;
default: throw std::invalid_argument("Invalid dtype, only supports float16, bfloat16");
```

The `lora_grouped_gemm` operation (both eager and CUDA graph variants) explicitly rejects any dtype other than fp16/bf16. This means even if we fix the weight loading, the actual LoRA forward computation would fail if weights remain in fp8.

Since blocker 1's fix casts weights to model dtype (bf16/fp16) before use, blocker 2 is automatically resolved — the weights would already be in bf16/fp16 by the time they reach the GEMM kernel.

**Conclusion**: The fix for blocker 1 (cast before scale) is sufficient to make fp8 LoRA *weight files* loadable. The weights get cast to bf16/fp16 during loading, so the GEMM kernel works as-is. True fp8 LoRA computation (keeping weights in fp8 through the GEMM) would require extending the C++ kernel, but that's a separate feature request.

### Comment in GatedMLP about fp8 LoRA

There's also an existing comment/WAR in `gated_mlp.py:134-141`:
```python
if has_lora:
    # NOTE: This is a WAR, since LoRA grouped_gemm does not support FP8 yet.
    # TODO: Remove this path when LoRA grouped_gemm supports FP8
    logger.warning(...)
    return swiglu(x)  # Forces non-FP8 activation
```

This confirms the team is aware the LoRA GEMM doesn't support fp8 and has workarounds in place.

## Proposed Plan for FP8 LoRA Weight Support

### Minimal fix (load fp8 weights, compute in bf16)

1. **`lora_manager.py:1148-1159`**: Move dtype conversion before scale multiplication:
   ```python
   t_in = t_in.cuda().contiguous().to(str_dtype_to_torch(model_config.dtype))
   t_out = t_out.cuda().contiguous().to(str_dtype_to_torch(model_config.dtype))
   # Now scaling works
   t_out = t_out * scale
   ```

2. No C++ kernel changes needed — weights are already in bf16/fp16 by the time they reach the kernel.

3. Add a test covering fp8 weight loading + inference for each model type.

### Future: Native fp8 LoRA computation

Adding fp8 support to `lora_grouped_gemm` would require changes at multiple levels:

#### C++ / CUDA changes

1. **`cpp/tensorrt_llm/thop/loraOp.cpp`** — Add `torch::kFloat8_e4m3fn` case to the dtype
   switch in all three registered ops (eager, CUDA graph, fused param fill):
   ```cpp
   case torch::kFloat8_e4m3fn: loraRuntimeDataType = nvinfer1::DataType::kFP8; break;
   ```

2. **`cpp/tensorrt_llm/kernels/cuda_graph_grouped_gemm.cu`** — The CUTLASS grouped GEMM
   template is instantiated with `cutlassType` for both A and B operands. FP8 would need:
   - New template instantiation with `cutlass::float_e4m3_t` element types
   - Mixed-precision accumulation in fp32 (already done for fp16/bf16)
   - The CUTLASS 3.x grouped GEMM API may differ for fp8 — need to check if
     `cutlass::gemm::device::GemmGrouped` supports fp8 or if the newer
     `cutlass::gemm::collective` API is needed
   - Alignment requirements differ: fp8 has 1-byte elements vs 2-byte for fp16

3. **`cpp/tensorrt_llm/kernels/loraKernel.cu`** (if it exists) or the underlying
   `LoraImpl` class — needs fp8 dtype handling in `run()` and `getWorkspaceSize()`

#### Python changes

4. **`tensorrt_llm/_torch/peft/lora/layer.py`** — The `LoraLayer.__call__` method
   calls `torch.ops.trtllm.lora_grouped_gemm` with the input tensor's dtype. If the
   model input is fp8-quantized, the LoRA weights would also need to be in fp8 or a
   mixed-precision path would be needed.

5. **`tensorrt_llm/_torch/modules/gated_mlp.py:134-141`** — Remove the WAR that forces
   non-FP8 activation when LoRA is active.

6. **`tensorrt_llm/lora_manager.py`** — Keep fp8 weights in fp8 format instead of
   casting to model dtype, and pass appropriate scaling factors.

#### Key challenges

- **Scaling factors**: FP8 has limited dynamic range (e4m3fn: ~[-448, 448]). LoRA weights
  typically have small magnitudes, so they fit in fp8 range. But the `alpha/rank` scaling
  must be handled carefully — either fuse it into the GEMM epilogue or apply it post-GEMM.
- **CUTLASS support**: Verify that CUTLASS grouped GEMM supports fp8. The standard GEMM
  does (sm89+), but grouped GEMM may not be instantiated for fp8 yet.
- **Mixed precision**: The most practical approach is likely mixed-precision: fp8 weights
  with bf16/fp32 accumulation, matching how the base model already handles fp8.

#### Build attempt results (2026-04-05)

We attempted to implement fp8 by adding `cutlass::float_e4m3_t` template instantiations
to `groupGemm.cu`, `splitkGroupGemm.cu`, and `cuda_graph_grouped_gemm.cu` using the
CUTLASS 2.x `DefaultGemmGrouped` API with `cutlass::arch::Sm89` and instruction shape
`<16, 8, 32>`. The build failed with multiple CUTLASS static assertion failures:

1. **Minimum alignment**: fp8 requires `kAlignmentAB >= 4` (4 bytes = 4 fp8 elements).
   CUTLASS's `memory_sm80.h` does not support loads smaller than 4 bytes for fp8.
   Fixed by clamping minimum alignment to 4.

2. **K dimension**: fp8's instruction K=32 means threadblock K must be >= 64 for the
   required >=2 warp-level GEMM iterations. The `!isLoraIn` tile `<32, 128, 32>` fails;
   changing to `<16, 32, 64>` (matching `isLoraIn`) fixes the iteration count.

3. **Epilogue incompatibility** (blocker): CUTLASS 2.x's `DefaultEpilogueTensorOp` with
   `ElementOutput = float_e4m3_t` generates invalid `global_load` specializations.
   The epilogue tile iterator tries to do 64-byte aligned loads that are not supported.
   This is a **fundamental limitation of CUTLASS 2.x's grouped GEMM API** — it does not
   have proper fp8 epilogue support.

**Conclusion**: Native fp8 LoRA GEMM requires migrating to **CUTLASS 3.x's collective
API** (`cutlass::gemm::collective::CollectiveMainloop`), which has first-class fp8 support.
This is a significant refactor affecting `groupGemm.cu`, `splitkGroupGemm.cu`, and
`cuda_graph_grouped_gemm.cu`.

#### Effort estimate

- High complexity: ~1-2 weeks for a CUDA engineer familiar with CUTLASS 3.x
- Requires porting the entire LoRA grouped GEMM from CUTLASS 2.x to 3.x API
- Alternative: use cuBLAS `cublasLtMatmul` with fp8 for a simpler (but possibly
  less performant) approach
- Test marker: `test_lora_plugin_vs_lora_op.py::TestLoraGroupedGemmFp8` currently
  verifies the kernel rejects fp8; update it to verify correctness when support lands
