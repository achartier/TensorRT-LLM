# FP8 LoRA Weight Serialization Issue

> **Status (2026-07-22): resolved for the PyTorch backend.** Upstream PR #12959 (`3db10d9179`) implemented the Option 6 design below by removing LoRA weight tensors from request broadcast and loading adapters per rank from the shared path. Replacing the communicator pickle format is no longer required for FP8 LoRA. The remaining end-to-end blocker is PEFT cache dtype handling, now prototyped on `fp8-lora-grouped-gemm` as a homogeneous first-adapter-selected cache.

## The Problem

When LoRA adapter weights are stored in FP8 format (`torch.float8_e4m3fn`), they cannot
be kept in fp8 through the TRT-LLM inference pipeline. The weights must be upcast to
bf16/fp16 during loading, which means the native fp8 `lora_grouped_gemm` kernel (CUTLASS
3.x on Hopper) never receives fp8 weight operands — it only benefits from fp8 activations
when the base model runs in fp8 mode.

## Root Cause

The failure occurs in `tensorrt_llm/_torch/distributed/communicator.py:safe_broadcast()`,
which uses Python's `pickle.dumps`/`pickle.loads` to serialize LoRA weight data for
broadcast across MPI ranks (even in single-GPU mode, the LLM API spawns worker processes
that communicate via this path).

**PyTorch bug**: `pickle.loads` on fp8 tensors fails in PyTorch 2.11.0a0:

```python
import torch, pickle

t = torch.randn(4, 4, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
pickle.loads(pickle.dumps(t))  # AttributeError!
```

The error is:
```
File "torch/storage.py", line 535, in _load_from_bytes
    return torch.load(io.BytesIO(b), weights_only=False)
File "torch/serialization.py", line 1785, in persistent_load
    dtype = storage_type.dtype
AttributeError: type object 'torch.storage.UntypedStorage' has no attribute 'dtype'
```

The issue is in PyTorch's `__reduce_ex__` implementation for tensors. When pickle
serializes a CUDA (or CPU) tensor, it calls `__reduce_ex__` which produces a pickle
payload that goes through `_load_from_bytes` → `torch.load` → `_legacy_load`. The legacy
load path calls `storage_type.dtype` on `UntypedStorage`, which doesn't have a `dtype`
attribute. This works for standard dtypes (float16, bfloat16, float32) because they use
typed storage classes (`HalfStorage`, `BFloat16Storage`, etc.) that do have `.dtype`.
FP8 only uses `UntypedStorage` and has no typed storage class.

Note: `torch.save`/`torch.load` with a file-like object works fine for fp8 — the issue
is specifically in the pickle `__reduce_ex__` → `_load_from_bytes` → `_legacy_load` path.

## Where it manifests

### Data flow (traced with instrumentation on H100)

```
lora_manager.load_from_hf()
  → keeps weights in fp8 (t_in, t_out as float8_e4m3fn tensors)
  → stores in _lora_weights list and cpp_lora_weights dict

base_worker._enqueue_request()
  → tllm.LoraConfig(weights=self._lora_manager.cpp_lora_weights[uid])
    ← THIS embeds actual fp8 weight tensors in the C++ LoraConfig object
  → tllm.Request(lora_config=lora_config)
    ← The C++ Request object now holds fp8 tensors

request_utils.RequestBroadcaster._broadcast_requests()
  → payloads = (new_requests, py_request_objects)
    ← new_requests contains RequestQueueItem with the Request object (6.4MB)
  → dist.broadcast(payloads)
    → communicator.safe_broadcast(comm, obj)

safe_broadcast (communicator.py)
  → pickle.dumps(obj)     ← succeeds (6.4MB serialized, contains 'float8' in bytes)
  → pickle.loads(serialized)  ← FAILS
    → torch.storage._load_from_bytes
      → torch.load(io.BytesIO(b), weights_only=False)
        → _legacy_load
          → persistent_load
            → storage_type.dtype  ← AttributeError on UntypedStorage
```

### Key finding

The fp8 tensors are NOT directly visible in the Python object graph — the
recursive scan of the broadcast object's `__dict__` attributes finds zero fp8
tensors. They are hidden inside the C++ `tllm.Request` nanobind/pybind object,
which pickle serializes via its C++ `__reduce_ex__` implementation. The fp8
tensor data only appears in the serialized bytes.

`pickle.dumps` succeeds because the C++ serialization writes the fp8 tensor
bytes directly. `pickle.loads` fails because PyTorch's `_load_from_bytes` path
uses `_legacy_load` which calls `storage_type.dtype` on `UntypedStorage`.

This is triggered when the executor broadcasts LoRA adapter data to worker
processes during `_fetch_and_activate_new_requests` in the PyExecutor event loop.
Even in single-GPU mode, the root rank serializes and deserializes its own data
through this pickle round-trip.

## Current Workaround

In `lora_manager.py`, fp8 weights are cast to the model's dtype (bf16/fp16) immediately
after loading from the safetensors file, before the `alpha/rank` scale multiply:

```python
t_in = t_in.cuda().contiguous()
t_out = t_out.cuda().contiguous()
# ...
model_dtype = str_dtype_to_torch(model_config.dtype)
t_in = t_in.to(model_dtype)
t_out = t_out.to(model_dtype)
t_out = t_out * scale
```

This means fp8 LoRA weight files can be loaded, but the weights are converted to bf16
before entering the pipeline. The fp8 `lora_grouped_gemm` kernel still provides value
when the base model's **activations** are in fp8.

## Implementation Plan: Remove weights from broadcast (Option 6)

### Summary

The `add_request_peft` fallback path in `resource_manager.py:2903` already handles
`lora_weights=None` by loading from disk via `py_lora_path` — but it's unreachable
on non-root ranks because `py_lora_path` isn't broadcast. The fix is 3 small changes.

### Changes

**1. `tensorrt_llm/executor/base_worker.py` (lines 392-400)**

For PyTorch backend only, always set `weights=None, config=None` in `LoraConfig`.
TRT backend is untouched — gated on `self._is_pytorch_backend`.

```python
if self._is_pytorch_backend:
    lora_config = tllm.LoraConfig(
        task_id=request.lora_request.adapter_id,
        weights=None, config=None)
else:
    # existing TRT path unchanged
```

Keep `_load_lora_adapter` call on rank 0 to pre-populate the local cache.

**2. `tensorrt_llm/_torch/pyexecutor/request_utils.py` (lines 489-514)**

Add `py_lora_path` to `_collect_py_objects()` so it's broadcast to all ranks:

```python
py_lora_path = collect_py_objects_from_requests(new_requests, "py_lora_path")
```

**3. `tensorrt_llm/_torch/pyexecutor/resource_manager.py` (lines 2903-2910)**

Set `request.lora_config` from manager (currently only sets `lora_weights`):

```python
request.lora_weights = self._lora_manager.cpp_lora_weights[uid]
if request.lora_config is None:
    request.lora_config = self._lora_manager.cpp_lora_config[uid]
```

### Edge cases

- **First load**: Each rank independently calls `load_from_ckpt` from shared FS
  (parallel reads, faster than serializing + broadcasting 6+ MB)
- **Cache hit**: `is_task_cached` path unchanged — removes tensors from request
- **PP > 1**: `py_lora_path` propagates via `send_object`/`recv_object` in the
  PP chain. Each PP stage has its own `PeftCacheManager` with correct `pp_rank`
- **TP > 1**: All TP ranks receive `py_lora_path` via broadcast. Each rank's
  `LoraManager` has the correct `tp_rank` mapping for weight sharding
- **Adapter eviction**: `is_task_cached=False` triggers reload from `py_lora_path`
- **TRT backend**: Unchanged — still embeds weights in `LoraConfig`
- **Ray (no MPI)**: `safe_broadcast` is a no-op, single rank. Transparent

### Tests

Existing tests should pass unchanged:
- Single-GPU LoRA: `test_llm_pytorch.py` (all LoRA tests)
- TP LoRA: `test_llm_multi_gpu_pytorch.py` (tp=2, tp=4 tests)
- PP LoRA: `test_nemotron_h_lora_sanity.py` (pp=2 test)
- Resource manager: `test_resource_manager.py::test_add_request_peft`

## Future Solutions

### Option 1: Fix the pickle path in `communicator.py`

Replace `pickle.dumps`/`pickle.loads` with `torch.save`/`torch.load` for tensor-
containing objects:

```python
# Instead of:
serialized = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
# ...
result = pickle.loads(serialized)

# Use:
buf = io.BytesIO()
torch.save(obj, buf)
serialized = buf.getvalue()
# ...
result = torch.load(io.BytesIO(serialized), weights_only=False)
```

**Pros**: Simple, fixes the root cause for all tensor dtypes.
**Cons**: May have performance or compatibility differences vs pickle. Need to verify
`torch.save` handles all object types that `safe_broadcast` might carry.

### Option 2: Custom pickle reducer for fp8 tensors

Register a custom `__reduce_ex__` or `copyreg` handler for fp8 tensors that avoids the
legacy load path:

```python
import copyreg
def _reduce_fp8_tensor(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return (torch.load, (io.BytesIO(buf.getvalue()),), {'weights_only': False})

copyreg.pickle(torch.Tensor, _reduce_fp8_tensor)  # Only for fp8
```

**Pros**: Targeted fix, doesn't change the general pickle path.
**Cons**: Fragile, version-dependent, may interact poorly with PyTorch internals.

### Option 3: Serialize weights separately from the broadcast object

Keep fp8 weights as raw byte buffers in the broadcast payload and reconstruct
them on the receiving side:

```python
# Sender: convert tensor to bytes + metadata
raw = t.cpu().numpy().tobytes()
metadata = {'dtype': t.dtype, 'shape': t.shape, 'device': t.device}

# Receiver: reconstruct
t = torch.frombuffer(raw, dtype=metadata['dtype']).reshape(metadata['shape']).to(metadata['device'])
```

**Pros**: Completely avoids pickle for tensor data.
**Cons**: Requires refactoring the broadcast protocol. `torch.frombuffer` may not support
fp8 directly.

### Option 4: Wait for PyTorch fix

This is a PyTorch bug in the `_legacy_load` path. It may be fixed in a future
PyTorch release (the non-legacy `torch.save`/`torch.load` already works). Track
and upgrade when available.

**Pros**: No TRT-LLM code changes needed.
**Cons**: Blocked on PyTorch release timeline.

### Option 5: Bypass the broadcast for LoRA weights

LoRA weights are loaded from files that are accessible to all ranks (shared filesystem).
Instead of broadcasting weight tensors, broadcast only the file path and have each rank
load independently:

```python
# Instead of: broadcast(weights_dict)
# Do: broadcast(lora_dir_path) → each rank loads from file
```

**Pros**: Avoids serializing tensors entirely. May also reduce peak memory.
**Cons**: Requires rearchitecting the LoRA loading flow. Assumes shared filesystem.

### Recommendation

**Option 1** (fix `communicator.py` to use `torch.save`/`torch.load`) is the simplest
and most correct fix. It should be done as a separate PR since it affects the
distributed communication layer, not just LoRA. Once landed, the `lora_manager.py`
upcast can be removed for Hopper+ GPUs, allowing true fp8 weights end-to-end.
