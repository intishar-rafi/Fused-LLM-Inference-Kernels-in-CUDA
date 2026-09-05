# Fused LLM Inference Kernels in CUDA

A from-scratch CUDA kernel library implementing the core compute path of a modern transformer's feed-forward block — from warp-level primitives up through a fused, end-to-end forward pass. Every kernel was implemented and unit-tested individually, then composed into one working pipeline.

## What this is

The project builds up, in order, to `run_transformer_ffn` — a single function that reproduces the pre-norm feed-forward sublayer used in models like LLaMA, Mistral, and Qwen:

```
residual_out = x + residual
out          = residual_out + SwiGLU_MLP(RMSNorm(residual_out))
```

This is the same computational shape that runs inside every transformer layer, in every forward pass, in production LLM inference engines.

## Architecture

```mermaid
flowchart TD
    subgraph Foundation["Warp & Block Reductions"]
        A1["warp_reduce_sum"]
        A2["warp_reduce_max"]
        A3["block_reduce_sum"]
        A4["block_reduce_max"]
    end

    subgraph Elementwise["Residual & Activations"]
        B1["add_residual_kernel"]
        B2["gelu_kernel"]
        B3["silu_kernel"]
        B4["swiglu_kernel"]
    end

    subgraph Norm["Normalization"]
        C1["rmsnorm_kernel"]
        C2["layernorm_kernel"]
        C3["fused_add_rmsnorm_kernel"]
    end

    subgraph Softmax["Softmax"]
        D1["softmax_row_kernel"]
        D2["causal_softmax_kernel"]
    end

    subgraph PosEmb["Embeddings & RoPE"]
        E1["embedding_lookup_kernel"]
        E2["rope_kernel"]
    end

    subgraph Linear["Linear Layers"]
        F1["linear_kernel"]
        F2["fused_linear_bias_gelu_kernel"]
        F3["mlp_swiglu_forward"]
    end

    subgraph Composite["Composite Blocks"]
        G1["rmsnorm_residual_block"]
        G2["run_transformer_ffn"]
    end

    A1 --> A3
    A2 --> A4
    A3 --> C1
    A3 --> C2
    A4 --> D1
    A3 --> D1
    A3 --> C3
    B4 --> F3
    F1 --> F3
    F1 --> F2
    C3 --> G1
    B1 --> G2
    G1 --> G2
    F3 --> G2

    style G2 fill:#2d5,stroke:#333,stroke-width:2px

## Kernel breakdown

| Part | Kernels | What it does |
|---|---|---|
| 1. Warp & Block Reductions | `warp_reduce_sum`, `warp_reduce_max`, `block_reduce_sum`, `block_reduce_max` | Combine values across GPU threads using shuffle intrinsics and shared memory — the foundational primitive every later kernel builds on |
| 2. Residual & Activations | `add_residual_kernel`, `gelu_kernel`, `silu_kernel`, `swiglu_kernel` | Elementwise skip connections and the nonlinearities used in transformer MLPs |
| 3. Normalization | `rmsnorm_kernel`, `layernorm_kernel`, `fused_add_rmsnorm_kernel` | Row-wise normalization, including a fused residual-add + RMSNorm kernel to avoid an extra memory round-trip |
| 4. Softmax | `softmax_row_kernel`, `causal_softmax_kernel` | Numerically stable softmax (max-subtraction trick), including causal masking for autoregressive attention |
| 5. Embeddings & RoPE | `embedding_lookup_kernel`, `rope_kernel` | Token embedding gather and rotary positional embeddings |
| 6. Linear Layers & Fused MLP | `linear_kernel`, `fused_linear_bias_gelu_kernel`, `mlp_swiglu_forward` | Dense projections and the full SwiGLU MLP block (gate + up + down projections) |
| 7. Composite Blocks | `rmsnorm_residual_block`, `run_transformer_ffn` | Compose everything above into the complete pre-norm transformer FFN forward pass |

## Key engineering ideas demonstrated

- **Warp-level parallelism**: `__shfl_xor_sync` for register-only reductions across 32 threads, no memory traffic
- **Two-level reduction**: warp reductions combined via shared memory to reduce across an entire thread block
- **Kernel fusion**: `fused_add_rmsnorm_kernel` and `fused_linear_bias_gelu_kernel` keep intermediate values in registers instead of round-tripping through global memory between separate kernel launches
- **Numerical stability**: max-subtraction before `exp()` in softmax to prevent overflow
- **Correct edge-case handling**: ceiling division (`(n + 31) / 32`) for block/thread counts that aren't exact multiples of 32

## Verified working

The full pipeline was assembled and run end-to-end via `scaffold.cu`, which drives all 20 kernels/host functions with generated input and confirms the final FFN output is finite and correctly computed. Independently compiled and run outside the original development platform:

```
$ nvcc -o scaffold scaffold.cu -arch=sm_75
$ ./scaffold
FFN out[0..3]: 42.815037 33.395168 -21.104704 -0.468143
FFN out[last]: -43.480938
scaffold OK
```

## How to run

Requires an NVIDIA GPU and the CUDA toolkit (`nvcc`). Easiest zero-setup option is a free Google Colab GPU runtime:

```bash
git clone https://github.com/intishar-rafi/Fused-LLM-Inference-Kernels-in-CUDA.git
cd Fused-LLM-Inference-Kernels-in-CUDA
nvcc -o scaffold scaffold.cu -arch=sm_75
./scaffold
```

## Files

- `model.cu` — all 20 kernels and host functions
- `scaffold.cu` — test harness that drives the full pipeline with generated input and prints the result
