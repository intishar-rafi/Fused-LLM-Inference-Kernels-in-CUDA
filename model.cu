// Fused LLM Inference Kernels in CUDA


// Step 1 - warp_reduce_sum
__device__ float warp_reduce_sum(float val) {
    // TODO: implement warp-level sum reduction using shuffle intrinsics
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xffffffff, val, offset);
    }
    return val;
}

// Step 2 - warp_reduce_max
__device__ float warp_reduce_max(float val) {
    // TODO: implement warp-level max reduction using shuffle intrinsics
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_xor_sync(0xffffffff, val, offset);
        val = fmaxf(val, other);
    }
    return val;
}

// Step 3 - block_reduce_sum
__device__ float block_reduce_sum(float val, float* shared) {
    // TODO: block-level sum via warp_reduce_sum + shared memory; result valid on thread 0
    int lane = threadIdx.x % 32;      // this thread's position within its own warp (0-31)
    int warp_id = threadIdx.x / 32;   // which warp this thread belongs to

    val = warp_reduce_sum(val);       // reduce within the warp first; every lane now holds its warp's sum

    if (lane == 0) {
        shared[warp_id] = val;        // one representative per warp writes the warp total to shared memory
    }
    __syncthreads();                  // wait for ALL warps to finish writing before anyone reads

    int num_warps = (blockDim.x + 31) / 32;   // ceiling division — number of warps, rounding UP
                                                // (blockDim.x / 32 would silently drop a partial warp)

    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    // only the first num_warps threads pick up one partial sum each from shared memory;
    // everyone else gets a harmless 0.0 since they're not needed for the final step

    if (warp_id == 0) {
        val = warp_reduce_sum(val);   // first warp combines the partial sums into the final total
    }

    return val;   // correct total lives on thread 0
}

// Step 4 - block_reduce_max
__device__ float block_reduce_max(float val, float* shared) {
    // TODO: block-wide max via warp_reduce_max + shared memory

    int lane = threadIdx.x % 32;      // this thread's position within its own warp (0-31)
    int warp_id = threadIdx.x / 32;   // which warp this thread belongs to

    val = warp_reduce_max(val);       // reduce within the warp first; every lane now holds its warp's max

    if (lane == 0) {
        shared[warp_id] = val;        // one representative per warp writes the warp max to shared memory
    }
    __syncthreads();                  // wait for ALL warps to finish writing before anyone reads

    int num_warps = (blockDim.x + 31) / 32;   // ceiling division — round UP to avoid dropping a partial warp

    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : -INFINITY;
    // only the first num_warps threads pick up one partial max each from shared memory;
    // everyone else gets -INFINITY (the identity value for max — can never win a max() comparison)

    if (warp_id == 0) {
        val = warp_reduce_max(val);   // first warp combines the partial maxes into the final result
    }

    return val;   // correct max lives on thread 0 (other threads' return values are undefined, per spec)
}

// Step 5 - add_residual_kernel
__global__ void add_residual_kernel(const float* x, const float* residual,
                                    float* out, int n) {
    // TODO: implement elementwise residual addition out[i] = x[i] + residual[i]

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // global index of this thread across the WHOLE grid, not just its block.
    // blockIdx.x = which block this thread is in (0, 1, 2, ...)
    // blockDim.x = how many threads per block (e.g. 256)
    // threadIdx.x = this thread's index within its own block (0 to blockDim.x-1)
    // e.g. block 2, thread 10, blockDim=256 -> i = 2*256 + 10 = 522

    if (i < n) {
        // bounds check: guards against threads that don't correspond to a real element.
        // Since blocks = ceil(n / threads), the last block may have MORE threads
        // than needed (e.g. n=100, threads=256 -> 256 threads launched but only
        // indices 0-99 are valid). Without this check, threads 100-255 would
        // read/write out of bounds memory -> undefined behavior / crash.
        out[i] = x[i] + residual[i];
    }
}

// Step 6 - gelu_kernel
__global__ void gelu_kernel(const float* x, float* out, int n) {
    // TODO: Apply GELU (tanh approximation) elementwise to x, write into out

    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index (same pattern as before)

    if (i < n) {   // bounds check (same pattern as before)
        float xi = x[i];
        float inner = 0.7978845608f * (xi + 0.044715f * xi * xi * xi);
        // 0.7978845608f is sqrt(2/π), precomputed as a constant so we don't
        // call sqrtf() every single thread — it's the same number regardless of xi
        out[i] = 0.5f * xi * (1.0f + tanhf(inner));
    }
}

// Step 7 - silu_kernel
__global__ void silu_kernel(const float* x, float* out, int n) {
    // TODO: apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))

    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index

    if (i < n) {   // bounds check
        float xi = x[i];
        out[i] = xi / (1.0f + expf(-xi));
        // direct translation of x / (1 + exp(-x))
        // expf() is the float version of e^(...), matching the tanhf/f-suffix convention
    }
}

// Step 8 - swiglu_kernel
__global__ void swiglu_kernel(const float* gate, const float* up, float* out, int n) {
    // TODO: out[i] = silu(gate[i]) * up[i] for all i in [0, n)

    int i = blockIdx.x * blockDim.x + threadIdx.x;   // global index (same pattern as every kernel so far)

    if (i < n) {   // bounds check — protects reads from BOTH gate[i] and up[i], and the write to out[i]
        float g = gate[i];
        float silu_g = g / (1.0f + expf(-g));   // SiLU applied to the gate value, same formula as silu_kernel
        out[i] = silu_g * up[i];                // gate the "up" projection by the SiLU-activated gate
    }
}

// Step 9 - rmsnorm_kernel
__global__ void rmsnorm_kernel(const float* x, const float* weight, float* out, int n, float eps) {
    // TODO: Apply RMSNorm per row (one block per row)

    int row = blockIdx.x;                    // which row this whole block is responsible for
    const float* x_row = x + row * n;         // pointer to the start of this row in x
    float* out_row = out + row * n;           // pointer to the start of this row in out

    extern __shared__ float shared[];         // scratch buffer for block_reduce_sum (one float per warp)

    // ---- Phase 1: compute sum of squares across the row ----
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xi = x_row[i];
        local_sum_sq += xi * xi;
    }

    float total_sum_sq = block_reduce_sum(local_sum_sq, shared);

    __shared__ float rms;                     // one shared value, visible to every thread in the block
    if (threadIdx.x == 0) {
        float mean_sq = total_sum_sq / n;
        rms = sqrtf(mean_sq + eps);
    }
    __syncthreads();                          // wait until thread 0 has written `rms`

    // ---- Phase 2: normalize and scale every element ----
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = (x_row[i] / rms) * weight[i];
    }
}

// Step 10 - layernorm_kernel
__global__ void layernorm_kernel(const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    // TODO: per-row LayerNorm using block_reduce_sum for mean and variance

    int row = blockIdx.x;
    const float* x_row = x + row * n;
    float* out_row = out + row * n;

    extern __shared__ float shared[];   // scratch buffer, reused for both reductions below

    // ---- Phase 1: accumulate sum(x) and sum(x^2) in one pass over the row ----
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float xi = x_row[i];
        local_sum += xi;
        local_sum_sq += xi * xi;
    }

    float total_sum = block_reduce_sum(local_sum, shared);
    __syncthreads();                     // shared[] must be free of stale data before reusing it below
    float total_sum_sq = block_reduce_sum(local_sum_sq, shared);

    __shared__ float mean;
    __shared__ float inv_std;            // 1 / sqrt(var + eps), precomputed once so Phase 2 can multiply instead of divide
    if (threadIdx.x == 0) {
        mean = total_sum / n;
        float mean_sq = total_sum_sq / n;
        float var = mean_sq - mean * mean;
        inv_std = rsqrtf(var + eps);
    }
    __syncthreads();

    // ---- Phase 2: normalize, scale, shift ----
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

// Step 11 - fused_add_rmsnorm_kernel
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    // TODO: fuse residual addition with RMSNorm (one block per row)

    int row = blockIdx.x;
    const float* x_row = x + row * n;
    const float* residual_row = residual + row * n;
    float* out_row = out + row * n;
    float* residual_out_row = residual_out + row * n;

    extern __shared__ float shared[];

    // ---- Phase 1: fused residual add + accumulate sum of squares ----
    float local_sum_sq = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float added = x_row[i] + residual_row[i];
        residual_out_row[i] = added;      // write the required residual_out immediately
        local_sum_sq += added * added;    // accumulate using the ADDED value, not raw x
    }

    float total_sum_sq = block_reduce_sum(local_sum_sq, shared);

    __shared__ float rms;
    if (threadIdx.x == 0) {
        float mean_sq = total_sum_sq / n;
        rms = sqrtf(mean_sq + eps);
    }
    __syncthreads();

    // ---- Phase 2: normalize the added value and scale by weight ----
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        out_row[i] = (residual_out_row[i] / rms) * weight[i];
    }
}

// Step 12 - softmax_row_kernel
__global__ void softmax_row_kernel(const float* x, float* out, int rows, int cols) {
    // TODO: implement numerically stable row-wise softmax (one block per row)

    int row = blockIdx.x;
    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    extern __shared__ float shared[];

    // ---- Phase 1: find the row max ----
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        local_max = fmaxf(local_max, x_row[i]);
    }
    float row_max = block_reduce_max(local_max, shared);

    __shared__ float m;
    if (threadIdx.x == 0) {
        m = row_max;
    }
    __syncthreads();   // make sure m is visible before Phase 2, AND shared[] is safe to reuse

    // ---- Phase 2: compute exp(x - m), store into out[] temporarily, accumulate sum ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = expf(x_row[i] - m);
        out_row[i] = e;          // stash the exponential here; will divide it in Phase 3
        local_sum += e;
    }
    float row_sum = block_reduce_sum(local_sum, shared);

    __shared__ float sum;
    if (threadIdx.x == 0) {
        sum = row_sum;
    }
    __syncthreads();

    // ---- Phase 3: normalize ----
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        out_row[i] = out_row[i] / sum;
    }
}

// Step 13 - causal_softmax_kernel
__global__ void causal_softmax_kernel(const float* x, float* out, int rows, int cols) {
    // TODO: numerically stable causal softmax (one block per row);
    //       mask columns c > row to 0; use block_reduce_max / block_reduce_sum

    int row = blockIdx.x;
    const float* x_row = x + row * cols;
    float* out_row = out + row * cols;

    extern __shared__ float shared[];

    // ---- Phase 1: max over ALLOWED columns only (i <= row) ----
    float local_max = -INFINITY;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i <= row) {
            local_max = fmaxf(local_max, x_row[i]);
        }
        // i > row: masked, contributes nothing to the max
    }
    float row_max = block_reduce_max(local_max, shared);

    __shared__ float m;
    if (threadIdx.x == 0) {
        m = row_max;
    }
    __syncthreads();

    // ---- Phase 2: exp(x - m) for allowed columns, zero for masked columns ----
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i <= row) {
            float e = expf(x_row[i] - m);
            out_row[i] = e;
            local_sum += e;
        } else {
            out_row[i] = 0.0f;   // masked position, written directly, never touched again
        }
    }
    float row_sum = block_reduce_sum(local_sum, shared);

    __shared__ float sum;
    if (threadIdx.x == 0) {
        sum = row_sum;
    }
    __syncthreads();

    // ---- Phase 3: normalize allowed columns; masked columns stay 0 ----
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        if (i <= row) {
            out_row[i] = out_row[i] / sum;
        }
        // i > row: already 0.0f from Phase 2, nothing to do
    }
}

// Step 14 - embedding_lookup_kernel
__global__ void embedding_lookup_kernel(const int* token_ids, const float* weight, float* out, int seq_len, int vocab_size, int embed_dim) {
    // TODO: gather embedding vectors for each token id into out

    int idx = blockIdx.x * blockDim.x + threadIdx.x;   // flat index over ALL output floats
    int total = seq_len * embed_dim;

    if (idx < total) {
        int row = idx / embed_dim;   // which sequence position (0 to seq_len-1)
        int col = idx % embed_dim;   // which dimension within that embedding vector

        int token = token_ids[row];  // the vocabulary ID stored at this sequence position

        out[idx] = weight[token * embed_dim + col];
        // weight is [vocab_size, embed_dim] row-major, so row `token`'s data starts at
        // token * embed_dim, then + col picks the specific dimension — same pointer
        // arithmetic pattern as x_row = x + row * n from rmsnorm_kernel, just inlined here
    }
}
// Step 15 - rope_kernel
__global__ void rope_kernel(float* q, float* k,
                            const float* cos_table, const float* sin_table,
                            int seq_len, int n_heads, int head_dim) {
    // TODO: apply RoPE rotation in-place to every even/odd pair of q and k

    int half = head_dim / 2;
    int total = seq_len * n_heads * half;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total) {
        int pair = idx % half;              // which pair within the head (0 to half-1)
        int tmp = idx / half;
        int head = tmp % n_heads;           // which head
        int pos = tmp / n_heads;            // which sequence position

        int base = pos * n_heads * head_dim + head * head_dim;   // start of this (pos, head)'s vector
        int i_even = base + 2 * pair;
        int i_odd  = base + 2 * pair + 1;

        int cs_idx = pos * half + pair;     // index into cos_table/sin_table, shaped [seq_len, half]
        float c = cos_table[cs_idx];
        float s = sin_table[cs_idx];

        // --- rotate q ---
        float q_even = q[i_even];
        float q_odd  = q[i_odd];
        q[i_even] = q_even * c - q_odd * s;
        q[i_odd]  = q_even * s + q_odd * c;

        // --- rotate k (identical formula, same cos/sin) ---
        float k_even = k[i_even];
        float k_odd  = k[i_odd];
        k[i_even] = k_even * c - k_odd * s;
        k[i_odd]  = k_even * s + k_odd * c;
    }
}

// Step 16 - linear_kernel
__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // TODO: compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int m = idx / N;   // which row of x / which output row
        int n = idx % N;   // which row of weight / which output column

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += x[m * K + k] * weight[n * K + k];
        }

        if (bias != nullptr) {
            sum += bias[n];
        }

        out[idx] = sum;
    }
}

// Step 17 - fused_linear_bias_gelu_kernel
__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
    // TODO: fuse matmul, bias add, and GELU tanh approx into one kernel

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;

    if (idx < total) {
        int m = idx / N;
        int n = idx % N;

        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += x[m * K + k] * weight[n * K + k];
        }
        sum += bias[n];

        // GELU tanh approximation, applied in-register before ever touching global memory
        float inner = 0.7978845608f * (sum + 0.044715f * sum * sum * sum);
        out[idx] = 0.5f * sum * (1.0f + tanhf(inner));
    }
}

// Step 18 - mlp_swiglu_forward
void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up,
                        const float* w_down, float* out,
                        int M, int hidden_dim, int intermediate_dim) {
    // TODO: allocate temps, run gate/up linears, swiglu, then down projection

    int H = hidden_dim;
    int I = intermediate_dim;

    // ---- allocate temporary device buffers ----
    float *d_gate, *d_up, *d_swiglu;
    cudaMalloc(&d_gate,   M * I * sizeof(float));   // x @ w_gate^T  -> [M, I]
    cudaMalloc(&d_up,     M * I * sizeof(float));   // x @ w_up^T    -> [M, I]
    cudaMalloc(&d_swiglu, M * I * sizeof(float));   // silu(gate) * up -> [M, I]

    int threads = 256;

    // ---- gate projection: d_gate = x @ w_gate^T  (no bias -> pass nullptr) ----
    int blocks1 = (M * I + threads - 1) / threads;
    linear_kernel<<<blocks1, threads>>>(x, w_gate, nullptr, d_gate, M, I, H);

    // ---- up projection: d_up = x @ w_up^T ----
    linear_kernel<<<blocks1, threads>>>(x, w_up, nullptr, d_up, M, I, H);

    // ---- swiglu: d_swiglu = silu(d_gate) * d_up ----
    int total_i = M * I;
    int blocks2 = (total_i + threads - 1) / threads;
    swiglu_kernel<<<blocks2, threads>>>(d_gate, d_up, d_swiglu, total_i);

    // ---- down projection: out = d_swiglu @ w_down^T  -> [M, H] ----
    int blocks3 = (M * H + threads - 1) / threads;
    linear_kernel<<<blocks3, threads>>>(d_swiglu, w_down, nullptr, out, M, H, I);

    // ---- free temporaries ----
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_swiglu);
}

// Step 19 - rmsnorm_residual_block
void rmsnorm_residual_block(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int rows,
    int n,
    float eps
) {
    // TODO: launch fused_add_rmsnorm_kernel for the pre-norm residual+RMSNorm block

    int threads = 256;
    int num_warps = (threads + 31) / 32;              // how many warps per block
    size_t shared_bytes = num_warps * sizeof(float);   // one float of scratch per warp

    fused_add_rmsnorm_kernel<<<rows, threads, shared_bytes>>>(
        x, residual, weight, out, residual_out, n, eps
    );
}

// Step 20 - run_transformer_ffn
void run_transformer_ffn(const float* x, const float* residual,
                         const float* norm_weight, const float* w_gate,
                         const float* w_up, const float* w_down, float* out,
                         int M, int hidden_dim, int intermediate_dim,
                         float eps) {
    // TODO: residual+RMSNorm, SwiGLU MLP, then residual add into out

    int H = hidden_dim;

    // ---- allocate temp buffers for intermediate stages ----
    float *d_residual_out, *d_normed, *d_mlp_out;
    cudaMalloc(&d_residual_out, M * H * sizeof(float));
    cudaMalloc(&d_normed,       M * H * sizeof(float));
    cudaMalloc(&d_mlp_out,      M * H * sizeof(float));

    // ---- step 1: residual_out = x + residual; normed = RMSNorm(residual_out) * weight ----
    rmsnorm_residual_block(x, residual, norm_weight, d_normed, d_residual_out,
                           M, H, eps);

    // ---- step 2: mlp_out = SwiGLU_MLP(normed) ----
    mlp_swiglu_forward(d_normed, w_gate, w_up, w_down, d_mlp_out,
                       M, H, intermediate_dim);

    // ---- step 3: out = residual_out + mlp_out ----
    int threads = 256;
    int total = M * H;
    int blocks = (total + threads - 1) / threads;
    add_residual_kernel<<<blocks, threads>>>(d_residual_out, d_mlp_out, out, total);

    // ---- free temporaries ----
    cudaFree(d_residual_out);
    cudaFree(d_normed);
    cudaFree(d_mlp_out);
}


