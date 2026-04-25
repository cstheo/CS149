# Lecture 9 — Efficiently Evaluating DNNs

## Overview
Shows how everything learned about arithmetic intensity, blocking, SIMD, and the memory hierarchy applies to the dominant workload of modern AI: evaluating deep neural networks. The lecture argues that once you understand the roofline model plus loop-fusion/blocking, you understand the software side of DNN performance. It walks through the structure of convolutional and transformer networks, reduces their core layers to dense matrix multiplication (GEMM), and then develops the key implementation tricks: im2col / implicit GEMM, hierarchical blocking with SIMD inner kernels, fusion of conv+bias+ReLU+pool, the online-softmax trick behind FlashAttention, and the library/compiler ecosystem (cuDNN, CUTLASS, Triton, ThunderKittens, torch.compile).

## Key Concepts
- **Roofline model** — a program's achieved throughput is `min(peak_compute, BW · arithmetic_intensity)`. Higher intensity or faster hardware shifts the balance.
- **Arithmetic intensity (AI)** — ops per byte of off-chip traffic. The single most important knob for DNN performance.
- **Loop fusion** — combine producer/consumer passes into one loop so intermediates stay in registers/cache; raises AI by avoiding round-trips to DRAM.
- **Fully-connected layer** — `y = f(Wx + b)`; a matrix-vector product (batched: matrix-matrix).
- **2D convolution** — each output pixel is a dot product of a small `R×S` filter with a local input window; `K` filters produce `K` output channels; weights are *shared* across spatial positions.
- **im2col / explicit GEMM** — materialize a `(P·Q) × (R·S·C)` matrix from the input so that conv becomes a standard dense GEMM. Costs `R·S`× DRAM traffic and extra storage.
- **Implicit GEMM** — index directly into the activation tensor and only materialize a sub-block of the conv matrix in on-chip shared memory (CUTLASS approach). No extra DRAM traffic.
- **Hierarchical blocking** — tile for L2, L1, registers, and SIMD lanes so that reused blocks of A, B, C stay resident.
- **Depthwise separable conv** — factor a standard conv into a per-channel `3×3` depthwise conv + a `1×1` pointwise conv (MobileNet). Most of the compute becomes `1×1`, which is just GEMM.
- **Attention GEMM** — `S = QKᵀ`, `P = softmax(S)`, `O = PV`. `S` and `P` are `N×N`, so naive implementations are `O(N²)` in memory.
- **Online (streaming) softmax** — compute softmax in chunks by carrying running `m(x)` and `l(x)` and rescaling; enables fused attention (FlashAttention) without materializing `N×N`.
- **Low precision** — fp16, bf16, int8, int4, and beyond; trade numeric precision for bandwidth and throughput.

## Detailed Notes

### Why this lecture exists
Modern AI workloads (GoogLeNet/Inception, MobileNet, transformers) must run on a huge range of devices — datacenter GPUs, phones, NPUs — so evaluating a DNN efficiently is an extreme-efficiency problem. But the tools are familiar: the roofline, fusion, and blocking from earlier lectures.

### Recap: roofline + fusion (the whole software story)
The roofline plot has AI on the x-axis and achieved Ops/sec on the y-axis: a diagonal BW-bound region rising until it meets a horizontal compute-bound ceiling. Improving the memory system pushes the diagonal up; improving compute pushes the ceiling up; improving AI moves the program rightward on the curve.

The canonical way to raise AI is **loop fusion**:

```c
// Program 1: three passes, AI ≈ 1/3
void add(int n, float* A, float* B, float* C) {
    for (int i=0;i<n;i++) C[i] = A[i] + B[i];
}
void mul(int n, float* A, float* B, float* C) {
    for (int i=0;i<n;i++) C[i] = A[i] * B[i];
}
add(n, A, B, tmp1);
mul(n, tmp1, C, tmp2);
add(n, tmp2, D, E);

// Program 2: fused, AI = 3/5
void fused(int n, float* A, float* B, float* C, float* D, float* E) {
    for (int i=0;i<n;i++)
        E[i] = D[i] + (A[i] + B[i]) * C[i];
}
```

Two reminders for next time: (1) data movement costs energy, and (2) on-chip storage competes with compute — so minimize buffers.

### What a DNN is, computationally
A neuron is `f(Σ wᵢxᵢ + b)` — an inner product plus a nonlinearity (e.g., ReLU `max(0,x)`). A fully-connected layer applies this unit across a whole output vector, so it is exactly a matrix-vector product with an elementwise nonlinearity on top. Deep networks stack such layers; the network's "topology" is just how they are wired.

### Convolutional layers
A 2D convolution slides a small `R×S` filter over an input image:
```c
for (int j=0; j<H; j++)
  for (int i=0; i<W; i++) {
    float tmp = 0.f;
    for (int jj=0; jj<3; jj++)
      for (int ii=0; ii<3; ii++)
        tmp += input[(j+jj)*(W+2) + (i+ii)] * weights[jj*3 + ii];
    output[j*W + i] = tmp;
  }
```
Key properties: (a) **locally connected** — each output depends only on a small window of inputs; (b) **weight sharing** — the same `R·S` weights are reused across every spatial location. Applying many filters produces a stack of response maps (one per filter), and subsequent Conv/ReLU/Pool layers build deeper features (with pooling reducing spatial resolution).

The full batched conv has seven nested loops (batch, y, x, filter, in-channel, fy, fx):

```c
for (int img=0; img<N; img++)
 for (int j=0; j<H; j++)
  for (int i=0; i<W; i++)
   for (int f=0; f<K; f++) {
     float tmp = bias[f];
     for (int kk=0; kk<C; kk++)
      for (int jj=0; jj<R; jj++)
       for (int ii=0; ii<S; ii++)
         tmp += W_[f][jj][ii][kk] * X[img][j+jj][i+ii][kk];
     Y[img][j][i][f] = tmp;
   }
```
Huge reuse potential: each weight is used at every `(i,j)`; each input element is used by every filter.

### Conv as GEMM (im2col / explicit GEMM)
Unfold each `R·S·C` input patch into a column of a matrix `X'`, stack the `K` filters as rows of a weight matrix `W'`, then `Y' = W' · X'` is a dense matrix-matrix multiply. Symbol table: filter support `R×S`, input channels `C`, number of filters `K`, batch `N`, output spatial `P×Q`.

Pros: reuses highly tuned GEMM (MKL, cuBLAS). Cons: the unfolded input matrix is `R·S`× bigger than the original activations — that's extra DRAM traffic and storage just to make GEMM happy.

### GEMM tuning: blocking and SIMD
Naive matmul has terrible AI:
```c
for (int j=0; j<M; j++)
  for (int i=0; i<N; i++)
    for (int k=0; k<K; k++)
      C[j][i] += A[j][k] * B[k][i];
```
Block it so a tile of `C` accumulates while tiles of `A` and `B` stay in cache:
```c
for (int jb=0; jb<M; jb+=BJ)
 for (int ib=0; ib<N; ib+=BI)
  for (int kb=0; kb<K; kb+=BK)
   for (int j=0; j<BJ; j++)
    for (int i=0; i<BI; i++)
     for (int k=0; k<BK; k++)
       C[jb+j][ib+i] += A[jb+j][kb+k] * B[kb+k][ib+i];
```
Block size is bounded by cache capacity — too large and the block no longer fits; too small and reuse is lost. Real implementations **hierarchically block** for L2, L1, registers, then vectorize the inner loop.

Three SIMD schemes for the inner kernel:
1. **Splat A, vector B** — vectorize `i`, `splat` one `A[j][k]` across a lane vector and `simd_muladd` with a contiguous row of `B`. Good spatial locality on `B` but inflates working set by `SIMD_WIDTH` and still strides over `B`.
2. **Dot-product with transposed B** — pre-transpose the block of `B`, then vectorize `k` and use `simd_dot` of one row of `A` with one row of `Bᵀ`. Works when `i` is small.
3. **Outer-product with transposed A and C** — accumulate a `SIMD_WIDTH × SIMD_WIDTH` tile of `Cᵀ`, reading vectors from `B` and splatting scalars from `A`. This is the pattern tensor-core kernels use.

### Implicit GEMM (CUTLASS-style)
Keep the input tensor in its original layout. For each GPU tile of output, materialize only the required sub-block of the convolution matrix into shared memory on the fly, then run a fast shared-memory GEMM. No extra DRAM traffic, no `R·S`× storage blowup. NVIDIA CUTLASS exposes the building blocks: shared-memory GEMM, warp-level GEMM, tile iterators, reductions.

### MobileNet and shape diversity
MobileNet factors standard conv into depthwise (`3×3` per channel) + pointwise (`1×1` × channels), putting ~95% of compute into dense `1×1` convs that are pure GEMM. But different layers in one network have wildly different shapes — a small spatial map with many channels looks nothing like early layers — so **no single schedule is optimal for every layer**. Library writers must pick or search a strategy per layer.

### Attention is also GEMM
In a transformer, `Q, K, V ∈ ℝ^{N×d}`. Compute `S = QKᵀ` (`N×N`), `P = softmax_rows(S)` (`N×N`), `O = PV` (`N×d`). For long sequences `N` can be thousands, so `N²` storage is painful. But both the projections and `S`, `O` products are GEMMs — the same kernel dominates fully-connected layers, conv layers (via im2col/implicit GEMM), and attention.

### Memory traffic between layers → fuse!
Consider `Conv → Scale/Bias → MaxPool`. Writing 1 GB of conv output to DRAM, re-reading it for scale/bias, writing it back, re-reading to pool — catastrophic. Better: produce each conv output element, immediately apply scale+bias, and feed a 2×2 accumulator for the pool. One streaming pass, near-zero intermediate traffic:

```c
for (...spatial, filter...) {
  float tmp = 0.f;
  // ... inner accumulate ...
  output[img][j][i][f] = tmp*scale[f] + bias[f];   // fused
}
```
Pool fusion: block the spatial output loops by 2, keep the four partial outputs in registers, emit one pooled value per block.

### Softmax and the online trick
Naive row softmax reads and writes the full `M×N` matrix three times (max, exp/sum, divide) → `~5MN` reads, `~3MN` writes, AI ≈ constant. Fused softmax: for each row, stream it through on-chip memory once, computing `m`, `f`, `l`, and writing normalized values — `MN` reads + `MN` writes, provided a row fits on-chip.

Softmax is also associative in a specific sense, which allows **chunked / online** computation. Split `x = [x^{(1)} | x^{(2)}]`:
```
m(x)   = max(m(x^{(1)}), m(x^{(2)}))
f(x)   = [ e^{m(x^{(1)})-m(x)} f(x^{(1)}),  e^{m(x^{(2)})-m(x)} f(x^{(2)}) ]
l(x)   = e^{m(x^{(1)})-m(x)} l(x^{(1)}) + e^{m(x^{(2)})-m(x)} l(x^{(2)})
```
I.e., you can softmax a long row by sweeping through chunks while carrying `(m, l)` and rescaling previously-accumulated partial results.

### FlashAttention: fused attention
Using online softmax, never materialize `S` or `P`. Pseudocode:
```
for each j (block of K/V columns):
  for each i (block of Q rows):
    load Qi, Kj, Vj, Oi, (mi, li)
    Sij = Qi · Kjᵀ
    compute Mij = rowmax(Sij), Pij = exp(Sij - Mij), lij = rowsum(Pij)
    update (mi, li) and rescale Oi, then Oi += Pij · Vj
    store Oi, (mi, li)
```
Benefits: (1) no `N²` matrix in DRAM; (2) only 3 block reads (Q, K, V) and one block write (O) per outer iteration; (3) `O` stays in on-chip cache. Cost: some redundant arithmetic (re-scaling `O` on every step), but compute is cheap compared to bandwidth. ThunderKittens provides tile primitives (async tile load/store, layouts) that make writing FlashAttention-style kernels tractable for CS149-level programmers.

### Library/compiler landscape
- **cuDNN** — NVIDIA's pre-tuned DNN kernels. Offers multiple convolution algorithms (direct, im2col-GEMM, implicit GEMM, Winograd, FFT); the framework picks one based on shape.
- **CUTLASS** — building blocks (shared-memory GEMMs, warp GEMMs, iterators) for writing your own kernels when shapes don't match cuDNN's tuned sizes.
- **Triton** — Python-embedded language for "load blocks, do block ops, store blocks." Full matmul is ~20 lines with two levels of blocking.
- **ThunderKittens** — CUDA C++ tile-programming library; async tile loads, rich memory layouts.
- **torch.compile / XLA / TVM / cuDNN backend** — compiler stacks that automatically fuse operator graphs into custom kernels, avoiding intermediate tensors and per-op launch overhead. Replaces the older model of "library writer hand-codes a few fused ops."
- **AWS NKI, vendor SDKs** — per-chip low-level libraries for specialized accelerators.

### Why GPUs (and why maybe not)
Good fit: DNNs are dominated by dense GEMM with high AI, which needs lots of flops and high bandwidth; GPUs have both, plus tensor cores, plus cuDNN. Not ideal: GPUs are general-purpose, so they spend area/power on features (caches, schedulers, general ISA) that a pure-inference chip could strip out. Next lecture moves to specialized accelerators (TPU, NPU, Trainium, IPU, Cerebras WSE, Apple Neural Engine) that take this further.

### Optimization axes (summary)
- **Model design** — depth, width, filter count, stride; sometimes auto-searched (NAS).
- **Scheduling** — blocking, fusion, layout; mostly hand-tuned, increasingly compiler-driven.
- **Approximation** — low-precision (fp16 / bf16 / int8 / int4 / 1-bit), sparsity.

## Examples / Worked Problems
- Loop-fusion exercise raising AI from `1/3` to `3/5` (shown above).
- Matmul blocking exercise — pick `BJ, BI, BK` so `A`, `B`, `C` tiles all fit in L1; self-check question: "do you want as big a BLOCKSIZE as possible?" No: too big and blocks no longer fit in cache, so reuse is lost.
- Three SIMD inner kernels for blocked GEMM (splat-A, transposed-B dot, outer-product on transposed A/C).
- Fuse `Conv → Scale/Bias` into the conv loop by computing `tmp*scale + bias` at write-out; extension: also fuse a 2×2 max-pool by blocking the spatial output loops by 2.
- Online softmax chunk-combining formulas (shown above).
- FlashAttention pseudocode — two nested block loops over Q rows and K/V columns, carrying `(m, l, O)`.
- Tiling question for attention: V100 has 80 SMs and 900 GB/s HBM; small `(N=1, P=Q=64)` produces only 2 MB of output — not enough parallel work to fill the machine. `(N=32, P=Q=256)` produces 1 GB and saturates it.

## Takeaways
1. **Dense GEMM dominates DNN compute.** FC, conv (via im2col or implicit GEMM), and attention all reduce to matrix-matrix multiply.
2. **Arithmetic intensity is the lever.** Blocking raises AI against cache/SRAM; fusion raises AI across operator boundaries.
3. **Don't materialize what you don't have to.** Implicit GEMM avoids the `R·S`× im2col blow-up; FlashAttention avoids the `N²` attention matrix entirely.
4. **Different layers want different schedules.** One network contains shapes that favor different tilings — hence the rise of per-shape tuning (cuDNN heuristics) and compilers (torch.compile, Triton autotune).
5. **The software toolchain is a hierarchy.** cuDNN for the common case, CUTLASS for custom kernels, Triton/ThunderKittens for productive tile-level code, compilers for automatic fusion.
6. **Low precision is free AI.** Every halving of element size doubles effective bandwidth and often throughput.

## Open Questions / Follow-ups
- Winograd and FFT-based convolution algorithms (listed by cuDNN but not covered) — when do they win over GEMM?
- Automatic scheduling research — Halide, TVM, Triton autotuner, Ansor: how close are they to hand-tuned?
- How far can low precision go before accuracy collapses? 4-bit and 1-bit regimes.
- Exactly how specialized accelerators differ from GPUs — deferred to the next lecture on TPUs/NPUs.
- For attention: how do KV-cache reuse, paged attention, and speculative decoding fit into this same arithmetic-intensity framework?

## Sources
- Slides: `slices/09_dnneval.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Papers/figures referenced: GoogLeNet / Inception-v4 (Szegedy et al.), MobileNet (Howard et al.), FlashAttention (Dao et al.).
- Libraries: NVIDIA cuDNN, CUTLASS, Triton (OpenAI), ThunderKittens (Stanford Hazy Research), torch.compile, AWS NKI.
- Course site: https://gfxcourses.stanford.edu/cs149
