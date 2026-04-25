# Lecture 11 — Programming Specialized Hardware for AI

## Overview
AI workloads are driving a Cambrian explosion of custom silicon — TPUs, GPUs with tensor cores, dataflow chips — each trading programmability for energy efficiency. This lecture surveys three representative points in that spectrum: Google's TPU (systolic arrays for dense matmul), NVIDIA's H100/B100 (asynchronous compute/memory, tamed by DSLs like ThunderKittens), and SambaNova's SN40L RDU (reconfigurable dataflow with metapipelining). The common thread: specialized hardware wins by packing many arithmetic units, customizing on-chip datapaths to move intermediates directly between units, and keeping large on-chip buffers — but each architecture pushes a different programming model onto the kernel author.

## Key Concepts
- **Efficiency vs. programmability spectrum** — CPU → GPU → DSP → domain accelerator (TPU) → FPGA → ASIC. Each step right gains ~10–1000× efficiency but loses general programmability.
- **Systolic array** — a 2D grid of PEs where data rhythmically flows between neighbors and multiplies/accumulates in place. Data-driven, no instructions per PE.
- **Dataflow type** — Weight-Stationary (WS), Output-Stationary (OS), or Input-Stationary (IS), depending on which operand sits in each PE vs. streams through.
- **Synchronous vs. asynchronous execution** — blocking issue serializes LD/AO/ST; async lets later ops start before earlier ones finish, enabling overlap of memory and compute.
- **Tensor Core / TMA / TMEM** — specialized MMA units, Tensor Memory Accelerator for bulk async copies, and tensor memory holding operands outside the register file. On B100 a single thread issues MMAs — warps are no longer the unit of tensor-core work.
- **Hardware lottery** — ideas win when they fit the available hardware (dense matmul → transformers), not necessarily because they are universally superior.
- **Dataflow architecture (RDA)** — no instruction fetch/decode; AI graph is mapped spatially onto a reconfigurable fabric of Pattern Compute Units (PCUs) and Pattern Memory Units (PMUs) connected by switches.
- **Metapipelining** — hierarchical coarse-grained pipelining ("pipeline of pipelines") that streams tiles through stages with double-buffered FIFOs, exploiting nested-loop parallelism.
- **Kernel fusion on RDU** — one kernel per decoder vs. ~800 GPU kernel calls per token; eliminates launch overhead and GBs of off-chip intermediate traffic.

## Detailed Notes

### Efficiency vs. programmability spectrum
From easiest to hardest to program: energy-optimized CPU → throughput-oriented GPU (~10×) → programmable DSP → domain-specific accelerator such as Google TPU (~20×) → FPGA / reconfigurable logic (~50×, jury still out) → fixed-function ASIC (~100–1000×, not programmable, costs 10–100M$ to tape out). DSLs (for DNNs especially) try to make the programmable side of that spectrum usable without giving up too much performance.

### Sync vs. async execution
In synchronous execution a tile pipeline runs `LD0 → AO0 → ST0 → LD1 → AO1 → ST1 …` strictly serially. In asynchronous execution the hardware or software overlaps them: while AO_a0 runs, LD_a1 is already issued, and ST_a0 overlaps with AO_a1. Achieving this requires async instructions, explicit barriers/fences, and/or OoO hardware.

### Google TPU v1
On the die, arithmetic units occupy ~30% of area; control is tiny, which is the whole point. Its ISA is short: read/write host memory, read weights, matrix_multiply/convolve, activate.

**Systolic array matrix-vector product** (`y = W x`): weights are loaded once per PE in a 2D grid. Inputs `x0, x1, x2, x3` enter from the left, one column per cycle, staggered so each PE sees the correct `xi` paired with its resident `wij`. Partial sums walk downward, so the bottom row's accumulators see `y_j = Σ_i x_i · w_ij` after the wavefront passes. Extending to matmul (`Y = W X`): additional input columns trail behind, needing multiple 32-bit accumulators per output column to hold separate Y-columns simultaneously.

**Dataflow types**:

| Type | Stays in PE | Streams through | Goal |
|---|---|---|---|
| Weight-Stationary | Weights | Activations + partial sums | Minimize reloading weights |
| Output-Stationary | Partial sums | Inputs + weights | Minimize moving accumulated results |
| Input-Stationary | Activations | Weights + partial sums | Minimize reloading inputs |

**SIMD vs. systolic**: SIMD is control-driven (instructions broadcast), has limited data reuse, global register/memory communication, centralized control. Systolic is data-driven (wavefront), temporal+spatial reuse, neighbor-local communication, distributed control — hence higher perf/mm² and perf/W.

**Scaling up**: for `C(8×4096) = A(8×8) · B(8×4096)` with 4096 accumulators, the TPU streams B-columns through the array while keeping A resident and accumulating columns of C in place.

**Hardware lottery**: the TPU was built around dense MM (arithmetic intensity ∝ n). Transformer models happen to be dense MM dominant, so they thrive — reinforcing HW specialization and further entrenching transformers. Sara Hooker's point: winners are sometimes hardware-suited, not universally better.

### NVIDIA H100 / B100 tensor cores
Modern tensor cores are fed from shared memory (SMEM) and new tensor memory (TMEM), not the register file, because register bandwidth can't keep up. On B100 a **single thread** issues MMAs — warps no longer gate the tensor pipeline. Programming it looks like:

- `tcgen05.alloc` — allocate TMEM and descriptors.
- `cp.async.bulk.tensor` + `mbarrier` — prefetch/stream tiles with the TMA asynchronously.
- `tcgen05.mma` + `tcgen05.commit` — launch async MMA batches.
- `tcgen05.fence` — order and retire. "Not your father's CUDA."

**TMA (Tensor Memory Accelerator)** is a block data-movement engine: a single thread issues an async load/store of a tensor region described by a *copy descriptor*; a barrier signals completion. Bypasses L1 and eliminates thousands of address-calc instructions.

**How ideal is a GPU?**

| Ideal feature | Why | H100 status |
|---|---|---|
| Tiled tensors (16×16, 32×32) | Max TFLOPS, low instr overhead | yes |
| Async compute | Overlap compute/memory | yes (`mma_async`) |
| Async memory access | Overlap compute/memory | yes (TMA+TMEM) |
| Async chip-to-chip | Overlap compute/mem/comm | partial (TB Cluster) |
| Compute-unit to compute-unit comm | Fusion, streaming dataflow | weak |

**Why kernels matter**: NVIDIA's 2025 quarterly revenue is >$47B; big AI runs consume clusters worth hundreds of millions for months. FlashAttention-2 went from ~70% MFU on A100 to ~35% on H100; it took two years and FA-3 to climb back to ~65%. Poor kernels burn billions in compute. Essentially all TFLOPS sit in tensor cores (~94–98% on recent SKUs), so any kernel that doesn't keep them fed is leaving the machine idle.

### ThunderKittens — an embedded CUDA DSL
Template library on top of CUDA, with three design principles:
1. **16×16 tile as primitive data type** — TK manages register/shared-memory layouts; provides basic ops.
2. **Asynchrony everywhere** — expose primitives so experts can hand-schedule.
3. **High-level GPU coordination** — built-in producer/consumer patterns.

Templated types: register tiles (`rt_*`), register vectors, shared tiles (`st_*`), shared vectors. Operations: initializers, unary (`exp`), binary (`mul`), row/col ops (`row_sum`). The canonical tile pipeline is **Global → Shared (load) → Registers → Tensor cores (compute) → Registers → Shared (store) → Global**; TK maps producer warps to the load/store legs and consumer warps to the compute leg.

#### TK matmul — Step 1: layouts
```cuda
#include "kittens.cuh"
#include "prototype.cuh"
using namespace kittens;
using namespace kittens::prototype;
using namespace kittens::prototype::lcf;

struct matmul_layout {
    using a_global_layout = gl<bf16, 1, 1, -1, -1, st_bf<64, 64>>;   // TMA desc for 64x64 A-tile
    using b_global_layout = gl<bf16, 1, 1, -1, -1, st_bf<64, 256>>;  // TMA desc for 64x256 B-tile
    using c_global_layout = gl<bf16, 1, 1, -1, -1>;                  // no TMA desc for C
    struct globals       { a_global_layout A; b_global_layout B; c_global_layout C; };
    struct input_block   { st_bf<64, 64> a[2]; st_bf<64, 256> b; };  // SMEM input tiles
    struct finish_block  { st_bf<64, 256> c[2]; };                   // SMEM output tiles
    struct consumer_state{ rt_fl<16, 256> accum; };                  // register accumulator
};
```

#### Step 2: pipeline + producers
```cuda
struct matmul_template {
    using layout = matmul_layout;
    static constexpr int NUM_CONSUMER_WARPS=8, INPUT_PIPE_STAGES=4;
    static constexpr int PRODUCER_BARRIER_ARRIVALS=1, CONSUMER_BARRIER_ARRIVALS=2;

    __device__ static inline void common_setup(common_setup_args<layout> args) {
        args.num_iters = args.task_iter == 0 ? args.globals.A.cols/64 : -1;
    }
    struct producer {
        __device__ static void setup(producer_setup_args<layout> args) {
            warpgroup::decrease_registers<40>();
        }
        __device__ static void load(producer_load_args<layout> args) {
            if (warpgroup::warpid() == 0) {
                tma::expect(args.inputs_arrived, args.input);
                for (int i = 0; i < 2; i++) {
                    tma::load_async(args.input.a[i], args.globals.A,
                                    {blockIdx.x*2+i, args.iter}, args.inputs_arrived);
                }
                tma::load_async(args.input.b, args.globals.B,
                                {args.iter, blockIdx.y}, args.inputs_arrived);
            }
        }
    };
```

#### Step 3: consumers — the compute
```cuda
    struct consumer {
        __device__ static void setup(consumer_setup_args<layout> args) {
            warpgroup::increase_registers<232>();
            zero(args.state.accum);
        }
        __device__ static void compute(consumer_compute_args<layout> args) {
            warpgroup::mma_AB(args.state.accum,
                              args.input.a[warpgroup::groupid()],
                              args.input.b);
            warpgroup::mma_async_wait();
            if (warpgroup::laneid() == 0) arrive(args.inputs_finished);
        }
        __device__ static void finish(consumer_finish_args<layout> args) {
            int wg = warpgroup::groupid();
            warpgroup::store(args.finish.c[wg], args.state.accum);
            warpgroup::sync();
            warpgroup::store(args.globals.C, args.finish.c[wg], args.state.accum,
                             {blockIdx.x*2+wg, blockIdx.y});
        }
    };
};
```
Producers decrease their register budget so consumers can claim 232 registers and hold big accumulators. A single lane arrives on the barrier when a tile is consumed. Result writes bounce through SMEM first so the store to HBM coalesces cleanly.

### SambaNova SN40L — reconfigurable dataflow
An AI model is a dataflow graph of GEMM + parallel patterns (map, reduce, zip, filter, gather/scatter). A dataflow processor maps that graph spatially: **PCUs** (Pattern Compute Units, 16×8 bf16 systolic+SIMD), **PMUs** (Pattern Memory Units, 0.5 MB scratchpads with flexible addressing), **AGCUs** (Address Generator/Coalescing Units, the portal to HBM), and a mesh of switches provide the interconnect. SN40L: 1,040 PCUs/PMUs, 638 bf16 TFLOPS, 520 MB on-chip SRAM, 64 GB HBM, 1.5 TB DDR. No instruction fetch/decode; execution is extremely asynchronous.

**Composable primitives**: MM, Map, Zip, Reduce, Gather, Scatter. The compiler tiles, parallelizes, metapipelines, then places & routes onto the fabric.

### Metapipelining
A pipeline of pipelines that exploits nested-loop parallelism: each parallel-pattern loop becomes a streaming pipeline by inserting pipe stages inside the loop body; stages run concurrently, overlapping loop iterations. Intermediate tiles live in **double buffers** between stages, tolerating imbalanced stage latencies. Plays well with tiling and with fusion — in fact metapipelining can stream-fuse when classical kernel fusion can't.

**GDA intuition**: `map(N) { r => row = matrix.slice(r); diff = row - sub; vprod = outer(diff, diff); }` maps to four stages — `ld`, `sub`, `outer-mul`, `store vprod` — with AGCU and PMU stages between them. Two iterations (`r=12`, `r=25314`) run in lockstep across the four stages.

#### Matmul metapipe
```cpp
auto format = DataFormat::kBF16;
int64_t M = args::M.getValue();
int64_t N = args::N.getValue();
int64_t K = args::K.getValue();

auto A = INPUT_REGION("A", (M, K), format);
auto B = INPUT_REGION("B", (K, N), format);
auto C = OUTPUT_REGION("C", (M, N), format);

auto MM = 256, NN = 64;
auto a_tile_shape = std::vector<int64_t>({MM, K});
auto b_tile_shape = std::vector<int64_t>({K, NN});
auto c_tile_shape = std::vector<int64_t>({MM, NN});

METAPIPE(M / MM, [&]() {
    auto a_tile = LOAD_TILE(A, a_tile_shape);
    METAPIPE(N / NN, [&]() {
        auto b_tile = LOAD_TILE(B, b_tile_shape, row_par = 4);
        auto c      = MAT_MUL(a_tile, b_tile);
        auto c_tile = BUFFER(c);
        STORE_TILE(C, c_tile);
    });
});
```
Mapping: outer metapipe streams A-row-strips through a PMU; inner metapipe streams B-column-tiles through a separate PMU; four PCUs are row-parallel on the MAT_MUL; another PMU buffers C; AGCU writes the result back. Nothing touches HBM between stages except the tile loads/stores chosen by the programmer.

### FlashAttention as a metapipeline
The QK^T → mask → softmax → dropout → xV chain is laid out linearly on the fabric (PMU→PCU→PMU→…). Tiles 0…15 stream through like a wavefront: while tile 0 is at "xV", tile 1 is at "dropout", tile 2 at "softmax", tile 3 at "mask", etc. Dataflow execution with token control avoids lock-based synchronization — the plumbing itself gates progression.

### Llama-3.1 8B: GPU vs. RDU
On GPU (TensorRT-LLM) a single decoder decomposes into ~10 kernels (K1 GEMM-Q/K/V, K2 QK^T, K3 softmax, K4 PV, K5 O-proj, K6 all-reduce, K7 RMSNorm, K8 gate/up/SiLU/mul, K9 down-proj, K10 add+reduce). Low kernel fusion ⇒ low data locality ⇒ high launch and synchronization overhead.

On SN40L the **entire decoder fuses into one kernel**. Aggressive fusion is possible because 520 MB of on-chip SRAM (5× H100's 100 MB) holds the intermediates, and dataflow removes the "every kernel boundary is a barrier" assumption. ~3 kernel calls per token (RDU) vs. ~800 on GPU — 100× fewer launches. The kernel loop fully overlaps HBM weight load with compute and even with all-reduce, so HBM bandwidth is saturated and inter-chip collectives don't consume HBM capacity.

## Examples / Worked Problems
- **Systolic `y = Wx` on a 4×4 grid**: trace `x0..x3` entering over successive cycles; at the end the bottom accumulators hold `y_j = Σ_i x_i·w_ij`.
- **Systolic `Y = WX`**: multiple X-columns stream staggered through the grid; need 4×32-bit accumulators per column to collect independent output columns.
- **Large matmul tiling**: `C(8×4096) = A(8×8)·B(8×4096)` with 4096 accumulators — stream B in 4-wide column stripes, keep A resident, accumulate columns of C in place.
- **TK matmul pipeline** — four-stage producer/consumer with INPUT_PIPE_STAGES=4 and 8 consumer warps; decrease producer registers to 40, increase consumer registers to 232.
- **RDU matmul metapipe** — nested `METAPIPE(M/MM)` over `METAPIPE(N/NN)` with `LOAD_TILE / MAT_MUL / BUFFER / STORE_TILE` mapped to AGCU → PMU → 4× PCU → PMU → AGCU.
- **FlashAttention on RDA** — sixteen tiles pipelined across QK^T → Mask → Softmax → Dropout → xV stages, token-controlled.
- **Llama decoder on RDU** — one fused kernel per decoder, ~100× fewer launches than GPU.

## Takeaways
1. **Specialization buys efficiency by removing generality**: fewer instructions, more arithmetic units, custom on-chip datapaths, lots of SRAM.
2. **Systolic arrays** amortize weight loads across many MACs by moving data between neighbor PEs instead of via a shared register file — the geometry *is* the schedule.
3. **Async is unavoidable at the top of the GPU curve**: B100 tensor cores are fed by TMA into TMEM, single threads launch MMAs, warps matter less. Programmer complexity explodes without DSLs.
4. **ThunderKittens** picks the 16×16 tile as a first-class type and builds producer/consumer patterns on top of TMA + MMA async, recovering readability while exposing the async primitives experts need.
5. **Dataflow architectures** (SambaNova SN40L) trade instruction-stream generality for spatial execution of entire graphs, collapsing ~800 GPU kernel launches into ~3 per token for LLM inference.
6. **Metapipelining** is the abstraction that makes dataflow programmable: nested parallel patterns become nested streaming pipelines with double-buffered FIFOs.
7. **Hardware lottery matters**: architectures and models co-evolve. Dense MM made transformers great; transformers made TPUs worth building; the cycle continues.

## Open Questions / Follow-ups
- How do compilers (XLA, Mosaic GPU, Cute-DSL, SambaNova's SambaFlow) actually lower a PyTorch graph onto each of these backends?
- When does the FP8/FP4/decompression-engine specialization on B100 break existing kernels, and how are numerics validated?
- How does RDA scale across chips — is chip-to-chip communication also modeled as a dataflow edge, or does it reintroduce synchronous barriers?
- Quantifying the "hardware lottery": which promising model families are under-explored because they don't map cleanly to dense MM?
- Energy-per-token comparisons for TPU vs. H100 vs. SN40L under identical workloads.

## Sources
- Slides: `slices/11_SpecializedHardwareProgramming.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Raw transcript: `notes/.raw/11_SpecializedHardwareProgramming.txt`.
- Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017 (TPU figures and perf/W data).
- Prabhakar, Zhang et al., "Plasticine: A Reconfigurable Architecture for Parallel Patterns," ISCA 2017.
- Ben Spector et al., ThunderKittens — embedded CUDA DSL for AI kernels.
- Sara Hooker, "The Hardware Lottery."
- NVIDIA H100/B100 PTX reference (`tcgen05.*`, `cp.async.bulk.tensor`, TMA/TMEM).
- SambaNova SN40L RDU technical brief.
