# Lecture 12 — Mapping AI Applications to the AI Datacenter

## Overview
Surveys how specialized hardware executes DNNs efficiently and how its programming model copes with extreme asynchrony. Three vehicles: Google TPU (systolic arrays for dense matmul), NVIDIA H100/B100 (asynchronous compute/memory with complex programming, simplified by DSLs like ThunderKittens), and SambaNova SN40L (reconfigurable dataflow with tiling, streaming, and *metapipelining*). Also covers HBM memory, DRAM fundamentals, and how model parallelism (TP/DP/PP/EP) maps onto datacenter clusters with compute–communication overlap.

## Key Concepts
- **Systolic array** — 2D grid of MACs that streams data through neighbors; ideal for dense matmul with minimal register/instruction overhead.
- **HBM (High-Bandwidth Memory)** — 3D-stacked DRAM with 1024-bit interface per stack via TSVs; more BW, higher power efficiency, smaller form factor than DDR.
- **Dataflow architecture** — no instructions; compute graph is *spatially* laid out on reconfigurable compute/memory units (PCU/PMU) connected by mesh switches. Extreme asynchrony, no fetch/decode overhead.
- **Metapipelining** — a "pipeline of pipelines": nested loops turned into streaming stages with double-buffered intermediates; enables kernel fusion beyond what GPU launch/sync costs permit.
- **Kernel fusion on RDA** — SN40L fuses an entire transformer decoder into one kernel (~3 kernel calls/token vs ~800 on GPU, 100× fewer launches).
- **Compute–communication overlap** — pipelined AllReduce overlapped with GEMM/weight load keeps HBM busy and sustains 70%+ utilization at 32 sockets.
- **Model parallelism primitives** — TP, DP → ReduceScatter+AllGather or AllReduce; PP → Send/Recv; EP → All-to-All.
- **DRAM mechanics** — PRE + RAS + CAS access; burst mode amortizes latency; banks pipeline requests; DIMMs interleave bytes across chips for parallel transfer.
- **Data-movement energy** — ~1 pJ int op, ~20 pJ FP op, ~26 pJ on-chip SRAM (64 b), ~1200 pJ DRAM (64 b). Move less data.

## Detailed Notes

### Memory primer — why HBM matters
A traditional CPU uses a 64-bit DRAM bus; a GPU uses HBM with a 1024-bit per-stack interface. HBM is enabled by **3D chip stacking**: multiple DRAM dies stacked and connected by **through-silicon vias (TSVs)**. A *logic layer* at the base acts as memory controller; a silicon *interposer* forms a short, wide interconnect between the stack and the processor. Benefits: more bandwidth, lower per-bit energy, small form factor.

GPU HBM progression:
- AMD Radeon Fury (2015): 4 HBM × 1024b = 4096b, 512 GB/s.
- NVIDIA P100 (2016): 4 HBM2 × 1024b, 720 GB/s, 16 GB.
- NVIDIA H100 (2022): 6 HBM3 × 1024b = 6144b, 3.2 TB/s, 80 GB.

### AI models as dataflow graphs
A transformer-style model is naturally a DAG of ops: Sample → GEMM1 → Pool → GEMM2 → Softmax → Sum, with weight inputs attached. This motivates mapping the graph *onto silicon* rather than serializing it through an instruction stream.

### Reconfigurable Dataflow Architecture (SambaNova SN40L RDU)
A chip is a mesh of:
- **PCU (Pattern Compute Unit)** — systolic + SIMD (e.g., 16×8 bf16).
- **PMU (Pattern Memory Unit)** — 0.5 MB SRAM with flexible address generation.
- **S (mesh switches)** — high-BW on-chip interconnect.
- **AGCU (Address Generator / Coalescing Unit)** — portal to HBM/IO.

SN40L specs: 1,040 PCUs+PMUs, 638 bf16 TFLOPS, 520 MB on-chip SRAM, 64 GB HBM, 1.5 TB DDR.

Design features and their purpose:
| Feature | Why |
|---|---|
| Tiled tensors (e.g. 16×16, 32×32) | Max TFLOPS on GEMM, low instruction overhead |
| Async compute | Overlap compute with memory access |
| Async memory access | Overlap compute with memory access |
| Async chip-to-chip comm | Overlap compute/memory/comm |
| Unit-to-unit comm | Fusion, pipelining, streaming dataflow |

No instructions ⇒ no fetch/decode; extreme asynchrony ⇒ no sequential execution.

### Dataflow programming with data-parallel patterns
Programs are composed from **MM, Map, Zip, Reduce, Gather, Scatter…**. A simplified softmax `x → Map(exp) → Reduce(+) → Zip(/) → o` is (1) tiled, (2) parallelized, (3) *metapipelined*, then place-&-routed and codegen'd to PCUs/PMUs/AGCUs spatially.

### Metapipelining
A **hierarchical coarse-grained pipeline** ("pipeline of pipelines") that exploits nested-loop parallelism:
- Each loop body is turned into pipe stages that run in parallel.
- Iterations overlap; intermediates live in **double buffers** between stages to tolerate imbalanced stage latencies.
- Buffers can change access patterns (e.g., transpose) — metapipelining works even where naive fusion would not.

Intuition (GDA slide): an outer `map(N)` wraps a sequence of inner loops (slice row → elementwise diff → outer product), each becoming a pipe stage (Pipe1…Pipe4) with AGCU/PMU/PCU resources. Multiple `r` iterations are in-flight simultaneously across stages.

### Worked example: Matmul metapipeline (SN40L DSL)
```cpp
auto format = DataFormat::kBF16;
int64_t M = args::M.getValue();
int64_t N = args::N.getValue();
int64_t K = args::K.getValue();

auto A = INPUT_REGION("A", (M, K), format);
auto B = INPUT_REGION("B", (K, N), format);
auto C = OUTPUT_REGION("C", (M, N), format);

auto MM = 256; // tile along M
auto NN = 64;  // tile along N

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
Mapping: `LOAD_TILE`→AGCU, tile buffer→PMU, `MAT_MUL`→a row of PCUs (`row_par=4`), result→PMU→AGCU→HBM. The outer `METAPIPE` streams A tiles; the inner `METAPIPE` streams B tiles, keeping the PCU row busy.

### FlashAttention as streaming dataflow
The attention kernel is tiled (Tile 0…15) and pipelined: `QK^T → Mask → Softmax → Dropout → xV → QK^T …` with each arrow implemented as a PCU/PMU pair. Tiles flow through the spatial pipeline in a skewed schedule (Tile 0 at the xV stage while Tile 4 is starting QK^T). Token-level control replaces lock-based synchronization ⇒ **Metapipeline = streaming dataflow**.

### Llama 3.1-8B: decoder fusion
A decoder block is: `RMSNorm → {Q,K,V} GEMM → QK matmul → Scale/Maskfill → Softmax → PV matmul → O GEMM → AllReduce → Add → RMSNorm → {Gate,Up} GEMM → SiLU·Mul → Down GEMM → Add → AllReduce`.

- **GPU (TensorRT-LLM)**: ≥10 kernels per decoder. FlashAttention fuses the attention sub-block, but GEMMs stay separate. Low fusion, low locality, high launch/sync overhead.
- **RDU**: the *entire* decoder fuses into one kernel (K0). Leverages SN40L's **520 MB** on-chip SRAM (5× H100's ~100 MB) to eliminate GBs of intermediate off-chip traffic.

Kernel-call counts per token: **~3 on RDU vs ~800 on GPU — 100× fewer**. The "Kernel Loop" overlaps weight load with compute so HBM is saturated and launch overhead is amortized across all decoders.

### Dataflow ⇒ overlap AllReduce with GEMM
In the decoder kernel, the AllReduce after the O/Down GEMMs is pipelined with the next compute stage so it "does not consume HBM capacity or bandwidth" — it flows chip-to-chip while compute continues.

### Datacenter scale: scale up vs scale out
DGX SuperPOD (A100 era): 1,024 GPUs = 128 DGX-A100 nodes × 8 GPUs, NVLINK 3.0 intra-node, Mellanox HDR 200 Gb/s InfiniBand fat-tree, separate compute/storage networks, adaptive routing + SharpV2 offload.

### Parallelism taxonomy
| Dim | What's split | Comm primitive |
|---|---|---|
| Data Parallel (DP) | batch | RS+AG or AllReduce |
| Tensor Parallel (TP) | hidden dim (within layer) | RS+AG or AllReduce |
| Pipeline Parallel (PP) | layers | Send/Recv |
| Expert Parallel (EP) | experts | All-to-All |
| Sequence / Context Parallel (SP/CP) | sequence dim | varies |

### Distributed matmul example (scale-up on RDUs)
`A[M×K] · B[K×N] = out[M×N]` with BS=16, M=24,576, K=131,072, N=8,192. Split K across S sockets ⇒ each does `[M×K/S]·[K/S×N]`, producing S partial `[M×N]` sums; combine via S-way **ReduceScatter**.

### Overlap: quantified on RDU
Benchmark: 844.44 TFLOPs.

| RDUs | Sys TFLOPS | Compute roofline (ms) | RS @100% links (ms) | Peak w/o overlap | Measured w/ overlap |
|---|---|---|---|---|---|
| 8  | 12,744 | 66.3 | 8.6  | 88.5% | 72% |
| 16 | 25,488 | 33.1 | 9.7  | 77%   | 75% |
| 32 | 50,976 | 16.5 | 15.0 | 52%   | 79% |

Without overlap, comm dominates as S grows. With compute–comm overlap, utilization *holds* across 32 sockets — actually *rising* at 32 because compute time falls faster than comm grows and overlap hides the rest.

### Fine-grained pipeline parallelism (training)
Naive PP leaves bubbles where later stages idle. Split each mini-batch into **micro-batches** and interleave forward/backward across them to keep all stages active — classic GPipe/1F1B schedule. Production 1T-parameter training uses (TP=8, PP=64, DP=6) on 3,072 GPUs at micro-batch 1, global batch 3,072, ~49% peak FLOPS.

### Memory primer — how DRAM really works
- 1 transistor + capacitor per bit; ~2 Kbits per row; read is *destructive* so row must be written back on precharge.
- Accessing a byte: **Precharge** (~10 ns) → **Row Activate (RAS)** (~10 ns) → **Column Select (CAS)** (~10 ns) → **transfer**. Repeated column accesses inside the open row skip PRE+RAS.
- **Burst mode** amortizes PRE/RAS over many consecutive column transfers.
- **Banks** let the controller pipeline: precharge bank 1 while bank 0 transfers.
- **DIMMs** interleave bytes across 8 chips so a 64-bit word comes out in parallel; a 64-byte cache line takes multiple bus beats.
- **Memory controller** is a scheduler; FR-FCFS (first-ready, first-come-first-serve) prioritizes open-row hits, reorders for throughput, coalesces small requests to enable bursts.
- DDR4-2400 example (i7-7700K): 64b × 1.2 GHz × 2 transfers/clk = 19.2 GB/s × 2 channels = 38.4 GB/s, ~13 ns CAS.

### Access latency is variable
- Best case: row is already open ⇒ CAS only.
- Worst case: PRE + RAS + CAS. Memory controller decides *when* to precharge (after every access vs on conflict).

### Energy of moving data (recap)
| Operation | Energy |
|---|---|
| 32-bit FP op | ~0.9 pJ |
| Local SRAM (on-chip) 32b | ~5 pJ |
| LPDDR load 32b | ~640 pJ |
| LPDDR load 64b | ~1200 pJ |

Reading 10 GB/s from DRAM ≈ 1.6 W — more than an entire mobile-GPU power budget. **Recompute > reload** is often the energy-optimal choice.

### How the memory bottleneck is being fought
- **Software**: schedule for locality (tiling, fusion, metapipelining).
- **Hardware**: smarter memory controllers, deeper caches, 3D stacking, wider buses, in-/near-memory compute research, compression.
- **Principles**: locate data near processor; move compute to data; trade computation for data movement.

## Examples / Worked Problems
- **Matmul metapipe** — DSL code above; shows tile sizes (MM=256, NN=64), `row_par=4`, and the spatial mapping A→AGCU/PMU, PCU row, C→PMU/AGCU.
- **FlashAttention metapipe** — 16 tiles streaming through `QK^T → Mask → Softmax → Dropout → xV` with token-based synchronization.
- **Llama 8B decoder fusion** — GPU ~800 kernel calls/token vs RDU ~3, using 520 MB of on-chip SRAM.
- **Distributed GEMM** — K-split across 4 RDUs + ReduceScatter to produce the final `[M×N]`.
- **Overlap quantified** — 32-RDU config sustains 79% measured utilization vs 52% theoretical without overlap.
- **DRAM cache-line fetch** — byte-interleaved across 8 DRAM chips on a DIMM; first 64 b arrives in parallel on beat 1, next 64 b on beat 2 (burst mode).

## Takeaways
1. AI hardware is built around big matmul (systolic) + large on-chip SRAM + async compute/memory/comm.
2. H100-style async is powerful but hard to program; DSLs (ThunderKittens) bridge the gap.
3. Dataflow/RDA (SN40L) trades instruction generality for spatial layout — yields aggressive kernel fusion (decoder-level) and huge locality wins.
4. Metapipelining converts nested loops into streaming pipelines of pipelines, making fusion tractable even when naive fusion fails.
5. Kernel launch + HBM traffic is the real cost at scale — 100× fewer kernel calls on RDU translates directly into HBM BW freed for weights.
6. Compute–communication overlap is essential to keep utilization high as clusters grow; GPUs compensate with larger interconnects.
7. HBM and 3D stacking exist because moving data is the dominant energy cost; DRAM physics (PRE/RAS/CAS, banks, bursts) still governs achievable bandwidth.
8. The full parallelism zoo (TP/DP/PP/EP/SP/CP) maps to a small set of collectives (AR, RS+AG, Send/Recv, All-to-All); picking the mix trades memory, comm, and pipeline bubbles.

## Open Questions / Follow-ups
- How does the RDA compiler actually do place-and-route and scheduling of a graph onto PCUs/PMUs?
- ThunderKittens DSL details for H100 async programming — referenced but not elaborated.
- In-memory / near-memory compute research directions (HBM4 custom logic die hints at this).
- Hardware-accelerated compression for further reducing off-chip traffic.
- How PP bubble size formally depends on micro-batch count and stage imbalance.

## Sources
- Slides: `slices/12_AI_DatacenterMapping.pdf` (Stanford CS149, Fall 2025).
- Referenced: Prabhakar, Zhang et al., *Plasticine: A Reconfigurable Architecture for Parallel Patterns*, ISCA 2017; SambaNova SN40L; NVIDIA H100/B100; Dally (NVIDIA) & Olson (ARM) energy numbers; Han, ICLR 2016; AMD HBM imagery; https://creativestrategies.com/gpu-networking-basics/.
