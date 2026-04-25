# Lecture 10 — Hardware Specialization

## Overview
Examines why general-purpose CPUs and GPUs are energy-inefficient and how specialization (DSPs, FPGAs, ASICs, tensor cores, systolic arrays, reconfigurable dataflow architectures) delivers orders-of-magnitude better performance per watt. Frames modern AI acceleration as the dominant driver of specialized silicon design and introduces the "ideal AI accelerator" checklist: tiled tensors, asynchronous compute/memory/communication, and compute-unit-to-compute-unit dataflow. Walks through NVIDIA H100/B100, Google TPU with systolic arrays, and SambaNova-style reconfigurable dataflow as three representative points on the efficiency/programmability spectrum.

## Key Concepts
- **Energy-constrained computing** — mobile (battery, no fan) and datacenter (scale, cooling) are both power-limited; AI growth is now bounded by energy.
- **Performance = Ops/sec = (Ops/Joule) × (Joules/sec)** — with fixed Watts, efficiency (Ops/Joule) is the only lever. Specialization is the path to better Ops/Joule.
- **Instruction overhead** — on general CPUs, fetch/decode/rename/register-file access dominates energy; functional units consume a small fraction even with SIMD.
- **DSP** — programmable but simpler control; uses VLIW/SIMD to amortize instruction overhead (e.g., Qualcomm Hexagon does 29 RISC ops/cycle).
- **FPGA** — middle ground: array of programmable LUTs + flip-flops + hard blocks (DSP/SRAM/CPUs), programmed via HDL (Verilog).
- **ASIC** — fixed-function silicon; ~100–1000× perf/watt over CPU, but non-programmable and costs $10–100M to design/verify.
- **Tensor core** — SIMD-style matrix multiply-accumulate (e.g., 16×16×16 fp16→fp32) that amortizes instruction overhead across ~256 MACs per issue.
- **Systolic array** — 2D grid of PEs passing data wavefront-style to neighbors; locality is spatial + temporal, control is distributed, no register file pressure.
- **Reconfigurable dataflow (RDU)** — no instructions; map the AI graph spatially onto a tiled PMU (Pattern Memory Unit) / PCU (Pattern Compute Unit) fabric, fusing kernels.
- **Hardware lottery** — ideas win because they fit the available hardware, not because they are universally best (transformers + MM-heavy TPUs).

## Detailed Notes

### Why general-purpose processors are inefficient
Executing one instruction on a modern CPU touches: icache + address translation, decode + uop cache, dependency/hazard checks, scheduler, register-file SRAM reads, bypass network into the ALU, the arithmetic op itself, writeback into the register file, and retirement. The arithmetic is tiny; the choreography is large.

Hameed et al. (ISCA 2010) on H.264: even with SIMD, functional units (FU) consume a small slice of energy — pipeline registers (Pip), register fetch (RF), data cache (D$), control (Ctrl) and instruction fetch (IF) dominate. Across motion estimation, subpixel ME, intra-prediction/DCT/quantization, and arithmetic coding, the story repeats.

Chung et al. (MICRO 2010) on FFT: an ASIC delivers the performance of one CPU core at ~1/1000 the area and ~1/100 the power; GPU cores are ~5–7× more area-efficient than CPU cores. Going from CPU → GPU → FPGA → ASIC monotonically improves throughput and energy per op.

### Efficiency vs. programmability spectrum
From easiest to program to most efficient:
1. **Energy-optimized CPU** — easiest to program, baseline.
2. **Throughput-oriented processor (GPU)** — ~10× more efficient when work is compute-bound and data-parallel.
3. **Programmable DSP** — VLIW/SIMD, Qualcomm Hexagon class.
4. **Domain-specific accelerator** (e.g., Google TPU) — ~20× efficient, limited domain, programmed via DSLs.
5. **FPGA / reconfigurable logic** — ~50× (still uncertain); hard to program, DSL research active.
6. **ASIC** — 100–1000× more efficient, not programmable, $10–100M NRE (video codecs, audio, camera RAW, neural nets).

Rules of thumb vs high-quality C on CPU: GPU ≈ 10× perf/watt (compute-bound, data-parallel); ASIC ≈ 100–1000× (compute-bound, not floating-point-heavy).

### Specialized processors today
- **Qualcomm Hexagon DSP** — in Snapdragon SoCs, used for modem/audio/image. VLIW issues multiple dissimilar ops per clock (vs. SIMD which issues the same op over a vector).
- **Anton (DE Shaw Research)** — molecular dynamics supercomputer: 512 ASICs for particle-particle interactions + throughput subsystem for FFTs + custom low-latency N-body network. Anton 3 (2025) is ~20× faster than a contemporary GPU on MD.
- **Google TPU** — ASIC for deep learning, arithmetic is ~30% of chip area, tiny control footprint. Key instructions: `read_host_memory`, `write_host_memory`, `read_weights`, `matrix_multiply`/`convolve`, `activate`.
- **FPGAs** — array of LUTs (Xilinx Virtex-7 has 6-input, 1-output LUT6, treated as a 64-entry table); chain 8 LUT6s to build a 40-input AND (delay = 3). Modern FPGAs devote large area to hard blocks (SRAM, DSP multipliers, ARM/RISC-V cores). Available on AWS EC2 F1/F2.

### Data movement dominates energy
Ballpark numbers (Dally/Olson):
- Integer op: ~1 pJ
- FP op: ~20 pJ
- 64 b from on-chip SRAM (~1 mm away): ~26 pJ
- 64 b from LPDDR DRAM: ~1200 pJ

Design rule: always reduce data movement.

### Amortizing instruction overhead with complex instructions
Estimated instruction-stream overhead as percentage of useful arithmetic:
- Half-precision FMA: 2000%
- Half-precision DP4 (vec4 dot): 500%
- Half-precision 4×4 MMA (matrix mul-accumulate): 27%

Conclusion: one MMA instruction that launches hundreds of MACs amortizes decode/RF access across many ops — this is the core reason tensor cores exist.

### Numerical formats
- **FP32**: sign + 8-bit exponent + 23-bit mantissa.
- **BF16**: sign + 8-bit exponent + 7-bit mantissa — same range as FP32, lower precision.
- **FP8 E4M3**: sign + 4-bit exp + 3-bit mantissa, range 0–448.
- **FP8 E5M2**: sign + 5-bit exp + 2-bit mantissa, range 0–57344.
Lower-precision formats cut MAC energy and SRAM/bandwidth cost roughly linearly with bitwidth squared (for multipliers).

### NVIDIA GPU tensor cores: A100 → H100 → B100
**A100 SM**: 64 fp32 ALUs, 32 int32 ALUs, 4 tensor cores (each executes 8×4 × 4×8 matrix MMA per instruction, A/B in fp16, accumulator in fp32). GA100 has 108 SMs → 6,912 fp32 MACs, 432 tensor cores, 19.5 fp32 TFLOPs, 312 fp16/fp32 tensor TFLOPs.

**H100**: 4th-gen tensor core, Tensor Memory Accelerator (TMA), CUDA thread-block clusters (up to 16 blocks guaranteed to co-reside across SMs and run concurrently), HBM3 ≤ 80 GB, TSMC 4 nm, 80 B transistors, 144 SMs → 989 fp16 TFLOPs on tensor cores vs 134 fp16 / 67 fp32 TFLOPs on SIMD. Hierarchy:

| CUDA | Compute | Memory |
|---|---|---|
| Grid | GPU | 80 GB HBM / 50 MB L2 |
| Cluster | CPC | 256 KB SMEM per SM |
| Thread block | SM | 256 KB SMEM |
| Threads | SIMD lanes | 1 KB RF/thread, 64 KB/SM partition |

Each H100 SM sub-core has a warp selector (1 warp/clock), its own 64 KB RF, and issues into SIMD fp32/int/fp64 units plus a tensor core (16×16×16 fp16→fp32). Four sub-cores share a 256 KB SMEM/L1 and a TMA.

**TMA** — single-thread issues an async bulk copy of a tensor tile between global and shared memory via a copy descriptor; HW generates addresses, signals an mbarrier on completion (`cuda::memcpy_async`).

**B100** tensor cores: data lives in SMEM and a new TMEM; single threads issue MMAs — **no more warps** for tensor core work. Programming flow: `tcgen05.alloc` for TMEM + descriptors → prefetch tiles with `cp.async.bulk.tensor` + mbarrier → `tcgen05.mma` batches with `tcgen05.commit` → `tcgen05.fence` to order/retire. ("Not your father's CUDA.")

Over the V100→A100→H100→B100 generations, the fraction of GPU TFLOPs coming from tensor cores rose from ~50% → 89% → 94% → 96% → 98%. All the TFLOPs are in the tensor cores.

### Ideal AI accelerator (checklist)
AI models are **dataflow graphs** (e.g., Sample → GEMM1 → Pool → GEMM2 → Softmax → Sum with weight inputs). The accelerator should:

| Feature | Why |
|---|---|
| Tiled tensors (e.g., 16×16, 32×32) | Max TFLOPS on GEMM; low instruction overhead |
| Asynchronous compute | Overlap compute and memory access |
| Asynchronous memory access | Overlap compute and memory access |
| Asynchronous chip-to-chip comm | Overlap compute, memory, and communication |
| Compute-unit-to-compute-unit comm | Kernel fusion, pipelining, streaming dataflow |

NVIDIA GPUs satisfy the first four (mma_async, TMA+TMEM) but only partially the last (thread-block clusters are a step toward CU-to-CU comm).

### Google TPU and the systolic array
TPU v1: ~30% of die is arithmetic (a 256×256 systolic MAC array), tiny control. For `y = Wx` with weights stationary in PEs:

- Cycle 0: load weights into PEs w_ij.
- Cycle 1: x0 enters top-left PE, computes `x0*w00`.
- Cycle 2: x1 enters row 0, x0*w00 passes down; PE(0,1) computes `x0*w10` while PE(1,0) accumulates `x0*w00 + x1*w01`.
- Continue: each cycle a new x enters from the left edge, partial sums propagate down to column accumulators; after `n` cycles the first output column is ready; after `2n` cycles the full `y` is in accumulators.

For matrix-matrix `Y = WX`, inject multiple x-columns staggered across cycles, each producing a column of `Y` in its own 32-bit accumulator row. Example: A(8×8) × B(8×4096) = C(8×4096) needs 4096 accumulators per row and streams B through while W stays resident.

**SIMD vs. systolic**:

| Feature | SIMD | Systolic Array |
|---|---|---|
| Dataflow | Control-driven (instructions) | Data-driven (wavefront) |
| Locality / reuse | Limited | Temporal + spatial |
| Communication | Global (register/memory) | Local (neighbor PEs) |
| Control | Centralized | Distributed |
| Perf/mm², perf/W | Medium | Very high |

TPU perf/watt vs CPU+GPU baseline (Jouppi et al. 2017): massive wins on the geo/weighted-mean across Google workloads; gap widens further if you exclude host-machine cost (incremental vs total).

### Reconfigurable Dataflow Architectures (SambaNova Plasticine)
Map the AI dataflow graph *spatially* onto a tiled fabric of:
- **PMU** — Pattern Memory Units (on-chip SRAM tiles).
- **PCU** — Pattern Compute Units (tile compute: GEMM + map/filter/reduce patterns).
- **Switches (S)** — configurable interconnect.

No instructions ⇒ no fetch/decode overhead. Extreme asynchrony, fully pipelined. Enables **dataflow kernel fusion** — e.g., FlashAttention (QK^T → Mask → Softmax → Dropout → xV) executed as a meta-pipeline: tile 0 flows through QK^T, then while it's in Mask, tile 1 starts QK^T, etc. A full PMU/PCU mesh holds Q, K^T, V, Weights, and intermediate outputs of every stage, with results streaming neighbor-to-neighbor instead of roundtripping through HBM.

### The hardware lottery
TPUs are exceptional at dense MM (operational intensity ∝ n). This encouraged researchers to design **transformer** models (also MM-heavy), which then dominated — which then justified specializing HW *even more* for MM. Sara Hooker: an idea can win because it suits the hardware, not because it's fundamentally superior. Worth remembering when choosing research directions.

### Summary of specialized AI hardware
- Many arithmetic units (systolic arrays, tensor cores).
- Configurable datapaths moving data directly between PEs at multiple granularities.
- Large on-chip storage (SMEM/TMEM/PMU) for fast intermediates.
- Schedule computation by laying it out *spatially* on the chip rather than temporally through a register file.

## Examples / Worked Problems
- **Instruction-overhead amortization**: fp16 FMA → 2000% overhead, DP4 → 500%, 4×4 MMA → 27%. Doubling the "work per instruction" via MMA cuts overhead to ~1/70× of a scalar FMA.
- **Systolic y = Wx (4×4)**: traced cycle-by-cycle, showing a wavefront where PE(0,0) computes `x0*w00` at cycle 1 and PE(3,3) finishes `x0*w30 + ... + x3*w33` around cycle 7. Weights stay put; x flows right; partial sums flow down.
- **Matrix multiply C(4×4096) = A(4×8) × B(8×4096) on a 4×4 systolic array**: stream B through while holding A rows; one 32-bit accumulator per output column (4096 accumulators total).
- **H100 SM datapath accounting**: 4 sub-cores × (32-wide fp32 MUL-ADD per clock + tensor core 16×16×16 MMA) vs. 1 warp fetch/decode per sub-core — ratio explains why tensor cores contribute ~96% of the SM's peak TFLOPs.

## Takeaways
1. Specialization is the only remaining lever for better Ops/Joule — CPUs spend most of their energy on instruction overhead, not arithmetic.
2. The gain hierarchy is roughly CPU (1×) → GPU (~10×) → DSP (~20×) → FPGA (~50×) → ASIC (100–1000×), trading programmability for efficiency.
3. Data movement is the dominant energy cost; the accelerator-design goal is to keep intermediates on-chip and move them between compute units directly, not via register files or DRAM.
4. Tensor cores are the MMA-amortization trick inside an otherwise general GPU; systolic arrays take it further by eliminating register-file traffic entirely; RDUs take it further still by eliminating instructions.
5. The "ideal AI accelerator" is characterized by tiled tensors + async compute + async memory + async chip-to-chip + CU-to-CU streaming. GPUs check 4 of 5; RDUs attack the last one.
6. Beware the hardware lottery: what's "best" today is partly an artifact of what silicon exists.

## Open Questions / Follow-ups
- How do DSLs (CUTLASS/Cute-DSL, Triton, ThunderKittens, Mosaic GPU) abstract the tile + async + tensor-core programming model? What's lost vs. hand-tuned PTX?
- What does the B100 programming model ( `tcgen05.*` + TMEM) look like end-to-end for a real kernel?
- Can FPGAs become mainstream for AI inference if HLS/DSL ergonomics improve enough?
- For RDUs: how does the compiler auto-place a transformer graph onto a finite PMU/PCU mesh? What happens when the graph is bigger than the chip?
- How to measure/compare "ideal accelerator" compliance for non-NVIDIA chips (Trainium 2, TPUv5, Cerebras, Apple Neural Engine)?

## Sources
- Slides: `slices/10_Specialized.pdf` (Stanford CS149, Fall 2025).
- Hameed et al., "Understanding sources of inefficiency in general-purpose chips," ISCA 2010 (H.264 energy breakdown).
- Chung et al., "Single-Chip Heterogeneous Computing: Does the Future Include Custom Logic, FPGAs, and GPGPUs?" MICRO 2010 (FFT area/power study).
- Jouppi et al., "In-Datacenter Performance Analysis of a Tensor Processing Unit," ISCA 2017.
- Prabhakar, Zhang, et al., "Plasticine: A Reconfigurable Architecture for Parallel Patterns," ISCA 2017.
- Sara Hooker, "The Hardware Lottery," 2020.
- Bill Dally (NVIDIA) and Tom Olson (ARM) — energy ballpark numbers.
- Herb Sutter, "The Free Lunch Is Over" (context for the power wall).
