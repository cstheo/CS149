# Lecture 2 — A Modern Multi-Core Processor (Part I)

## Overview
Computer architecture from a software engineer's perspective. The lecture introduces the three mechanisms modern throughput processors use to run programs fast: **multi-core** (more instruction streams), **SIMD** (more ALUs per stream), and **hardware multi-threading** (hide memory latency by interleaving streams). Worked through with a running `sin(x)` Taylor-series example, progressively retargeted from scalar C → pthreads → `forall` → AVX intrinsics → a multi-core SIMD multi-threaded chip. Closes with CPU and GPU case studies (Kaby Lake, V100) and the "three requirements" for using such a machine efficiently.

## Key Concepts
- **Instruction stream** — a sequential list of instructions produced by a compiler; runs on one hardware thread.
- **Superscalar** — within one stream, issue multiple independent instructions per clock via out-of-order logic. ILP is automatic, discovered by hardware.
- **Multi-core (Idea #1)** — spend transistors on more (simpler) cores rather than making one core smarter. Each core runs its own stream → **thread-level parallelism**.
- **SIMD (Idea #2)** — one fetch/decode drives many ALUs with the same instruction on different data. Amortizes control cost.
- **Explicit vs. implicit SIMD** — CPU compiler emits vector instructions (AVX/NEON); GPU compiles scalar, hardware groups N program instances to share a SIMD unit.
- **Coherent vs. divergent execution** — coherence (same instructions across lanes) is required for SIMD efficiency; divergence causes lanes to be masked off, wasting throughput.
- **Hardware multi-threading (Idea #3)** — each core holds multiple execution contexts; when one thread stalls, another runs. Hides, doesn't reduce, latency.
- **Interleaved vs. simultaneous multi-threading (SMT)** — pick a thread per clock vs. issue from multiple threads the same clock. Intel Hyper-Threading is SMT-2.
- **Throughput trade-off** — multi-threading may slow individual threads (context pressure) but raises aggregate work done per unit time.

## Detailed Notes

### Review: processor, program, memory
- A program is a list of instructions. A simple processor has a **fetch/decode**, an **ALU**, and an **execution context** (registers).
- One instruction per clock in the simple model: `ld r0, addr[r1]`, `mul r1, r0, r0`, etc.
- Memory is a byte-addressed array. `ld R0 ← mem[R2]` reads N bytes at the address in R2.
- **Stall** — processor blocked waiting for a dependency (typically a load). DRAM latency is hundreds of cycles.
- **Caches** — on-chip copies of recently used memory, organized in **cache lines** (e.g., 4 B in the toy, 64 B in real CPUs). Assume LRU for the course. Hierarchy: L1 (≈32 KB, 4 cyc) → L2 (≈256 KB, 12) → L3 (≈8–20 MB, 38) → DRAM (~248 cyc on Kaby Lake).
- **Data movement dominates energy**: int op ~1 pJ, FP op ~20 pJ, 64 b from SRAM ~26 pJ, 64 b from LPDDR ~1200 pJ. 10 GB/s from DRAM ≈ 1.6 W — more than a mobile GPU's entire budget.

### The running example: `sinx` via Taylor series
```c
void sinx(int N, int terms, float* x, float* y) {
    for (int i=0; i<N; i++) {
        float value = x[i];
        float numer = x[i] * x[i] * x[i];
        int denom = 6;        // 3!
        int sign  = -1;
        for (int j=1; j<=terms; j++) {
            value += sign * numer / denom;
            numer *= x[i] * x[i];
            denom *= (2*j+2) * (2*j+3);
            sign  *= -1;
        }
        y[i] = value;
    }
}
```
Compiled as a scalar stream it runs one element at a time: `ld, mul, mul, …, st`. Every loop iteration is independent of the others — but the C code doesn't tell the machine that.

### Idea #1 — Multi-core
Pre-multicore chips spent most transistors on making a single stream faster: wide OoO, deeper caches, branch predictors, prefetchers. Post-2005, diminishing returns from wider issue and the power wall flipped the strategy: use the transistor budget for **more cores**, each simpler.

- Two simpler cores may each be ~25% slower on a single thread, but 2 × 0.75 = 1.5× aggregate.
- A serial program sees no speedup — it just runs on the slower core. Parallelism must be **expressed**.

#### Expressing parallelism
- **C++ threads / pthreads**: spawn a worker, split the array range.
```c
my_thread = std::thread(my_thread_func, &args);   // N/2 elements
sinx(N - args.N, terms, x + args.N, y + args.N);  // other N/2 on main
my_thread.join();
```
- **Data-parallel `forall`** (Kayvon's fictitious language): the programmer declares iteration independence; the compiler can generate both threads *and* vector instructions.
```text
forall (int i from 0 to N) { ... y[i] = value; }
```

CPU examples: Intel Comet Lake i9 (10 cores), Apple A15 (2 big + 4 small), Apple M1 (4 big + 4 small + GPU + NPU), RTX 4090 (144 SMs). Heterogeneous cores and specialization are the norm.

### Idea #2 — SIMD
One fetch/decode broadcasts the same instruction to N ALUs; each ALU has its own register lane and operates on different data. Same silicon-per-ALU, far less silicon-per-instruction.

#### Explicit SIMD on x86 (AVX2, 8-wide float)
```c
#include <immintrin.h>
for (int i=0; i<N; i+=8) {
    __m256 origx = _mm256_load_ps(&x[i]);
    __m256 value = origx;
    __m256 numer = _mm256_mul_ps(origx, _mm256_mul_ps(origx, origx));
    __m256 denom = _mm256_broadcast_ss(&three_fact);
    int sign = -1;
    for (int j=1; j<=terms; j++) {
        __m256 tmp = _mm256_div_ps(
            _mm256_mul_ps(_mm256_set1_ps(sign), numer), denom);
        value = _mm256_add_ps(value, tmp);
        numer = _mm256_mul_ps(numer, _mm256_mul_ps(origx, origx));
        denom = _mm256_mul_ps(denom,
                   _mm256_broadcast_ss((2*j+2)*(2*j+3)));
        sign *= -1;
    }
    _mm256_store_ps(&y[i], value);
}
```
Compiles to `vloadps / vmulps / vstoreps` — one instruction does 8 lanes.

SIMD widths in current ISAs:
- Intel AVX2: 256 b (8×f32 or 4×f64)
- Intel AVX-512: 512 b (16×f32)
- ARM Neon: 128 b (4×f32)

Sources of vector code: hand-written intrinsics, parallel language semantics (`forall`, ISPC), or auto-vectorizing compilers.

#### Divergence under control flow
For `if (t > 0.0) { A } else { B }` inside a `forall`, the SIMD unit must execute both sides, **masking** lanes whose predicate was false. Visual: 8 lanes, mask pattern `TTFTFFFF` — during the `if`-body, lanes that were `F` do no useful work; during the `else`-body, the `T` lanes are wasted.

- Worst case SIMD utilization under an 8-wide `if/else`: 1/8 (only one lane active on each side).
- Breakout answer sketch: any predicate that isolates a single lane during the expensive path yields 1/W utilization.
- **Terminology**: "instruction stream coherence" = same instruction across many lanes. Required for SIMD efficiency. *Not* required across cores (each core has its own fetch/decode).

#### Implicit SIMD on GPUs ("SIMT")
Compiler emits scalar instructions. Hardware groups N program instances (a **warp**, 32 on NVIDIA) and, each clock, finds a common instruction to issue across the group's SIMD ALUs. Divergent lanes are masked. GPU SIMD widths run 8–32; divergence can drop you to 1/32 of peak.

### Idea #3 — Hardware multi-threading
Caches + prefetching help when access patterns are predictable. When the next address depends on a just-computed value (`int y = A[some_function()]`), a prefetcher can't help and the core stalls ~100s of cycles.

**Fix**: store several execution contexts on the core; when the running thread stalls, switch to another that's runnable. Latency is *hidden*, not shortened.

#### Utilization exercise (single-issue, scalar core)
Program = 3 arithmetic ops, then 1 load with 12-cycle latency. Cycle budget per thread = 3 (work) + 12 (stall) = 15.

| # hw threads | Utilization |
|---|---|
| 1 | 3/15 = 20% |
| 2 | 6/15 = 40% |
| 5 | 15/15 = 100% |
| 6+ | still 100% (no further gain) |

If the program instead does 6 math + 12-cycle load → only **3** threads needed. **Higher arithmetic-per-load ratio ⇒ fewer threads needed to hide memory.**

#### Context storage is finite
The on-chip context area trades off:
- **Many small contexts** (GPU: 64 warps/SM ≈ 2048 lanes of state) → high latency hiding, tiny per-thread working set.
- **Few large contexts** (CPU: 2 per core for Hyper-Threading) → lots of registers and cache per thread, less hiding ability.

#### Interleaved vs. simultaneous multi-threading
- **Interleaved (temporal)**: each clock, pick *one* thread and run its instruction on the ALUs.
- **SMT**: each clock, pick instructions from *multiple* threads. Intel Hyper-Threading = SMT-2.

### Putting it together
A single core can simultaneously be superscalar + SIMD + multi-threaded. The lecture builds the stack incrementally in its bonus review:
1. Simple core — 1 scalar instr/clk.
2. Superscalar — up to 2 independent scalar instrs/clk from one stream.
3. SIMD — 1 vector (e.g., 8-wide) instr/clk.
4. Heterogeneous superscalar — 1 scalar + 1 vector/clk.
5. Multi-threaded — 2+ contexts; one instr/clk from one chosen thread.
6. Multi-threaded superscalar — 2 instrs/clk, possibly from different threads (SMT).
7. Multi-core version of any of the above.

#### Case study: Intel Kaby Lake core (i7-7700K)
- 4 cores, each 2-way multi-threaded (Hyper-Threading).
- Each core: up to **4 independent scalar** instructions + **3 8-wide AVX2 vector** instructions per clock (up to 2 vector MUL + 3 vector ADD).
- Peak: 4 cores × 8-wide SIMD × 3 AVX units × 4.2 GHz ≈ **400 GFLOPs**.

#### Case study: NVIDIA V100 SM
- 80 SMs on the chip.
- Per SM: **64 warp contexts** × 32 lanes = 2048 concurrent data items. 16-wide SIMD physical ALUs run a 32-wide warp in 2 clocks.
- 256 KB registers per SM, divided among warps.
- Tensor cores + fp64 units alongside fp32/int.
- Whole chip: 80 × 2048 = **163,840** items concurrent for full latency hiding. 16 TFLOPs @ ~250 W.

#### Kayvon's fictitious reference chip
16 cores × 8 SIMD lanes × 4 threads/core = **512 independent work items** needed to fully load the machine.

## Examples / Worked Problems
- **Scalar → AVX rewrite of `sinx`**: same algorithm, processes 8 elements per loop iteration with `_mm256_*` intrinsics.
- **Cache example recap** (from Lecture 1): 0x0,0x1,…,0xF,0x0 on a 2-line × 4 B LRU cache → cold misses on line starts, then a **capacity miss** on 0x0 because it was evicted while streaming.
- **Utilization arithmetic**: with 3 math + 12-cycle load, the core is busy 3/15 per thread; multiply by #threads until it saturates at 100%. Changing ratio to 6 math shifts break-even from 5 threads to 3.
- **Divergence worst case**: an 8-wide SIMD unit running `if (lane==0) expensive() else trivial()` drops to ~1/8 peak on the expensive path.
- **Thought experiment on OS scheduling**: 2 application threads onto a 2-core × 2-context chip — OS picks one context per core so both cores' fetch/decode are active (maximize parallelism, not packing). For 5 threads onto 4 contexts, three threads time-share one context pair.

## Takeaways
1. Modern throughput rests on three orthogonal levers: **multi-core**, **SIMD**, **multi-threading**. A well-tuned program exploits all three.
2. The program has to *expose* parallelism (threads, `forall`, intrinsics). Hardware can't invent parallelism the source code doesn't admit.
3. SIMD efficiency demands **coherent control flow** across lanes. Branchy code on wide SIMD (especially 32-wide GPUs) can waste most of the machine.
4. Multi-threading **hides** memory latency rather than reducing it — trading single-thread speed for throughput. You need enough runnable work to fill the latency gap; high arithmetic intensity shrinks that requirement.
5. To utilize a modern chip: (a) enough parallel work for all cores × lanes, (b) coherent work within SIMD groups, (c) more parallel work than ALUs so the scheduler can hide stalls.

## Open Questions / Follow-ups
- Cache internals (direct-mapped, set-associative, replacement beyond LRU) — deferred; Kayvon suggests external reading.
- How ISPC expresses gang/SIMD parallelism — Lecture 3 (`03_multicore2-ispc`).
- GPU SIMT details: warp scheduling, divergence reconvergence, memory hierarchy — Lecture 7 (`07_gpuarch`).
- Operating-system-level scheduling of threads onto hardware contexts — alluded to in the closing thought experiment; covered later alongside consistency and synchronization.
- Precise semantics of "respects program order" for a superscalar/SMT schedule — still open from Lecture 1.

## Sources
- Slides: `slices/02_basicarch.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Course site: https://gfxcourses.stanford.edu/cs149
- Energy numbers: Bill Dally (NVIDIA), Tom Olson (ARM). Hardware case studies: Intel Kaby Lake, NVIDIA V100.
