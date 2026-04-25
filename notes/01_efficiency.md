# Lecture 1 — Why Parallelism? Why Efficiency?

## Overview
Opens CS149 by motivating parallel computing: single-thread performance has stopped scaling, so speedups now come from using many processing elements *and* specialized hardware. Introduces the three course themes (writing scalable parallel programs, understanding parallel hardware, thinking about efficiency), and ends with a primer on instructions, memory, and caches so later lectures can reason about data movement costs.

## Key Concepts
- **Parallel computer** — collection of processing elements that cooperate to solve problems quickly. Goal: performance *and* efficiency.
- **Speedup(P)** — `T(1) / T(P)`. Limited in practice by communication, load imbalance, and the inherently serial fraction of the work.
- **Fast ≠ efficient** — 2× speedup on a 10-core machine is fast but wasteful; efficiency asks how well the hardware is being used.
- **Instruction-level parallelism (ILP)** — independent instructions in one stream that a superscalar processor can issue in parallel. Returns diminish beyond ~4-wide issue.
- **Power wall** — dynamic power ∝ capacitance · V² · frequency; voltage must rise with frequency, so clocking up hits a thermal/power ceiling. Ended frequency scaling ~2005.
- **Memory hierarchy** — registers → L1 → L2 → L3 → DRAM, each larger and slower. On Kaby Lake: L1 ≈ 4 cyc, L2 ≈ 12, L3 ≈ 38, DRAM ≈ 248.
- **Locality** — spatial (neighbors in same cache line) and temporal (reuse same address) are what make caches work.
- **Data movement dominates energy** — moving a 64-bit word from DRAM costs ~1200 pJ vs ~1 pJ for an integer op. Efficiency ⇒ minimize data movement.

## Detailed Notes

### Course themes
1. **Writing parallel programs that scale** — decompose work, assign to processors, manage communication/synchronization. Learn the abstractions (e.g., ISPC, CUDA) and why they're shaped the way they are.
2. **How parallel hardware works** — because machine characteristics (communication cost, cache sizes, SIMD width) determine what programs are fast.
3. **Efficiency** — both programmer (use machine capabilities) and hardware-designer (which capabilities to include vs. silicon/power cost) perspectives.

### In-class demos (why parallelism is hard)
- Demo 1 (partial sums): communication cost limits speedup; shorter communication paths help.
- Demo 2 (4 processors): load imbalance leaves some workers idle; better distribution helps.
- Demo 3 (massively parallel): when communication ≫ computation, parallelism can't help much.

### Historical context
- Until ~2005, single-thread performance doubled every ~18 months from two sources: superscalar ILP and frequency scaling.
- Developers got free speedups by waiting. That era ended.

### What a processor does
A program is a list of instructions. A simple processor has:
- **Fetch/decode** — pick the next instruction.
- **ALU** (execution unit) — perform it.
- **Execution context** — registers holding program state.

Executing `add R0 ← R0, R1`: fetch → read R0,R1 → add in ALU → write result to R0. One instruction per clock in the simple model.

### ILP example
For `a = x*x + y*y + z*z`:
```
1  mul R0, R0, R0   ; x*x
2  mul R1, R1, R1   ; y*y
3  mul R2, R2, R2   ; z*z
4  add R0, R0, R1   ; depends on 1,2
5  add R3, R0, R2   ; depends on 3,4
```
Dependency DAG: the three muls are independent (ILP=3), then two serial adds (ILP=1 each). With 2 execution units, 5 instructions finish in 3 steps; with 3 units, still 3 steps (adds are the critical path).

### Superscalar execution
Hardware (or compiler) detects independent instructions and issues them to multiple execution units per clock. Out-of-order control logic reorders around stalls while preserving the appearance of program-order semantics (observable results match sequential execution).

**Diminishing returns**: most ILP is captured by a 4-wide issue processor (Culler & Singh, Johnson 1991). Going wider spends area/power for little gain — motivates explicit parallelism (threads, SIMD) instead.

### The power wall
- Dynamic power ∝ C·V²·f, plus static leakage even when idle.
- Higher f requires higher V, and V² amplifies the cost. Power turns into heat; the chip must be clocked down when hot.
- TDP examples: Apple M1 laptop 13 W; Core i9-10900K 95 W; RTX 4090 450 W; phone SoC 0.5–2 W.
- Herb Sutter's "The Free Lunch Is Over" (2005) captured the inflection point: transistor density kept climbing, but frequency, ILP, and power per core flattened.

### Multi-core, GPUs, specialization
- Intel Comet Lake i9 (2020): 10 cores on one die.
- AMD Threadripper 3990X: 64 cores across four 8-core chiplets.
- NVIDIA AD102 (RTX 4090): 76 B transistors, 18,432 fp32 lanes, 144 SMs.
- Frontier supercomputer: 606,208 AMD CPU cores + 37,888 Radeon GPUs, 21 MW.
- Mobile: Apple A15 packs 2 big + 4 small CPU cores, 5-core GPU, Neural Engine (NPU), image/video codecs, motion sensor DSP — each block specialized for power-efficient work.
- Datacenter: Google TPUs, AWS Trainium, Cerebras Wafer-Scale, GraphCore IPU, SambaNova — all specialized for DNN training/inference.

Assignment 1 preview: on a quad-core Intel CPU with AVX SIMD + hyper-threading, a well-parallelized program can be ~32–40× faster than single-threaded `-O3` C.

### Memory & caches
- Memory is a byte-addressed array. `ld R0 ← mem[R2]` reads N bytes starting at the address in R2 into R0.
- **Latency** — time from request to data arrival (~hundreds of cycles for DRAM).
- **Stall** — processor can't advance because a later instruction depends on an unfinished load.
- **Cache** — on-chip copy of a subset of memory values. Transparent to correctness, crucial to performance. Operates at **cache line** granularity (e.g., 64 B), with replacement policies like LRU.
- **Cache example (2 lines × 4 bytes, LRU)**:
  - Accessing 0x0,0x1,0x2,0x3 → one cold miss then 3 hits (spatial locality inside the line).
  - Accessing 0x0,0x1 again → hits (temporal locality).
  - Working set larger than the cache causes capacity misses that evict still-useful data.
- Hierarchy: registers → L1 (~32 KB) → L2 (~256 KB) → L3 (~20 MB) → DRAM (~GBs). Smaller = closer = faster.

### Data access costs (Kaby Lake, 4 GHz)
| Level | Latency (cycles) |
|---|---|
| L1 | 4 |
| L2 | 12 |
| L3 | 38 |
| DRAM | ~248 |

### Energy of data movement (ballpark, Dally/Olson)
| Operation | Energy |
|---|---|
| Integer op | ~1 pJ |
| FP op | ~20 pJ |
| 64 b from on-chip SRAM (~1 mm away) | ~26 pJ |
| 64 b from LPDDR DRAM | ~1200 pJ |

Reading 10 GB/s from DRAM burns ~1.6 W — larger than the entire mobile GPU power budget. **Exploiting locality is the whole game.**

## Examples / Worked Problems
- Scheduling `a = x*x + y*y + z*z` onto 2 or 3 execution units with dependency constraints (see ILP section).
- Cache example 1: access 0x0,0x1,0x2,0x3,0x2,0x1,0x4,0x1 on a 2-line × 4-byte LRU cache → two cold misses, rest hits. Shows spatial + temporal locality.
- Cache example 2: streaming access 0x0 through 0xF then back to 0x0 evicts 0x0 before reuse → a **capacity miss**. Shows locality of reference isn't automatic — you have to fit in the cache.

## Takeaways
1. Single-threaded performance barely improves; speed now comes from parallelism + specialization.
2. Parallelism is hard because of communication cost, synchronization, and load imbalance — not just Amdahl's law.
3. Fast is not efficient: always ask "how much of the machine am I actually using?"
4. The memory hierarchy and cache locality usually matter more than arithmetic throughput — data movement is the dominant cost in time *and* energy.
5. Modern systems are heterogeneous: CPU cores, SIMD lanes, GPUs, NPUs, media/sensor engines. Efficiency means matching work to the right unit.

## Open Questions / Follow-ups
- Direct-mapped vs set-associative caches, replacement policies — deferred; pointed to as external reading.
- AVX SIMD and hyper-threading details — lecture 02 (`02_basicarch`).
- How ISPC expresses SIMD parallelism — lecture 03 (`03_multicore2-ispc`).
- Quantifying "respects program order" for a parallel schedule — teased as a class question, formal answer later.

## Sources
- Slides: `slices/01_efficiency.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Course site: https://gfxcourses.stanford.edu/cs149
- Referenced: Herb Sutter, "The Free Lunch Is Over" (Dr. Dobb's, 2005); Olukotun & Hammond, ACM Queue 2005; Culler & Singh (Johnson 1991 data); Dally (NVIDIA), Olson (ARM) energy numbers.
