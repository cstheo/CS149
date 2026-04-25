# Lecture 3 — Multi-Core Architecture Part II (Latency/Bandwidth) + Parallel Programming Abstractions

## Overview
Two halves. The first finishes the multi-core throughput-hardware story by separating **latency** from **bandwidth** and showing why modern throughput processors are almost always **bandwidth-limited**: the arithmetic is cheap, feeding the ALUs is hard. The second half pivots to **programming abstractions**, introducing **ISPC** and the **SPMD** model as a running example to hammer home the course mantra: distinguish the *semantics* of an abstraction from its *implementation*.

## Key Concepts
- **Latency** — time for one operation to complete (e.g., one car SF→Stanford in 0.5 hr; one DRAM load in ~hundreds of cycles).
- **Bandwidth (throughput)** — rate of completed operations per unit time (cars/hour, bytes/sec, instructions/clock).
- **Ways to grow throughput** — go faster (raise clock), add lanes (parallel resources), or pack the pipe (pipeline / keep it full).
- **Pipelining** — overlap stages of independent work items so every stage is busy each clock; raises throughput without reducing per-item latency. Classic 4-stage CPU pipeline: IF → D → EX → WB.
- **Bottleneck = min bandwidth** along the path (two connected pipes: 100 L/s into 50 L/s → 50 L/s).
- **Arithmetic intensity** — math ops per byte moved from memory. Low intensity ⇒ bandwidth-bound.
- **Bandwidth-bound steady state** — once memory is 100% busy, adding more outstanding loads or lowering latency cannot speed things up; only raising memory bandwidth or reducing bytes per op can.
- **HBM** — High-Bandwidth Memory stacked near the GPU die (V100 HBM2: 900 GB/s over a 4096-bit bus).
- **Abstraction vs. implementation** — semantics tell you *what the answer is*; implementation tells you *how (and in what order) it is computed on real hardware*.
- **SPMD** — Single Program, Multiple Data. One function, many logical instances with different `programIndex`.
- **SIMD** — Single Instruction, Multiple Data. One instruction operates on a vector of lanes. ISPC's SPMD *abstraction* is compiled down to a SIMD *implementation*.
- **`uniform`** — ISPC type modifier: one value shared by all gang instances (a scalar in the generated SIMD code). Purely an optimization/clarity hint; not required for correctness.
- **Gang, `programCount`, `programIndex`** — a gang is the set of concurrent instances launched by one ISPC call; `programCount` is gang size (typically SIMD width), `programIndex` is the lane id.
- **`foreach`** — raises the level of abstraction: programmer declares a parallel iteration space; the compiler decides assignment to lanes (interleaved, blocked, dynamic, …).
- **Cross-instance primitives** — `reduce_add`, `reduce_min`, `broadcast`, `rotate`, `shift` — the only sanctioned way for gang instances to communicate.

## Detailed Notes

### Thought experiment: element-wise A × B → C
Three memory ops (2 loads + 1 store = 12 bytes) per single FP multiply. On a V100 (5120 fp32 ALUs at 1.6 GHz) this would need ~98 TB/s to keep the ALUs busy, against 900 GB/s of HBM2 → <1% ALU utilization. An 8-core Xeon E5v4 at 76 GB/s is ~3% efficient. Even at <1% efficiency the GPU still crushes the CPU, because 1% of 98 TB/s worth of ALUs is still a lot. Moral: trivially vectorizable code is **not** a good fit for throughput hardware unless you also feed it.

### Latency vs. bandwidth — the SF→Stanford highway
Cars at 100 km/hr, distance 50 km: latency = 0.5 hr per car. With one car on the road at a time, throughput = 2 cars/hr.

Three orthogonal ways to improve throughput:
1. **Drive faster** (200 km/hr) → latency drops *and* throughput doubles.
2. **More lanes** (parallel resources) → throughput scales with lane count; latency unchanged.
3. **Use the existing road better** (space cars every 1 km instead of one-at-a-time) → a pipelined/streaming pattern that multiplies throughput by how many items fit concurrently.

### Laundry pipeline
Wash 45 min + Dry 60 min + Fold 15 min = 2 hr latency. Option A: duplicate washers/dryers/students → throughput 2× for 2× resources. Option B: pipeline — while load *i* is drying, start load *i+1* in the washer. Latency still 2 hr per load, but steady-state throughput is set by the **slowest stage** (the dryer, 60 min), ≈ 1 load/hr with one of each appliance.

### Connected pipes (bottleneck)
Pipe 1 can flow 100 L/s, pipe 2 can flow 50 L/s. Connect them in series → system throughput = min(100, 50) = 50 L/s. Bandwidth of a pipeline equals the bandwidth of its slowest segment.

### Memory bandwidth walk-through
Consider a thread looping on three dependent instructions:
```
1. X = load 64 bytes
2. Y = add X + X
3. Z = add X + Y
```
Hardware assumptions: 1 math op / clock, loads issue in parallel with math, memory delivers 8 bytes / clock, up to 3 outstanding loads. One 64-byte load therefore takes 8 clocks to transfer, and only 3 can be in flight at once.

As the program runs, the core issues loads and math. When 3 loads are already in flight and the next iteration needs another, the core **stalls**. In steady state, the memory bus is transferring data 100% of the time; the core is idle whenever math exceeds the rate at which bytes arrive. **Key insight:** once memory is saturated, shortening memory *latency* or allowing more outstanding requests does nothing — only raising memory *bandwidth* (or issuing fewer/smaller loads) helps.

### Bandwidth is the critical resource
Performant programs on throughput hardware must:
- **Reuse loaded data** (temporal locality).
- **Share data across threads** (inter-thread cooperation — a.k.a. blocking/tiling).
- **Prefer recomputation over reload** — arithmetic is essentially free compared to DRAM traffic.
- **Access memory infrequently** relative to the amount of work done per byte (grow arithmetic intensity).

### Instruction pipelines
Students often ask how a core "does a multiply in one clock." It doesn't — *throughput* is 1/clock, *latency* is several clocks. A 4-stage IF/D/EX/WB pipeline issues one instruction per clock once full, but each individual instruction takes 4 clocks to complete. Real CPU pipelines can be ~20 stages; correctness for back-to-back dependent instructions is handled by forwarding, interlocks, and hazard logic.

### Abstraction vs. implementation (the recurring theme)
- **Semantics**: given a program and the meaning of its operations, what answer will it compute?
- **Implementation (scheduling)**: in what parallel order, on which execution unit / thread / SIMD lane, is each operation performed?

Conflating the two is the single biggest source of confusion in this course.

### ISPC: Intel SPMD Program Compiler
- Open-source, Matt Pharr et al. — see *The Story of ISPC*.
- You write one function; calling it from C/C++ spawns a **gang** of `programCount` concurrent instances. Each instance has its own copies of non-`uniform` locals and its own `programIndex`. When the call returns, all instances have finished.
- **Implementation:** the compiler emits SIMD instructions (AVX2, NEON, …). Gang size is the hardware SIMD width (or a small multiple). Conditional control flow becomes masked SIMD.
- So a single ISPC call runs on one thread of one core. Multi-core is handled by a second abstraction called **ISPC tasks** (covered in assignment 1).

### `sinx` in ISPC (interleaved assignment)
```c
// ISPC: sinx.ispc
export void ispc_sinx(
    uniform int N,
    uniform int terms,
    uniform float* x,
    uniform float* result)
{
    // assumes N % programCount == 0
    for (uniform int i = 0; i < N; i += programCount) {
        int idx = i + programIndex;
        float value = x[idx];
        float numer = x[idx] * x[idx] * x[idx];
        uniform int denom = 6;   // 3!
        uniform int sign  = -1;
        for (uniform int j = 1; j <= terms; j++) {
            value += sign * numer / denom;
            numer *= x[idx] * x[idx];
            denom *= (2*j + 2) * (2*j + 3);
            sign  *= -1;
        }
        result[idx] = value;
    }
}
```
Gang of 8 covers indices 0..7 in the first iteration, 8..15 in the next, etc. Because the 8 `x[idx]` values are contiguous in memory, the compiler lowers the load to a single packed vector load (`vmovaps` / `_mm256_load_ps`).

### `sinx` blocked version
Replace the stride with a per-instance contiguous block:
```c
uniform int count = N / programCount;
int start = programIndex * count;
for (uniform int i = 0; i < count; i++) {
    int idx = start + i;
    ...
}
```
Each instance owns a contiguous chunk. But now on iteration `i`, the 8 gang lanes touch 8 *non-contiguous* addresses (separated by `count`), so the load must be a **gather** (`vgatherdps` / `_mm256_i32gather_ps`) — significantly more expensive than a packed load. Same semantics, worse implementation. This is the abstraction-vs-implementation lesson in one slide: two schedules that compute the same answer, one matches the hardware and one does not.

### `foreach`: letting ISPC choose the schedule
```c
export void ispc_sinx(uniform int N, uniform int terms,
                      uniform float* x, uniform float* result) {
    foreach (i = 0 ... N) {
        float value = x[i];
        // ... same body as before, using i instead of idx ...
        result[i] = value;
    }
}
```
`foreach` declares a parallel iteration space for the **whole gang** (not each instance). The compiler is free to pick interleaved, blocked, or dynamic assignment. Conceptual implementations:
```c
// 1. Single-lane serial (baseline)
if (programIndex == 0)
    for (int i = 0; i < N; i++) { /* body */ }

// 2. Interleaved across instances (the usual SIMD-friendly choice)
for (uniform int loop_i = 0; loop_i < N; loop_i += programCount) {
    int i = loop_i + programIndex;
    /* body */
}

// 3. Blocked per instance
uniform int count = N / programCount;
int start = programIndex * count;
for (uniform int loop_i = 0; loop_i < count; loop_i++) {
    int i = start + loop_i;
    /* body */
}

// 4. Dynamic via an atomic work counter
uniform int nextIter = 0;
int i = atomic_add_local(&nextIter, 1);
while (i < N) {
    /* body */
    i = atomic_add_local(&nextIter, 1);
}
```

### Gotchas once `programIndex` / raw loops leak in
- **Undefined output when iterations collide.** `shift_negative` writes `y[i-1] = x[i]` for some `i` — different gang instances can write to the same address in the same step; ISPC makes no guarantee about who wins.
- **Aliased writes via indexing.** `absolute_repeat` writes `y[2*i]` and `y[2*i+1]`; fine because each iteration owns a unique 2-element slot.
- **Reductions are not free.** Two wrong ways to sum an array:
  - `float sum = 0; foreach(...) sum += x[i]; return sum;` — `sum` is varying (one per lane), can't return as a single `float`. Compile error.
  - `uniform float sum = 0; foreach(...) sum += x[i];` — `x[i]` is varying, can't assign to a uniform. Compile error.
  - Correct idiom: accumulate a per-lane partial, then reduce across lanes.
```c
export uniform float sum_array(uniform int N, uniform float* x) {
    float partial = 0.0f;              // varying: one per lane
    foreach (i = 0 ... N)
        partial += x[i];
    return reduce_add(partial);        // cross-lane reduction → uniform
}
```
This mirrors exactly what you'd write by hand with AVX intrinsics: accumulate into a `__m256` with `_mm256_add_ps`, then sum the 8 lanes at the end.

### Cross-instance primitives
```c
uniform int64 reduce_add(int32 x);
uniform int32 reduce_min(int32 a);
int32 broadcast(int32 value, uniform int index);
int32 rotate(int32 value, uniform int offset);   // i -> (i+offset) % programCount
int32 shift (int32 value, uniform int offset);   // shifted, zero-fill
```

### Parallel-reduce tree in 3 steps (gang of 8)
```c
// Product of x[0..7] in lg(8) = 3 steps.
export void vec8product(uniform float* x, uniform float* result) {
    float v1 = x[programIndex];
    float v2 = shift(v1, 1);
    if (programIndex % 2 == 0) v1 = v1 * v2;

    v2 = shift(v1, 2);
    if (programIndex % 4 == 0) v1 = v1 * v2;

    v2 = shift(v1, 4);
    if (programIndex % 8 == 0) *result = v1 * v2;
}
```
Demonstrates that SPMD + cross-lane primitives is enough to express advanced cooperative patterns, but also that the code is *specific* to `programCount == 8`.

### ISPC's design tension
- `foreach` + `uniform` is nearly a sequential-feeling abstraction: "independently, for each element…".
- But `programIndex` / `programCount` are *also* exposed. That makes ISPC low-level and expressive — at the cost of letting you write programs with undefined output or programs that are only correct for a particular gang size.

### Higher-level alternatives (teaser)
- Hide `programIndex`/`programCount`; force all parallelism through `foreach`; everything outside `foreach` must be `uniform`.
- Forbid array indexing entirely; express computation as `map(f, collection)` over a data structure. This is the NumPy / PyTorch style — upcoming lectures.

## Examples / Worked Problems
- **Bandwidth bound on V100.** A × B → C needs 12 bytes / fp32 MUL; 5120 ALUs × 1.6 GHz × 12 B ≈ 98 TB/s demand vs 900 GB/s supply → ~0.9% ALU utilization. Eight-core Xeon E5v4 with 76 GB/s gets ~3%.
- **Steady-state pipeline.** With 1 math/clock, 8 B/clock from memory, 3 outstanding 64-B loads: once in steady state the core stalls whenever it has consumed the current load before the next arrives. Memory is 100% busy; adding a 4th outstanding load or shortening latency does nothing.
- **Interleaved vs blocked in ISPC.** Same semantic result; interleaved lowers to a packed load, blocked lowers to a gather. Matches what you hand-write with AVX.
- **Reduction.** Per-lane `partial` + `reduce_add(partial)` — the only correct ISPC pattern and a direct analog of an AVX horizontal sum.
- **Tree product `vec8product`.** lg(8) steps using `shift` + masked multiplies; correct only when `programCount == 8`.

## Takeaways
1. Throughput comes from **parallel resources + pipelining**; latency and bandwidth are different knobs.
2. On throughput-oriented hardware, performance is almost always **bandwidth-bound**. Optimize for bytes moved, not FLOPs performed.
3. Once the memory bus is saturated, fewer/larger/reused loads is the only lever left; more outstanding requests or lower latency won't help.
4. Always separate **abstraction semantics** from **implementation**. Two programs with identical semantics can have wildly different implementations (packed load vs gather).
5. ISPC = **SPMD** abstraction compiled to **SIMD** implementation. `programCount`/`programIndex`/`uniform` expose the seam; `foreach` hides it.
6. Communication between lanes is *only* via the cross-instance primitives (`reduce_add`, `broadcast`, `shift`, `rotate`). Sharing ordinary variables across lanes is a type error or undefined behavior.

## Open Questions / Follow-ups
- **ISPC tasks** for multi-core execution — read in assignment 1.
- How the ISPC compiler lowers divergent control flow to masked SIMD — assignment 1 has you do it by hand.
- Formalizing what "respects program order" means for a parallel schedule — still teased from lecture 1.
- Higher-level data-parallel models (map over collections, NumPy / PyTorch style) — upcoming lectures.
- *The Story of ISPC* (Matt Pharr, 2018) — assigned background reading.

## Sources
- Slides: `slices/03_multicore2-ispc.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Course site: https://gfxcourses.stanford.edu/cs149
- ISPC: https://ispc.github.io/
- Matt Pharr, *The Story of ISPC*: https://pharr.org/matt/blog/2018/04/30/ispc-all.html
- AVX intrinsics referenced: `_mm256_load_ps`, `_mm256_add_ps`, `_mm256_i32gather_ps`, `_mm256_broadcast_ss` (Intel Intrinsics Guide).
