# Lecture 6 — Performance Optimization II: Locality, Communication, and Contention

## Overview
Second performance-optimization lecture. Generalizes "communication" to any data movement in the extended memory hierarchy — core↔cache, core↔memory, core↔remote node — and presents techniques to reduce its cost: exploit locality, pick assignments with higher arithmetic intensity, eliminate artifactual transfers, hide latency with asynchrony, and dodge contention. Contrasts the shared-address-space model with explicit message passing (using the grid solver as a running example), then closes with a performance-analysis toolkit: high-watermark experiments, the roofline model, and pitfalls of fixed-problem-size scaling studies.

## Key Concepts
- **Shared address space** — any core can load/store any address; requires hardware (ring bus, crossbar, coherence) that gets expensive at high core counts.
- **NUMA** — latency/bandwidth to a memory location depends on which core asks. True across sockets, and effectively true even within a socket because L3 slices sit at different ring hops from each core.
- **Message passing** — threads have *private* address spaces and exchange data only via explicit `send`/`recv`. Maps naturally to clusters; synchronization and data motion are the same primitive.
- **Synchronous vs non-blocking send/recv** — blocking returns on ack; asynchronous returns immediately with a handle and a later `checksend`/`checkrecv`. Non-blocking is what lets you overlap compute with communication.
- **Ghost cells** — replicated border data "owned" by a neighbor thread; communicated once per iteration in the solver.
- **Arithmetic intensity** — `compute / communication` (bytes or instructions). Higher is better; `1/AI` is the communication-to-computation ratio.
- **Inherent vs artifactual communication** — inherent = what the algorithm fundamentally requires given the assignment; artifactual = everything else (cache-line granularity, capacity misses, write-allocate overhead).
- **Contention** — a shared resource (memory bank, lock, queue, office-hours TA) has finite throughput; bursts of requests serialize and inflate observed latency.
- **Roofline model** — plot max achievable throughput vs arithmetic intensity. Diagonal region = bandwidth-bound (slope = peak BW); horizontal ceiling = compute-bound (peak FLOPs). Code lives under the roof.

## Detailed Notes

### The shared address space, under the hood
A single `ld X` can touch L1 → L2 → L3 → DRAM, and on a multi-core chip L3 is physically distributed. Intel's Kaby Lake (example): 4 cores + integrated GPU + 4×2 MB L3 slices + system agent, connected by four rings (request, snoop, ack, 32-byte data). Each L3 slice is attached to the ring twice; theoretical peak bandwidth cores→L3 at 3.4 GHz is ~435 GB/s *if each core hits its local slice*. Sun's Niagara 2 uses a crossbar instead — every core talks to every L2 bank directly, but the crossbar (CCX) occupies roughly one core's worth of die area, which is why crossbars don't scale.

Multi-socket systems are explicitly NUMA: each socket has its own memory controller and DRAM; accessing the other socket's memory crosses an inter-socket link. Footnote that matters: even a single socket is mildly NUMA, because a core's own L3 slice is closer than the others.

Shared address space is ergonomic (just read/write), but the HW cost grows with core count — one reason high-core-count chips are expensive.

### Message passing
Each thread has a private address space. Communication is `send(buf, dst, tag)` / `recv(buf, src, tag)`. No hardware shared-memory needed — commodity nodes on Infiniband can form a supercomputer. The programming cost is that data replication becomes the programmer's job.

#### Grid solver, message-passing version
Partition the N×N grid into row bands, one band per thread. Each thread allocates `rows_per_thread+2` rows locally — the extra two are **ghost rows** owned by the up/down neighbor. Before each iteration, exchange ghost rows, then do the normal 5-point stencil on local rows.

```c
// before computation: swap ghost rows with neighbors
if (tid != 0)
   send(&localA[1,0],                sizeof(float)*(N+2), tid-1, MSG_ID_ROW);
if (tid != get_num_threads()-1)
   send(&localA[rows_per_thread,0],  sizeof(float)*(N+2), tid+1, MSG_ID_ROW);
if (tid != 0)
   recv(&localA[0,0],                sizeof(float)*(N+2), tid-1, MSG_ID_ROW);
if (tid != get_num_threads()-1)
   recv(&localA[rows_per_thread+1,0],sizeof(float)*(N+2), tid+1, MSG_ID_ROW);

// local stencil (same as shared-memory version, but on local indices)
for (int i=1; i<rows_per_thread+1; i++)
  for (int j=1; j<n+1; j++) {
    float prev = localA[i,j];
    localA[i,j] = 0.2f*(localA[i-1,j]+localA[i,j]+localA[i+1,j]
                        +localA[i,j-1]+localA[i,j+1]);
    my_diff += fabs(localA[i,j]-prev);
  }

// reduction: all non-root send diff to thread 0, get back 'done' flag
```

Thread 0 sums the partial diffs, tests convergence, and broadcasts `done`. Synchronization rides on the same `send`/`recv` — mutexes, barriers, and flags are all just messages.

#### Deadlock with synchronous send/recv
A **synchronous** `send` blocks until the receiver has the data; `recv` blocks until data arrives. If every thread calls `send` first, every `send` blocks waiting for a matching `recv` that no one has posted yet → deadlock.

Fix: break symmetry. Even tids `sendDown(); recvDown(); sendUp(); recvUp();` while odd tids `recvUp(); sendUp(); recvDown(); sendDown();`. Drawn on a time axis it looks like a staircase: T0 sends while T1 is ready to receive, then T1 sends while T2 is ready to receive, etc.

Better fix in practice: **non-blocking** send/recv. `send` returns a handle immediately; the buffer must not be modified until `checksend(h)` reports completion. Lets the thread overlap communication with unrelated computation — the single biggest performance reason to prefer async.

### Communication is a generalized memory hierarchy
Think of every access, from one processor's point of view, as walking outward through levels: registers → local L1 → local L2 → sibling-core L2 → shared L3 → local DRAM → remote DRAM (1 hop) → remote DRAM (N hops). Each level is higher latency, lower bandwidth, larger capacity. "Managing locality" means trying to satisfy requests at the nearest level — the *same* discipline regardless of whether the next level is a cache or a cluster node.

### Bandwidth-bound execution recap
From lecture 3: if the code does ~2 math ops per cache line loaded, steady-state throughput is set by memory bandwidth, not the core. Memory bus is 100% utilized; cores stall in red. Key observations to convince yourself of:
- Raising memory *latency* does **not** slow steady-state throughput as long as enough requests are in flight to keep the bus busy.
- Raising bus *bandwidth* does — the bandwidth is the actual bottleneck.
- Raising the math/load ratio eventually makes the code compute-bound and the stalls disappear.

### Arithmetic intensity and assignment
Same algorithm, different assignment, very different AI:

Grid solver, N×N, P processors:
- **1D interleaved (row stripes, round-robin)**: each processor owns N/P rows scattered across the grid; *every* owned row needs ghost rows from another processor. `compute/comm ≈ (N/P)/2` per row → AI ∝ N/P.
- **1D blocked (contiguous row band)**: `compute ≈ N²/P`, `comm ≈ 2N` (two boundary rows) → AI ∝ N/P.
- **2D blocked (√P × √P tiles)**: `compute ≈ N²/P`, `comm ≈ N/√P` (four boundary strips) → AI ∝ N/√P.

2D blocked wins asymptotically: communication grows sublinearly in P because the assignment respects 2D locality.

### Artifactual communication (caches make you pay for more than you asked for)
- **Minimum transfer granularity**: load one 4-byte float → the whole 64-byte cache line comes in (16× inflation if you don't use the rest).
- **Write-allocate**: store 16 consecutive floats → line is *read* from memory, then entirely overwritten, then written back. 2× overhead that a write-no-allocate policy would avoid.
- **Capacity misses**: working set exceeds cache, same data gets refetched. This is what motivates blocking.

#### Example: row-major grid traversal, capacity-limited
Assume row-major layout, 4-element cache lines, 24-element (6-line) cache. While updating row `r`, the stencil needs row `r-1` and row `r+1`. After finishing row `r`, rows `r-1,r,r+1` sit in cache. By the time the sweep returns to column 0 of row `r+1` the row-above cells that were touched in row `r`'s sweep have been evicted by row `r`'s own data. Result: **3 cache-line loads for every 4 output elements** — two "already-computed" rows keep being refetched.

### Techniques to reduce communication

**Blocking (loop tiling).** Walk the grid in B×B tiles instead of full rows. Each tile's working set fits in cache, so rows get reused before eviction. With a tile sized to the cache, the traversal becomes **2 loads per 6 outputs** in the example — a ~2.25× bandwidth reduction for the same result.

**Loop fusion.** Compute `E = D + (A+B)*C` as one pass, not three library calls:
```c
// array-library style: AI = 1/3 (2 loads + 1 store per math op, three times)
add(n, A, B, tmp1);       // 2L,1S / 1 op
mul(n, tmp1, C, tmp2);
add(n, tmp2, D, E);

// fused loop: 4 loads + 1 store per 3 math ops → AI = 3/5
for (int i=0; i<n; i++)
  E[i] = D[i] + (A[i] + B[i]) * C[i];
```
Modularity loses to intensity — the temporaries never have to round-trip through memory. This is exactly why frameworks like NumPy need JIT fusion (numba/jax/torch.compile) to close the gap.

**Co-locate sharing.** Schedule threads that touch the same data on the same core (or the same NUMA node) so that "communication" stays in L1/L2. Reduces inherent communication by changing what counts as "between processors".

### Contention
Kayvon's office-hours parable: 5-minute walk + 5-minute answer = 10 min cost if no queue; but if 5 students converge at 3:00 pm, the 5th waits 20 min behind 4 predecessors and pays 23 min. Staggering appointments (3:00 and 4:30) brings everyone back to 10 min.

Two structural responses:
- **Tree-structured communication** over flat communication when updating a shared variable: serial depth grows as log P instead of P, but latency in the uncontended case is higher. Pick based on expected traffic.
- **Distributed work queues with work stealing** (Cilk's model): each worker owns a deque; no lock in the common case; when empty, steal from a random victim. You only pay synchronization cost when the thread would have been idle anyway.

### Summary of techniques
- Reduce **per-message overhead**: fewer, larger messages; coalesce; avoid tiny transfers.
- Reduce **latency**: restructure for locality (programmer), improve interconnect (HW).
- Reduce **contention**: replicate contended data, fine-grained locks, staggered access.
- Overlap **comm with compute**: async messages, prefetching, multi-threading, OOO. Needs more concurrency than execution units to hide latency.

### Performance analysis workflow
1. **Write the simplest parallel version first. Measure.** Don't optimize blindly.
2. **Decide what's limiting you.** Compute? Bandwidth? Latency? Sync? Each has different fixes.
3. **Establish high-watermarks** — modify the program to isolate each axis:
   - Add non-memory math. Time grows linearly with op count ⇒ compute-bound.
   - Remove math but keep loads. Time barely drops ⇒ bandwidth-bound.
   - Replace every access with `A[0]`. Upper bound on how much locality improvements could buy.
   - Remove all atomics/locks. Upper bound on how much sync removal could buy.
4. **Roofline model.** Plot achievable GFLOPs vs arithmetic intensity on log-log axes. Diagonal slope = peak memory BW; flat ceiling = peak compute. A kernel's point shows which regime it's in and how far from the roof. Drawing extra rooflines for "with SIMD" vs "scalar" reveals where micro-architectural features buy you headroom.
5. **Use hardware counters / profilers**: Intel PCM, VTune, PAPI, oprofile. OS-level "% CPU" graphs are useless for optimization; counters give IPC, L2/L3 hit rates, bytes from MC, etc.

### Scaling and problem size
Speedup denominator matters. Dividing by "parallel code on 1 core" is easy to look good with; the honest baseline is the *best sequential* implementation, which may use a different algorithm.

Two pitfalls of **fixed-size** speedup studies on the grid solver (SGI Origin 2000, 32 procs):
- **Too small (258×258):** with 32 procs ~310 cells per proc; AI drops as N/√P shrinks; communication dominates, curve flattens or even slows down. This doesn't mean the machine is bad — it means the workload is too small to fill it.
- **Too large, but fits one node:** per-processor chunks may suddenly fit in cache past some P, giving **super-linear speedup**. Same thing happens when a problem that thrashes to disk on one machine fits in RAM on a cluster — the "speedup" is really a memory-hierarchy effect.

Takeaway: scale problem size with machine size (weak scaling) to reflect realistic use. Strong scaling is a legitimate metric but don't draw conclusions from strong scaling a toy problem on a big machine.

## Examples / Worked Problems
- **Grid solver, message passing** — send/recv ghost rows each iteration; tree-reduce the diff through thread 0; fix synchronous-send deadlock by alternating send/recv order on even/odd threads.
- **Assignment AI comparison** — 1D blocked vs 1D interleaved vs 2D blocked gives AI of N/P, N/P, and N/√P respectively. 2D blocked's sublinear comm growth is the motivation for block partitioning.
- **Row-major traversal capacity miss** — 24-entry cache, 4-wide lines; sweeping rows evicts the previous row before the next row needs it, costing 3 line loads per 4 outputs. Blocking fixes this to 2/6.
- **Loop fusion** — `E = D + (A+B)*C` goes from AI=1/3 to AI=3/5 by eliminating temporary round-trips.
- **Office-hours contention** — 5 students arriving at 3:00 pm queue up; staggering to appointment slots restores 10 min/student. Generalizes to lock contention, memory-bank hotspots, and shared queues.

## Takeaways
1. Communication is **one concept** spanning the whole hierarchy: register↔cache, core↔DRAM, node↔node. Same locality rules apply everywhere.
2. The assignment you choose determines arithmetic intensity; 2D tiling beats 1D row stripes because communication scales as √P instead of P.
3. Artifactual communication (cache-line granularity, write-allocate, capacity misses) is often the majority of traffic — blocking and fusion are the standard cures.
4. Prefer **non-blocking** communication so latency can be hidden behind useful work; blocking send/recv can deadlock and always serializes.
5. Measure first. Use high-watermark experiments and roofline analysis to locate the actual bottleneck before optimizing.
6. Beware fixed-problem-size scaling studies: too small ⇒ misleading flatness; too large on small machines ⇒ misleading super-linear speedups.

## Open Questions / Follow-ups
- Cache coherence protocols that make the shared-address-space abstraction work across cores — later lecture.
- Exactly how MPI implements non-blocking send/recv (eager vs rendezvous protocols, intermediate buffering) — deferred.
- Work-stealing scheduler internals (Cilk's THE protocol, random victim choice) — appears again in the task-parallel lectures.
- Hardware counters: which ones actually mean what on modern x86 (IPC quirks, memory-controller counters) — see VTune / PCM docs.
- Roofline extensions: multiple bandwidth roofs (DRAM vs L3 vs L2), FMA vs scalar ceilings.

## Sources
- Slides: `slides/06_progperf2.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Course site: https://gfxcourses.stanford.edu/cs149
- Referenced: Culler, Singh & Gupta, *Parallel Computer Architecture: A Hardware/Software Approach* (grid-solver pseudocode and scaling figures on SGI Origin 2000); Williams, Waterman, Patterson, "Roofline: An Insightful Visual Performance Model" (CACM 2009); Intel Performance Counter Monitor (PCM); Intel VTune; PAPI; oprofile.
