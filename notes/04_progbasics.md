# Lecture 4 — Parallelizing Code: The Programming Thought Process

## Overview
Finishes the ISPC story from Lecture 3 by separating the SPMD *abstraction* programmers reason about from the SIMD *implementation* the compiler emits, then walks through the general recipe for building a parallel program — decompose, assign, orchestrate, map — under Amdahl's Law. The second half is a case study: parallelizing a 2D Gauss-Seidel grid solver, first by finding (and avoiding) dependencies via red-black reordering, then by expressing it in two programming models (data-parallel and shared address space / SPMD).

## Key Concepts
- **SPMD vs SIMD** — SPMD is the *abstraction* ("spawn a gang of `programCount` instances"); SIMD is the *implementation* (vector instructions, masked lanes). ISPC lets the abstraction leak via `programIndex`/`uniform`.
- **Interleaved vs blocked assignment** — Interleaved maps gang lane `k` to indices `i + k`, enabling a single packed vector load (`vmovaps`); blocked maps lane `k` to a contiguous chunk, which requires a strided/gather (`vgatherdps`) for the same access pattern.
- **`foreach`** — raises the level of abstraction: programmer declares iterations are independent; the system chooses how to map them to instances (interleaved / blocked / dynamic).
- **Cross-instance primitives** — `reduce_add`, `reduce_min`, `broadcast`, `shift`, `rotate` enable cooperation between lanes of a gang; needed because per-instance `varying` values can't be returned to scalar C code.
- **Amdahl's Law** — if fraction `S` is inherently sequential, max speedup ≤ `1/S`. A 0.1% serial fraction caps speedup at 1000× no matter how many ALUs.
- **Decomposition → Assignment → Orchestration → Mapping** — the four phases of turning a problem into a parallel program. Each may be done by the programmer, compiler, runtime, or hardware.
- **Red-black reordering** — algorithm change (permitted by domain knowledge of Gauss-Seidel) that trades identical floating-point output for massively parallel update phases.
- **Shared address space model** — threads communicate implicitly through shared memory; synchronize explicitly with **locks** (mutual exclusion) and **barriers** (phase separation).
- **Data-parallel model** — single thread of control with `for_all` loops; synchronization is implicit at loop boundaries, communication uses reduction primitives.

## Detailed Notes

### Finishing ISPC: abstraction vs. implementation

The Lecture-3 `sinx` example spawns a **gang** of `programCount` instances on a call to an `export`ed ISPC function. Each instance has its own `programIndex` and its own copy of non-`uniform` (i.e., `varying`) locals. `uniform` is only an *optimization hint*: all instances hold the same value, so the compiler can use scalar registers and skip per-lane masking. The program isn't wrong without `uniform`, just slower.

**Two ways to distribute array work across the gang:**

Interleaved (v1) — lane `k` handles indices `k, k+programCount, k+2·programCount, …`:
```c
for (uniform int i=0; i<N; i+=programCount) {
    int idx = i + programIndex;
    float value = x[idx];
    ...
}
```
At each iteration the 8 lanes of the gang access the 8 *contiguous* floats `x[i..i+7]` — one aligned packed vector load (`_mm256_load_ps` / `vmovaps`).

Blocked (v2) — lane `k` handles a contiguous chunk of `N/programCount` indices:
```c
uniform int count = N / programCount;
int start = programIndex * count;
for (uniform int i=0; i<count; i++) {
    int idx = start + i;
    float value = x[idx];
    ...
}
```
Now the 8 lanes simultaneously touch indices `0, count, 2·count, …` — eight non-contiguous addresses, requiring a **gather** (`_mm256_i32gather_ps` / `vgatherdps`), which is slower than a packed load. For streaming memory traffic, interleaved is the better fit for SIMD hardware.

### `foreach`: declarative iteration

```c
foreach (i = 0 ... N) {
    float value = x[i];
    ...
    result[i] = value;
}
```
The programmer says "these iterations are independent; assign them however you like." The compiler can pick any of four schemes: (1) one lane does everything; (2) interleave; (3) block; (4) dynamic via `atomic_add_local(&nextIter, 1)`. In practice current ISPC uses static interleaved assignment.

Inside a `foreach`, code reads almost sequentially — *independently, for each `i`, do this*. The exceptions are `uniform` variables and cross-lane primitives.

### Cross-instance standard library

Because a `varying float sum` is 8 distinct values (one per lane), you cannot just return it to scalar C code. The fix is a reduction primitive:

```c
export uniform float sum_array(uniform int N, uniform float* x) {
    float partial = 0.0f;            // varying: one per lane
    foreach (i = 0 ... N)
        partial += x[i];
    return reduce_add(partial);      // collapses 8 lanes → uniform float
}
```
Analogous to AVX intrinsics that accumulate 8 partial sums in a `__m256` register, then horizontally add the 8 lanes at the end. Other primitives: `reduce_min`, `broadcast(v, idx)`, `rotate(v, offset)`, `shift(v, k)`.

With `shift`, the gang can cooperate in non-trivial ways — e.g., product of 8 elements in 3 steps using tree reduction:
```c
float val1 = x[programIndex];
float val2 = shift(val1, 1);
if (programIndex % 2 == 0) val1 = val1 * val2;
val2 = shift(val1, 2);
if (programIndex % 4 == 0) val1 = val1 * val2;
val2 = shift(val1, 4);
if (programIndex % 8 == 0) *result = val1 * val2;
```

### Programs you can write but shouldn't

ISPC is low-level — it exposes `programIndex`/`programCount`, so you can write programs whose output depends on the gang size or that have races:

```c
// undefined: two iterations may write the same y[*] location
foreach (i = 0 ... N) {
    if (i >= 1 && x[i] < 0) y[i-1] = x[i];
    else                    y[i]   = x[i];
}
```
Writing `float sum = 0; foreach(...) sum += x[i]; return sum;` is a compile-time type error (can't return a varying), and making `sum` uniform is also an error (can't pick which lane's `x[i]` to add). Only the `reduce_add` version is correct.

A hypothetical higher-level ISPC could hide `programIndex`/`programCount` entirely, forcing pure `foreach`/collection-level thinking (à la NumPy, PyTorch). Take-home: **programming models are abstractions that admit many valid implementations; keep the two levels separate in your head.**

### ISPC tasks
The whole gang still runs on one x86 thread, using SIMD within that thread. To span cores you use `launch[N] task_fn(...)` — ISPC's task abstraction — which feeds a thread-pool worker that picks the next task off a list.

### The thought process for parallelizing

1. **Identify** work that can run in parallel.
2. **Partition** work (and associated data).
3. **Manage** data access, communication, and synchronization.

The pipeline (Culler/Singh/Gupta):

```
Problem
  │  Decomposition
  ▼
Subproblems (tasks)
  │  Assignment          (to threads/instances/lanes)
  ▼
Workers (parallel threads)
  │  Orchestration       (communication, sync, scheduling, layout)
  ▼
Communicating parallel program
  │  Mapping             (to hardware execution units)
  ▼
Execution
```
Any phase may be done by the programmer, compiler, runtime, or hardware.

### Amdahl's Law

If fraction `S` of the work is inherently sequential, maximum speedup on P processors is bounded by `1 / (S + (1−S)/P)`, which tends to `1/S` as P → ∞.

Worked example — process an N×N image:
- Step 1: double every pixel (fully parallel, cost `N²`).
- Step 2: average all pixels (reduction, cost `N²`).

Sequential: `2N²`. Parallelize only step 1 → time `N²/P + N²` → speedup capped at 2 no matter how large P gets. Parallelize step 2 as a tree reduction (`N²/P + P`) → speedup approaches P when `N ≫ P`.

Summit (148 M ALUs) with 0.1% serial → speedup ≤ ~1000. The vast majority of the machine sits idle on that serial tail.

### Decomposition is (mostly) the programmer's job
Automatic parallelization works for simple loop nests; the general case — especially data-dependent dependencies — is still an open research problem.

### Assignment
Static (at compile/launch time) vs. dynamic (at runtime). Examples:
- C++11 threads, programmer-assigned: split array into halves, spawn one thread, main thread does the other half.
- ISPC interleaved: programmer writes the assignment explicitly.
- ISPC `foreach`: system picks (currently static interleaved).
- ISPC `launch[100] task(...)`: dynamic — worker threads pull the next task off a list.

Assignment goals: workload balance, low communication cost.

### Orchestration & mapping
Orchestration is the messy middle: structure communication, add sync, lay out data for locality, schedule tasks. Mapping assigns threads to hardware — done by the OS (CPU thread → core), the compiler (ISPC instance → vector lane), or the hardware (CUDA block → SM). Decisions include co-locating cooperating threads on one core (for sharing) vs. pairing threads that use complementary resources (compute-bound with bandwidth-bound) on one core (for utilization).

### Case study: 2D grid solver

Sequential Gauss-Seidel sweep on an (N+2)×(N+2) grid:
```c
while (!done) {
    diff = 0.f;
    for (int i=1; i<n; i++)
        for (int j=1; j<n; j++) {
            float prev = A[i][j];
            A[i][j] = 0.2f * (A[i][j] + A[i][j-1] + A[i-1][j]
                            + A[i][j+1] + A[i+1][j]);
            diff += fabs(A[i][j] - prev);
        }
    if (diff / (n*n) < TOLERANCE) done = true;
}
```

**Finding dependencies.** Each cell reads its left and top neighbors *after they were updated this iteration*, and its right and bottom neighbors *from the previous iteration*. So within one sweep cell `(i,j)` depends on `(i-1,j)` and `(i,j-1)`. The only independent work lies along the anti-diagonals — cells on the same diagonal can be updated in parallel, but diagonals must proceed in order. Bad news: the first and last diagonals have only one cell (no parallelism there), and you'd synchronize after every diagonal.

**Algorithmic fix — red-black coloring.** Color the grid like a checkerboard. Reorder the algorithm to update all red cells, then all black cells. A red cell's four neighbors are all black (and vice-versa), so within a color *all updates are independent*. You need domain knowledge that Gauss-Seidel still converges under this reordering (it does, to an equivalent but not bit-identical solution). This is a common parallel-programming move: **change the algorithm to expose parallelism.**

Phases per iteration:
```
compute red cells (parallel) → wait → exchange red updates
compute black cells (parallel) → wait → exchange black updates
```

**Assignment choices.** Block-row vs block-column vs checkerboard blocks affect how much boundary data must be communicated to neighbors each iteration. For block-row assignment on P processors, only two rows cross processor boundaries; for block-column, two columns. Which is better depends on the hardware (cache layout, interconnect, whether the domain is shared-memory or distributed).

### Data-parallel expression

```c
while (!done) {
    for_all (red cells (i,j)) {
        float prev = A[i][j];
        A[i][j] = 0.2f * (A[i-1][j] + A[i][j-1] + A[i][j]
                        + A[i+1][j] + A[i][j+1]);
        reduceAdd(diff, fabs(A[i][j] - prev));
    }
    if (diff/(n*n) < TOLERANCE) done = true;
}
```
- Decomposition: one grid element = one piece of independent work.
- Assignment: left to the system.
- Orchestration: implicit — end-of-`for_all` is a barrier; `reduceAdd` is a cross-worker primitive.

### Shared-address-space / SPMD expression

All threads run `solve()`; they share `A`, `diff`, a lock, and a barrier. Each thread picks a row range from its `threadId`.

```c
void solve(float* A) {
    int tid  = getThreadId();
    int myMin = 1 + (tid * n / NUM_PROCESSORS);
    int myMax = myMin + (n / NUM_PROCESSORS);

    while (!done) {
        float myDiff = 0.f;
        diff = 0.f;
        barrier(myBarrier, NUM_PROCESSORS);           // (1)

        for (j = myMin; j < myMax; j++)
            for (i : red cells in row j) {
                float prev = A[i][j];
                A[i][j] = 0.2f * (A[i-1][j] + A[i][j-1] + A[i][j]
                                + A[i+1][j] + A[i][j+1]);
                myDiff += fabs(A[i][j] - prev);
            }

        lock(myLock);
        diff += myDiff;                                // one reduction per thread
        unlock(myLock);
        barrier(myBarrier, NUM_PROCESSORS);            // (2)

        if (diff/(n*n) < TOLERANCE) done = true;
        barrier(myBarrier, NUM_PROCESSORS);            // (3)
    }
}
```

**Why the lock?** Without it, `diff += myDiff` is a read-modify-write race: load `diff`, add, store — two threads can both load the old value and one update is lost. Mutual exclusion makes the sequence atomic.

**Why accumulate locally first?** An earlier (wrong-on-performance) version locked inside the inner loop — once per `(i,j)`. The fix: accumulate into thread-local `myDiff`, then take the lock once per thread per iteration. This is the general pattern: *reduce locally, combine globally.*

**Why three barriers?**
- Barrier (1) makes sure every thread sees `diff = 0.f` before anyone starts adding into it.
- Barrier (2) makes sure every thread has finished contributing to `diff` before anyone reads it for the convergence test.
- Barrier (3) makes sure every thread reads `diff` (and sets its own `done`) before any thread charges ahead to next iteration and resets `diff = 0.f`.

Conceptually, a barrier says *"everything before here in every thread happens before anything after here in any thread"* — a conservative, easy-to-reason-about way to express dependencies.

**Reducing to one barrier per iteration.** Keep three `diff[0..2]` slots and rotate (`index = (index+1) % 3`). Thread resets the *next* slot before the barrier, so at the barrier all threads agree that (a) current-iteration contributions are finished and (b) next-iteration slot is already zero. Trades memory footprint for fewer synchronization points — another common pattern.

### Comparing the two models
- **Data-parallel**: single logical control thread; parallelism exposed via `for_all`; sync is implicit (end of loop); communication via reductions.
- **Shared address space**: multiple SPMD threads; communication implicit via shared loads/stores; sync explicit via locks/barriers. Feels like a natural extension of sequential C — and is what almost every discussion in this course assumes by default.

## Examples / Worked Problems
- **Interleaved vs blocked memory access**: same computation, interleaved schedule becomes one `vmovaps`, blocked becomes a `vgatherdps` — motivates why interleaved is the SIMD-friendly default.
- **`sum_array` in ISPC**: shows why the naive `float sum = 0; sum += x[i]` is a type error (varying return), why `uniform float sum` is a type error (which lane's value?), and how `reduce_add(partial)` correctly implements a horizontal sum.
- **`vec8product`**: tree reduction in `lg 8 = 3` steps using `shift` — cross-lane cooperation without leaving SIMD.
- **Amdahl image-processing example**: parallelizing only the per-pixel step caps speedup at 2 regardless of P; tree-reducing the average lifts speedup back to ~P.
- **Red-black grid solver**: diagonals give correct-but-awful parallelism; color reordering gives flat parallel phases. Key skill: *change the algorithm, not just the schedule.*
- **Three barriers vs one**: rotating a 3-slot `diff[]` array removes the read/reset race — shows how extra memory can buy fewer syncs.

## Takeaways
1. **Abstraction ≠ implementation.** ISPC's SPMD gang is the programming model; SIMD vector instructions are the implementation. Keep these layers distinct when reasoning about correctness vs. performance.
2. **Assignment matters for memory.** Interleaved layouts turn into packed vector loads; blocked layouts turn into gathers. Same computation, very different throughput.
3. **Build programs in four phases**: decompose → assign → orchestrate → map. Each phase has its own goals and its own set of tools.
4. **Amdahl is brutal at scale.** A sub-1% serial tail caps speedup in the thousands. Always parallelize (or remove) the reduction steps, not just the obviously parallel ones.
5. **When the algorithm fights you, change the algorithm.** Red-black coloring shows how domain knowledge turns a synchronization-dominated computation into a clean two-phase parallel sweep.
6. **Reduce locally, combine globally.** Thread-local accumulation plus a single critical section is the standard way to avoid contention on shared reductions.
7. **Barriers express dependencies conservatively.** Understand *why* each one exists; extra memory (e.g., rotating buffers) can often eliminate some.

## Open Questions / Follow-ups
- ISPC tasks and multi-core execution — covered while doing Assignment 1.
- Cost of gathers/scatters on real AVX2/AVX-512 hardware — revisited with the multicore architecture lectures.
- Alternative parallel programming models beyond SPMD and data-parallel (e.g., message passing, dataflow, streaming) — coming lectures.
- Formal treatment of memory models and what "happens-before" means for shared-address-space programs — deferred.
- Choosing assignment (block-row vs block-column vs 2D tiles) based on machine characteristics — revisited with locality and communication lectures.

## Sources
- Slides: `slices/04_progbasics.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Grid solver example adapted from Culler, Singh, and Gupta, *Parallel Computer Architecture: A Hardware/Software Approach*.
- ISPC documentation: https://ispc.github.io/ispc.html
- Course site: https://gfxcourses.stanford.edu/cs149
