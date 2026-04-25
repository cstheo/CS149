# Lecture 13 — Domain-Specific Programming Systems and Automatic Performance Optimization

## Overview
Manual performance optimization in C++/ISPC/CUDA is hard, tedious, and non-portable — a CS149-educated programmer is scarce. This lecture argues for raising the level of abstraction (domain-specific languages), then automating the optimization search on top of that abstraction. Halide is the running case study: a DSL for image processing that separates *algorithm* from *schedule*, enabling both expert productivity and autoscheduling via tree search + learned cost models. The closing section sketches emerging LLM-agent approaches to kernel generation (KernelBench, self-improving retrieval, prompt optimization) and asks whether an LLM can serve as a great CS149 student.

## Key Concepts
- **Productivity / Performance / Generality triangle** — no language optimizes all three. Mainstream languages trade performance for generality; DSLs trade generality for productivity + performance.
- **Domain-Specific Language (DSL)** — restricted, often declarative language for one domain. Compiler exploits domain knowledge to emit efficient code and may even co-design hardware.
- **Halide** — DSL embedded in C++ for image processing pipelines; powers Google HDR+, parts of portrait mode, Instagram/Adobe filters.
- **Halide Func** — infinite, discrete N-D function from integer coordinates to values; defined by a pure expression. No explicit loops, no storage order.
- **Algorithm vs. Schedule** — two separate programs. Algorithm says *what* to compute; schedule says *how*: loop order, tiling, vectorization, parallelism, where producers are computed (`compute_root`, `compute_at`).
- **Scheduling primitives** — `tile`, `vectorize`, `parallel`, `unroll`, `compute_at`, `store_at`. Let programmer sketch a strategy; compiler emits threads + SIMD intrinsics.
- **Autoscheduler** — tree/beam search over the schedule space guided by a learned cost model (MLP outputting coefficients of a hand-crafted model).
- **LLM kernel agent** — prompt → CUDA code → profile → reflect → edit loop; KernelBench benchmarks correctness and speed.
- **Self-improving agent** — retrieval-augmented: database of (problem, good solution, optimization trajectory) that grows over time.

## Detailed Notes

### Why we need something better than C++/ISPC/CUDA
Assignments 1–4 are the proof: writing performant parallel code requires tiling, SIMD intrinsics, thread partitioning, and careful cache management — expert-only work that is also non-portable. Hand-written SSE code for a 3×3 blur is ~10× faster than naive C on a quad-core CPU but is SSE-only (no AVX2), CPU-only, and unreadable. This is the pain the rest of the lecture tries to remove.

### The design space
Hanrahan's diagram places languages in a triangle with vertices *Performance*, *Productivity*, *Generality*. No language is near all three. Popular languages (Python, Java, C++, CUDA) each sit close to two vertices. DSLs deliberately sacrifice generality to push performance + productivity simultaneously.

A DSL wins when:
- its primitives match how domain experts naturally think;
- its restrictions let the compiler (and sometimes the hardware) do more;
- one program runs efficiently across machines because the system chooses algorithms/strategies per target.

### Image-processing performance tutorial (motivation for Halide)

**Naive 3×3 box blur** (one pass): for each output pixel sum 9 neighbors × weights.
```c
for (int j=0; j<HEIGHT; j++)
  for (int i=0; i<WIDTH; i++) {
    float tmp = 0.f;
    for (int jj=0; jj<3; jj++)
      for (int ii=0; ii<3; ii++)
        tmp += input[(j+jj)*(WIDTH+2)+(i+ii)] * weights[jj*3+ii];
    output[j*WIDTH + i] = tmp;
  }
```
Work = 9·W·H (N²·W·H for N×N filter).

**Two-pass separable blur**: a box filter factors into 1D horizontal then 1D vertical.
```c
for (int j=0; j<HEIGHT+2; j++)
  for (int i=0; i<WIDTH; i++) {
    float tmp=0; for (int ii=0; ii<3; ii++)
      tmp += input[j*(WIDTH+2)+i+ii] * weights[ii];
    tmp_buf[j*WIDTH+i] = tmp;
  }
for (int j=0; j<HEIGHT; j++)
  for (int i=0; i<WIDTH; i++) {
    float tmp=0; for (int jj=0; jj<3; jj++)
      tmp += tmp_buf[(j+jj)*WIDTH+i] * weights[jj];
    output[j*WIDTH+i] = tmp;
  }
```
Work = 6·W·H (2N·W·H), but needs a W×(H+2) intermediate buffer and loads/stores through it — lower arithmetic intensity. Whether it wins depends on locality: if `tmp_buf` overflows cache between the two passes, we pay DRAM traffic that is pure overhead.

**Chunked two-pass v1**: interleave at row granularity. For each output row produce only the 3 rows of `tmp_buf` it needs, into a W×3 scratch buffer. Per output row: step 1 = 3·3·W, step 2 = 3·W → 12·W·H total. More arithmetic (recomputation), but the intermediate is tiny.

**Chunked two-pass v2**: chunk by CHUNK_SIZE rows. Produce CHUNK_SIZE+2 rows of `tmp_buf`, then CHUNK_SIZE rows of output.
```c
for (int j=0; j<HEIGHT; j+=CHUNK_SIZE) {
  for (int j2=0; j2<CHUNK_SIZE+2; j2++)
    for (int i=0; i<WIDTH; i++) { /* horizontal → tmp_buf */ }
  for (int j2=0; j2<CHUNK_SIZE; j2++)
    for (int i=0; i<WIDTH; i++) { /* vertical → output */ }
}
```
For CHUNK_SIZE=16: work = (34/16)·3·W·H ≈ 6.4·W·H, trending to 6·W·H as chunk grows — *and* `tmp_buf` fits in cache. Still not parallelized, not vectorized, not loop-unrolled.

The "good" hand-tuned SSE version fuses all of the above: multi-core (vertical partition), 256×32 tile iteration order, SIMD intrinsics, two passes fused so `tmp` lives in registers/cache. ~10× faster than the naive two-pass, but locked to SSE, CPU-only, and opaque.

### Halide: separate algorithm from schedule

**Algorithm** — pure expressions over integer domains, no loops, no storage:
```c++
Var x, y;
Func blurx, blury, bright, out;
Halide::Buffer<uint8_t> in = load_image("myimage.jpg");
Halide::Buffer<uint8_t> lookup = load_image("s_curve.jpg");

blurx(x,y) = 1/3.f * (in(x-1,y)     + in(x,y)     + in(x+1,y));
blury(x,y) = 1/3.f * (blurx(x,y-1)  + blurx(x,y)  + blurx(x,y+1));
bright(x,y) = min(blury(x,y) * 1.25f, 255);
out(x,y)    = lookup(bright(x,y));

Halide::Buffer<uint8_t> result = out.realize(1024, 1024);
```
The algorithm is a DAG of Funcs: `in → blurx → blury → bright → out`, plus `lookup → out`. Iteration order, fusion, storage, and parallelism are *not* specified.

Real pipelines are big: unsharp mask 9 Funcs, Harris 13, camera RAW 30, local Laplacian 103, VGG-16 eval 64, Google HDR+ **>2000**. A Func DAG is the natural representation, and because it is declarative + feed-forward the compiler can infer all dependencies.

**Schedule** — a second mini-program that chooses the implementation:
```c++
out.tile(x, y, xi, yi, 256, 32).vectorize(xi, 8).parallel(y);
blurx.compute_at(out, x).vectorize(x, 8);
```
This says: iterate `out` in 256×32 tiles, vectorize the inner `xi` loop 8-wide, parallelize tile rows across threads; for each tile compute just the needed `blurx` strip, vectorized 8-wide.

Three choices for `blurx`, same algorithm, different schedules:
- `blurx.compute_root()` — materialize all of `blurx` first (full W×(H+2) buffer), then all of `out`. Simple but high memory, poor producer-consumer locality.
- `blurx.compute_at(out, xi)` — compute the 3 `blurx` values needed for each output pixel on demand inside the innermost loop. Minimal footprint (3 elements), maximum recomputation.
- `blurx.compute_at(out, x)` — inside each tile, compute a 256×34 strip of `blurx` once, then consume it. Tile-sized intermediate lives in cache, reuse is captured, recomputation is bounded. This is the sweet spot for the earlier chunked C code.

Equivalent lowered loop nest for the sample schedule:
```c
for y=0 to num_tiles_y:                // parallelized across threads
  for x=0 to num_tiles_x:
    allocate blurx(258, 34);
    for yi=0 to 34:
      for xi=0 to 258 BY 8:             // 8-wide SIMD
        blurx(xi,yi) = ...
    for yi=0 to 32:
      for xi=0 to 256 BY 8:             // 8-wide SIMD
        out(x*256+xi, y*32+yi) = ...
```

### Halide philosophy and constraints
- Programmer describes the algorithm once; scheduling is a separate, high-level sketch.
- Compiler is *not* smart — it mechanically lowers schedules to pthreads + AVX intrinsics + loop boundary handling.
- Language restrictions that make this possible: regular N-D domains, feed-forward pipelines (with special support for reductions and fixed-depth recursion), and compiler-inferable dependencies.

Early results (Ragan-Kelley 2012):
- Camera RAW pipeline: 463 lines ARM NEON asm → Halide 2.75× less code, 5% faster.
- Bilateral filter: 122 lines C++ → 34 (algorithm) + 6 (schedule) lines Halide; CPU 5.9× faster, GPU 2× faster than hand CUDA.

### Autoscheduling Halide (Adams 2019, Mullapudi 2016)
Even at Google, of 80+ Halide programmers only a tiny group can write good schedules. So the compiler does it:

1. **Schedule as a sequence of decisions.** Walk the Func DAG from outputs backward. For each Func: pick where to `compute_at` in the current loop nest, pick tile sizes (outer dim parallel, inner dim vectorized). Repeat.
2. **Search.** The schedule space is enormous (hundreds of thousands of candidates per program). Use greedy / beam search over a tree of partial schedules, each with an estimated cost.
3. **Cost model.** A small MLP scores a (program, schedule) pair in ~tens of microseconds — 1.4 M schedules in 166 s. It doesn't directly predict runtime; it outputs 27 coefficients plugged into a hand-crafted analytic model. Training data: large corpus of randomly generated Halide programs, compiled and timed on real hardware.

**Result:** for image processing on CPUs, the autoscheduler matches or beats expert-authored schedules — the expert has to work hard to win. Earlier autoschedulers (Mullapudi 2016) already compressed days of expert work into minutes.

**Meta-lesson.** The same abstraction that made scheduling productive for humans (a small, structured primitive set) made it *enumerable* for a machine.

### LLM-based code generation (emerging)

**Trial-and-error with reflection.**
```
prompt + PyTorch code → LLM → CUDA code → run on H100 → profile
  → feed stats (SM util, DRAM util, L2 hit rate, timing, correctness)
  → LLM reflects, identifies bottleneck, edits code → repeat
```
**KernelBench** evaluates agents over hundreds of PyTorch operators: must produce correct and fast CUDA.

**DSLs help LLMs too.** Triton, CUTLASS/CuTe, TileLang raise the abstraction for DNN kernels — LLMs assemble high-performance primitives instead of writing raw CUDA, reducing hallucinations. Downside: these languages have less training data; quality lags popular languages but will close over time.

**Self-improvement ideas:**
1. **Fine-tune** an LLM on experience for a narrow task family. Needs many tasks and infra to tune large models.
2. **Retrieval-augmented DB.** Agent maintains a database of solved problems *and* the sequence of optimization decisions that produced each solution. On a new problem it retrieves similar cases. Reported result: the DB-augmented agent solves far more problems than the bare LLM starting point, and the gap grows as the DB grows.
3. **Prompt optimization.** A meta-LLM inspects trajectories of prior optimization loops and distills durable principles into an updated system prompt.
4. **Search + LLM.** Combine Halide-style exhaustive schedule search with LLM agents — very high optimization cost, but some of the best results.

## Examples / Worked Problems
- **Blur arithmetic intensity.** 1-pass = 9·W·H work over W·H output loads+stores; 2-pass = 6·W·H work + extra W·H intermediate reads+writes → lower arithmetic intensity. Chunked 2-pass recovers cache locality; as chunk size grows, work/pixel → 6.
- **Three compute_at choices for `blurx`.** Root (huge buffer, high memory traffic) vs inner-loop (3 elements, 3× recomputation) vs tile (strip per tile — balanced; matches the best hand C code). Good exercise for seeing how one scheduling primitive shifts the time/space tradeoff.
- **Schedule enumeration.** For a 3-Func DAG, enumerate `compute_at` locations for each producer given a fixed consumer loop-nest tile structure — shows how fast the space grows and motivates learned cost models.

## Takeaways
1. **Raise the abstraction, then automate.** DSLs are productive for humans *and* create a structured search space for autotuners and learned models.
2. **Separate what from how.** Halide's algorithm/schedule split is the pivotal design move — it lets one algorithm target many machines without rewriting.
3. **Good representations earn services.** Declarative, side-effect-free, feed-forward pipelines → compiler can infer dependencies, parallelize, vectorize, fuse, and now autoschedule.
4. **Autoscheduling already matches experts** on CPU image pipelines (Adams 2019), using tree search + an MLP-predicted cost model.
5. **LLM agents are the emerging third mechanism**: reflect-on-profile loops, retrieval of past solutions, and prompt self-optimization. DSLs like Triton/CuTe amplify LLM effectiveness.
6. **The debate worth watching:** is the lasting value in DSL design (stable, interpretable, searchable abstractions) or in the LLM agent (general, improving, expensive)? Likely both, combined.

## Open Questions / Follow-ups
- Can an LLM agent function as a great CS149 student, and at what token cost?
- How do we design DSLs specifically to be LLM-friendly (few primitives, clear semantics, rich profiling feedback)?
- When is the cost of an exhaustive autotuner worth paying vs a cheap LLM pass — and how to blend them?
- How to scale retrieval-based self-improvement without the DB drifting or overfitting to a narrow problem distribution?
- Applying Halide-style algorithm/schedule separation to domains beyond image processing and DNNs.

## Sources
- Slides: `slices/13_autooptimize.pdf` (Stanford CS149, Fall 2025).
- Raw lecture text: `notes/.raw/13_autooptimize.txt`.
- Ragan-Kelley, Adams, et al. *Halide: a language and compiler for optimizing parallelism, locality, and recomputation in image processing pipelines.* SIGGRAPH 2012 / PLDI 2013.
- Mullapudi et al. *Automatically Scheduling Halide Image Processing Pipelines.* SIGGRAPH 2016.
- Adams et al. *Learning to Optimize Halide with Tree Search and Random Programs.* SIGGRAPH 2019.
- KernelBench (LLM CUDA kernel benchmark); Triton, CUTLASS/CuTe, TileLang DSLs.
- Course site: https://gfxcourses.stanford.edu/cs149
