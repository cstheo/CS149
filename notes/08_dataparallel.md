# Lecture 8 — Data-Parallel Thinking

## Overview
Shifts the mental model from "what workers do" to **describing algorithms as operations on sequences**: map, filter, fold/reduce, scan, segmented scan, gather/scatter, sort, groupBy, partition. The promise is that each of these primitives has a high-performance parallel implementation, so programs composed from them can run efficiently on many-core machines and GPUs — *provided* they aren't bandwidth-bound. Deep-dives on parallel scan (work-inefficient O(N lg N), work-efficient O(N), hybrid two-level implementations), segmented scan, and how combinations of these primitives express irregular problems (sparse matvec, particle-grid binning, histograms) as regular, massively-parallel computations.

## Key Concepts
- **Sequence** — ordered collection `Sequence<T>` accessed only through bulk operations (not random element access). Examples: Haskell `seq T`, Scala `List[T]`, NumPy arrays, Pandas DataFrames, PyTorch/JAX tensors.
- **Map** — apply side-effect-free unary `f :: a -> b` to every element. Embarrassingly parallel because order does not matter.
- **Fold / reduce** — combine elements with binary `f :: (b,a) -> b` seeded by an identity. Parallelisable when an associative "combiner" `comb :: (b,b) -> b` exists (which is automatic if `f` itself is associative on `(b,b)`).
- **Scan** — prefix reduction. `scan_inclusive(⊕, A)[i] = a0 ⊕ … ⊕ ai`; exclusive excludes `ai`. When `⊕ = +`, this is a prefix sum.
- **Work vs span** — work is total operations performed; span is the longest dependency chain (parallel critical path). Scan: work-inefficient tree is O(N lg N) work / O(lg N) span; work-efficient (Blelloch) is O(N) work / O(lg N) span.
- **Segmented scan** — simultaneously scan over many variable-length contiguous segments using a flag vector that marks segment starts. Turns "for each X, for each Y in X" into one regular data-parallel pass.
- **Gather / scatter** — `output[i] = input[index[i]]` and `output[index[i]] = input[i]`. Gather landed in AVX2 (2013); scatter requires AVX-512; both are natively supported on GPUs but still costlier than contiguous loads.
- **Group-by / filter / sort / partition** — additional sequence primitives that reshape data so the remaining work becomes regular and parallel.
- **Bandwidth-bound caveat** — most data-parallel solutions make multiple passes over the data; arithmetic intensity stays low, so memory bandwidth often limits wall-clock speedup.

## Detailed Notes

### Why large parallelism matters
Modern targets (multi-socket CPUs, clusters, SIMD + multithreading, and GPUs) demand a lot of independent work. A V100 GPU runs at 1.245 GHz, has 80 SMs × 4 × 16 = 5,120 fp32 mul-add ALUs (≈12.7 TFLOPs), and can interleave up to 80 × 64 = 5,120 warps (163,840 CUDA threads) against 16 GB of HBM at 900 GB/s through a 6 MB L2. Programs with little exposed parallelism or low arithmetic intensity cannot keep such a chip busy. Spotting where dependencies exist (and where they don't) is the core skill: e.g., `x=a+b; y=b*7; z=(x-y)*(x+y)` has a DAG where `x` and `y` are independent and can run in parallel.

### Map
`map :: (a -> b) -> seq a -> seq b` applies `f` elementwise. In C++ STL:
```cpp
int f(int x) { return x + 10; }
int a[] = {3, 8, 4, 6, 3, 9, 2, 8};
int b[8];
std::transform(a, a+8, b, f);
```
In Haskell: `b = map f a`. Because `f` is pure, the runtime is free to partition the sequence into P chunks, map each in parallel, and concatenate:
```
map f s =
    partition s into P subsequences
    for each s_i in parallel: out_i = map f s_i
    out = concatenate(out_i)
```
JAX exposes this as `vmap`.

### Fold and parallel fold
Sequential `foldLeft(init, f, [a0..an-1])` walks left-to-right: `acc = f(acc, ai)`. Example with `f=+`, `init=10`, input `[3,8,4,6,3,9,2,8]` → `53`.

For parallelism you need a combiner:
```
f     :: (b,a) -> b
comb  :: (b,b) -> b
fold_par :: b -> ((b,a) -> b) -> ((b,b) -> b) -> seq a -> b
```
Each leaf folds a chunk with `f` starting from identity; internal tree nodes combine with `comb`. If `f :: (b,b) -> b` is already associative, `comb = f` and no separate function is needed.

### Scan — naive, work-inefficient parallel algorithm
```c
float op(float a, float b) { ... }
scan_inclusive(float* in, float* out, int N) {
  out[0] = in[0];
  for (int i = 1; i < N; i++)
    out[i] = op(out[i-1], in[i]);
}
```
A straightforward parallel version uses a doubling pattern: at step `d = 0..lg N − 1`, every element adds the element `2^d` positions to its left. After `lg N` steps the inclusive prefix is done. This performs O(N lg N) work with O(lg N) span — *more* work than the sequential algorithm, so raw parallelism is bought at an efficiency cost.

### Work-efficient exclusive scan (Blelloch)
Two phases over the array, sweeping a balanced binary tree embedded in the array:

```
Up-sweep (reduce):
for d = 0 to (lg N - 1):
    forall k = 0 to N-1 by 2^(d+1):
        a[k + 2^(d+1) - 1] = a[k + 2^d - 1] + a[k + 2^(d+1) - 1]

Down-sweep:
a[N-1] = 0
for d = (lg N - 1) down to 0:
    forall k = 0 to N-1 by 2^(d+1):
        tmp = a[k + 2^d - 1]
        a[k + 2^d - 1]     = a[k + 2^(d+1) - 1]
        a[k + 2^(d+1) - 1] = tmp + a[k + 2^(d+1) - 1]
```
Total work is O(N) and span is O(lg N). Conceptually, the up-sweep builds partial sums at each tree node (like a reduction); the down-sweep pushes the left-subtree sum to the right subtree while zeroing-out the root. The diagram in the slides (16 elements) shows the value `a0‥i` propagating down the tree so every position ends up with the exclusive-prefix value.

Constants and locality matter: the work-efficient algorithm does O(N) ops but with a larger constant and scattered access pattern; on a machine where sequential scan is already bandwidth-bound, simpler schemes can win.

### Two-core hybrid scan
On a small shared-memory machine, the practical scheme is:
1. P1 does sequential scan on `a[0..7]`, P2 does sequential scan on `a[8..15]` in parallel.
2. Let `base = a[0..7]`'s total.
3. P2 adds `base` to each of its results.

Work is O(N) with constant ~1.5× sequential. Access is contiguous (great spatial locality); cross-processor access to `base` is cheap on small systems but may matter on NUMA.

### SIMD (warp-level) scan in CUDA
For 32-wide SIMD it is better to use the O(N lg N) doubling algorithm because the work-efficient algorithm leaves most lanes idle and actually *increases* dynamic instruction count:
```cuda
__device__ int scan_warp(int *ptr, const unsigned int idx) {
    const unsigned int lane = idx % 32;  // 0..31
    __syncwarp();
    for (int i = 0; i < 5; i++) {         // 2^5 = 32
        int shift = 1 << i;
        if (lane >= shift) {
            int tmp1 = ptr[idx - shift];
            int tmp2 = ptr[idx];
            __syncwarp();
            ptr[idx] = tmp1 + tmp2;
            __syncwarp();
        }
    }
    return (lane > 0) ? ptr[idx-1] : 0;   // exclusive result
}
```
Work is N lg N, but SIMD utilisation is high.

### Block-level scan (128 or 1024 elements)
Compose the warp scan hierarchically. Each warp scans its 32-element slice; lane 31 of each warp writes its total into a small `bases` buffer; warp 0 scans `bases`; then every warp except warp 0 adds its corresponding base back into its partial results:
```cuda
__device__ void scan_block(int* ptr, const unsigned int idx) {
    const unsigned int lane    = idx % 32;
    const unsigned int warp_id = idx >> 5;

    int val = scan_warp(ptr, idx);                  // 1. per-warp scan
    if (lane == 31) ptr[warp_id] = ptr[idx];        // 2. per-warp totals
    __syncthreads();
    if (warp_id == 0) scan_warp(ptr, idx);          // 3. scan the totals
    __syncthreads();
    if (warp_id > 0) val = val + ptr[warp_id - 1];  // 4. add base
    __syncthreads();
    ptr[idx] = val;
}
```

### Million-element scan across blocks
Three kernel launches:
1. Launch-1: each thread block scans its 1024-element chunk (block-local).
2. Launch-2: another block scans the array of per-block totals to produce each block's base.
3. Launch-3: every block adds its base to its chunk.

Beyond ~1M elements, phase 2 itself must be split across blocks recursively.

### Implementation themes for scan
- Use only as much parallelism as the machine needs; extra parallelism wastes work and bandwidth.
- Match the memory hierarchy: do warp-level scans in shared memory, block-level scans in L1/shared, global scans through DRAM in stages.
- Use different algorithms at different levels: doubling-scan inside a warp, work-efficient across a block, sequential on a lone CPU core.

### Segmented scan
Many problems are "for each X, for each Y in X" where inner lengths are uneven. Represent as a flat data sequence plus a start-flag vector:
```
A     = [[1,2,3],[4,5,6,7,8]]
flag  = 1 0 0 1 0 0 0 0
data  = 1 2 3 4 5 6 7 8
```
A work-efficient segmented exclusive scan is the Blelloch algorithm augmented with flag propagation (flags OR up on the up-sweep; the down-sweep resets the accumulator whenever a segment start is encountered):
```
Up-sweep:
for d = 0 to (lg N - 1):
    forall k = 0 to N-1 by 2^(d+1):
        if flag[k + 2^(d+1) - 1] == 0:
            data[k + 2^(d+1) - 1] = data[k + 2^d - 1] + data[k + 2^(d+1) - 1]
        flag[k + 2^(d+1) - 1] = flag[k + 2^d - 1] || flag[k + 2^(d+1) - 1]

Down-sweep:
data[N-1] = 0
for d = (lg N - 1) down to 0:
    forall k = 0 to N-1 by 2^(d+1):
        tmp = data[k + 2^d - 1]
        data[k + 2^d - 1] = data[k + 2^(d+1) - 1]
        if flag_original[k + 2^d] == 1:   # a preserved copy is required
            data[k + 2^(d+1) - 1] = 0     # reset at segment start
        else if flag[k + 2^d - 1] == 1:
            data[k + 2^(d+1) - 1] = tmp
        else:
            data[k + 2^(d+1) - 1] = tmp + data[k + 2^(d+1) - 1]
        flag[k + 2^d - 1] = 0
```

### Gather and scatter
```
output_seq = gather(index_seq, data_seq)    # output[i] = data[index[i]]
output_seq = scatter(index_seq, data_seq)   # output[index[i]] = data[i]
```
Picture gather as reading through an index vector to pull values out of a data array into contiguous output; scatter does the reverse. Hardware support: AVX2 gather (2013); scatter arrived in AVX-512; GPUs do both but they remain more expensive than contiguous accesses.

### Sparse matrix × vector with scan primitives
Given `y = M x` in compressed sparse row format, e.g.
```
values     = [[3,1],[2],[4],[2,6,8]]
cols       = [[0,2],[1],[2],[1,2,3]]
row_starts = [0, 2, 3, 4]
x          = [x0,x1,x2,x3]
```
Steps:
1. **Gather** `gathered[i] = x[cols[i]]` → `[x0,x2,x1,x2,x1,x2,x3]`.
2. **Map** `products[i] = values[i] * gathered[i]` → `[3x0, x2, 2x1, 4x2, 2x1, 6x2, 8x3]`.
3. Build flags from `row_starts` → `[1,0,1,1,1,0,0]`.
4. **Inclusive segmented scan** with `+` → `[3x0, 3x0+x2, 2x1, 4x2, 2x1, 2x1+6x2, 2x1+6x2+8x3]`.
5. Take the last element of each segment → `y = [3x0+x2, 2x1, 4x2, 2x1+6x2+8x3]`.

### Scatter as sort + gather (or sort + segmented scan)
If `index` is a permutation (unique indices, every slot written), a scatter is equivalent to sorting the input by `index`:
```
scatter(index, input, output)  ≡  sort input by values in index
```
If indices repeat and you actually want `atomicOp`:
1. Sort `input` by `index` (carrying values along).
2. Compute segment-start flags where the sorted index changes.
3. Segmented-scan within each range with `op`.
4. Read the last element per segment as the result for that destination.

This turns fine-grained atomic synchronisation into a regular sort + scan.

### Building a uniform grid of particles on a GPU
Problem: bin 1M particles into a 16-cell 2D grid. This is foundational for N-body queries ("particles within radius R") because only neighbouring cells need to be searched.

- **Solution 1 — parallelize over particles with one global lock.** Every thread computes its cell and appends to a shared list. Massive contention.
- **Solution 2 — per-cell locks.** 16× less contention, still a bottleneck under skewed distributions.
- **Solution 3 — parallelize over cells.** No synchronisation, but only 16 tasks (GPUs need thousands) and work is 16× the sequential amount (every cell re-scans every particle).
- **Solution 4 — partial grids + merge.** Spawn N thread blocks, each building its own small grid in CUDA shared memory, then merge. Reduces contention by N and confines locking to fast block-local memory at the cost of merge work and N× memory.
- **Solution 5 — data-parallel (map + sort + scan).**
  1. **Map**: for each particle compute its `grid_cell`. e.g., `grid_cell = [9,6,6,4,6,4]`, `particle_index = [0..5]`.
  2. **Sort** particle indices by `grid_cell`: `particle_index = [3,5,1,2,4,0]`, `grid_cell = [4,4,6,6,6,9]`.
  3. **Find start/end** of each cell by comparing each element to its neighbour:
     ```
     this_cell = grid_cell[thread_index];
     prev_cell = grid_cell[thread_index - 1];
     if (thread_index == 0) cell_starts[this_cell] = 0;
     else if (this_cell != prev_cell) {
         cell_starts[this_cell] = thread_index;
         cell_ends[prev_cell]   = thread_index;
     }
     if (thread_index == numParticles - 1) cell_ends[this_cell] = thread_index + 1;
     ```
     Massively parallel, no fine-grained locks — but pays for a sort and extra passes (bandwidth).

### Parallel histogram from map + sort
Given `float input[N]` and `int f(float)` mapping to bin ids, build `histogram_bins[NUM_BINS]`:
```cuda
void compute_bin(float* input, int* bin_ids) {
    bin_ids[thread_index] = f(input[thread_index]);
}

void find_starts(int* bin_ids, int* bin_starts) {
    if (thread_index == 0 || bin_ids[thread_index] != bin_ids[thread_index-1])
        bin_starts[bin_ids[thread_index]] = thread_index;
}

// driver
launch<<<N>>>compute_bin(input, bin_ids);
sort(N, bin_ids, sorted_bin_ids);
launch<<<N>>>find_starts(sorted_bin_ids, bin_starts);
launch<<<NUM_BINS>>>bin_sizes(bin_starts, histogram_bins, N, NUM_BINS);
```
`bin_sizes` handles empty bins by scanning forward to the next non-empty start:
```cuda
void bin_sizes(int* bin_starts, int* histogram_bins, int num_items, int num_bins) {
    if (bin_starts[thread_index] == -1) {
        histogram_bins[thread_index] = 0;
    } else {
        int next_idx = thread_index + 1;
        while (next_idx < num_bins && bin_starts[next_idx] == -1) next_idx++;
        if (next_idx < num_bins)
            histogram_bins[thread_index] = bin_starts[next_idx] - bin_starts[thread_index];
        else
            histogram_bins[thread_index] = num_items - bin_starts[thread_index];
    }
}
```

### Other handy sequence primitives
- **groupBy** — `Seq (key, T) → Seq (key, Seq T)`. Gathers elements sharing a key.
- **filter** — drop elements failing a predicate.
- **sort**, **partition**, **flatten**, **join** — complete the vocabulary.

## Examples / Worked Problems
- Parallel fold with combiner on `[3,8,4,6,3,9,2,8]`, `+`, init 10 → 53 (pairwise tree reduction).
- Inclusive `+`-scan of `[3,8,4,6,3,9,2,8]` → `[3,11,15,21,24,33,35,43]`.
- 16-element exclusive scan via up-sweep/down-sweep (the diagrams in slides walk through each stage).
- Two-core scan of 16 elements: each core runs sequential scan on its 8-element half, then the right half is offset by the left half's total — constant ≈1.5× sequential work.
- Warp-level CUDA scan (N lg N work, high SIMD utilisation) composed into block-level and then grid-wide scan with three kernel launches for 1M elements.
- SpMV expressed as gather → map(*) → segmented-scan(+) → pick-last-per-segment.
- Scatter with collisions lowered to sort-by-index → flags → segmented scan of `atomicOp`.
- Particle binning solution 5: map → sort by cell → neighbour-compare to find cell starts/ends.
- Histogram: map → sort → find_starts → bin_sizes, handling empty bins via forward scan.

## Takeaways
1. Describe computation as operations on sequences; the primitives (map, filter, fold, scan, segmented scan, gather/scatter, sort, groupBy) already have good parallel implementations.
2. Understanding dependencies — and the absence of them — is the starting point for any parallel design.
3. Scan has O(N lg N) and O(N) algorithms; choose per level of the machine (doubling scan inside a warp, work-efficient across a block, sequential on a CPU core, hybrid across GPUs).
4. Segmented scan regularises nested "for each X, for each Y in X" structures so irregular work becomes data-parallel.
5. Recurring tactic: turn *irregular* parallelism into *regular* parallelism and *fine-grained* synchronisation into *coarse* synchronisation — usually via extra passes (sort/scan) over the data.
6. The price is bandwidth. Multi-pass data-parallel algorithms are often bandwidth-bound; arithmetic intensity and locality remain the ultimate constraints.
7. These primitives underpin CUDA Thrust, Pandas, JAX, Spark, and Hadoop — the same thinking scales from a single GPU warp to a datacenter.

## Open Questions / Follow-ups
- Exact constants and cache behaviour of the work-efficient scan vs. the hybrid two-level scan on real CPUs — measure in practice.
- How NUMA and inter-SM traffic change the cost model for cross-chunk accesses in hybrid scans.
- Assignment 3 provides block-level scan code similar to `scan_block` — worth studying to cement the warp/block/grid hierarchy.
- When is work-efficient segmented scan beaten by per-segment parallel reductions or by load-balancing tricks (e.g., merge-path)?
- Quantifying the bandwidth tax of data-parallel rewrites relative to a hand-tuned irregular implementation.

## Sources
- Slides: `slices/08_dataparallel.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Course site: https://gfxcourses.stanford.edu/cs149
- Related libraries/systems referenced: NVIDIA CUDA Thrust, JAX (`vmap`), Pandas DataFrames, Apache Spark / Hadoop, NumPy.
- Background: Guy Blelloch, *Prefix Sums and Their Applications* (work-efficient scan); Hillis & Steele, *Data Parallel Algorithms* (doubling scan).
