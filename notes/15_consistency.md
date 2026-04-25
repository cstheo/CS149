# Lecture 15 — Memory Coherence and Consistency

## Overview
Two closely related but distinct problems in shared-memory parallel systems. **Coherence** handles the fact that caches replicate memory: without a protocol, different cores can see different values at the same address. **Consistency** handles the ordering of operations to *different* addresses: how soon one thread's writes become visible to other threads, and which reorderings hardware/compilers may perform. The lecture walks through MSI/MESI snooping protocols, directory-based coherence (Intel Core i7), programmer implications (false sharing), then the consistency zoo from sequential consistency through TSO/PSO/weak ordering, memory fences, and the "SC for DRF" contract that modern languages give.

## Key Concepts
- **Coherence** — all processors agree on a single serial order of operations to each memory location; last write wins on reads.
- **Consistency** — the allowed orderings of operations to *different* addresses, as observed by other threads.
- **MSI protocol** — cache line states Invalid / Shared / Modified; snooping on a bus for `BusRd`, `BusRdX`, `BusWB`.
- **MESI protocol** — adds Exclusive (clean, single copy) to avoid a bus transaction when transitioning read-then-write on private data.
- **SWMR invariant** — Single-Writer, Multiple-Reader; at any time either one cache has the line in M, or any number have it in S.
- **Directory coherence** — per-line directory tracks which caches hold the line; replaces broadcast with point-to-point. Used by Intel Core i7 with inclusive L3 as directory.
- **False sharing** — two threads write different addresses that fall on the same cache line; line ping-pongs across caches as purely artifactual coherence traffic.
- **Sequential consistency (SC)** — some single global interleaving of all operations, consistent with each thread's program order. Maintains all four orderings WW/WR/RW/RR.
- **TSO / PSO / WO / RC** — progressively relaxed models allowing W→R, then W→W, then all reorderings, in exchange for performance.
- **Memory fence** — instruction that drains all prior memory ops before any later one proceeds. Restores ordering where programs need it.
- **SC for DRF** — data-race-free programs see sequentially consistent behavior on relaxed hardware; language standards (C11/C++11/Java 5) guarantee this contract.

## Detailed Notes

### The coherence problem
Modern processors replicate memory in per-core caches. Without a protocol, cores observe different values for the same address. This is *not* a mutual-exclusion problem — locks don't fix it, because the bug exists at the level of the cache/memory implementation itself, below the program's synchronization.

Example trace with initial `mem[X] = 0`, write-back caches:
```
Action        P1$  P2$  P3$  P4$  mem[X]
P1 load X     0 miss                 0
P2 load X     0    0 miss            0
P1 store X    1    0                 0
P3 load X     1    0    0 miss       0
P3 store X    1    0    2            0
P2 load X     1    0 hit 2           0
P1 load Y (evicts X)        2        1
```
P2 sees a stale 0 on a cache hit while P3 has the true value 2, and DRAM has neither. Clearly broken.

### Definition of coherence
A memory system is coherent if for every location X there exists a hypothetical serial order of all operations to X that (1) respects each processor's program order for X and (2) has every read return the value written by the most recent write in that order. Intuition: the cache must behave *as if it weren't there*.

### MSI snooping protocol
Three line states:
- **Invalid (I)** — not valid in this cache.
- **Shared (S)** — valid here, possibly elsewhere; memory is up to date.
- **Modified (M)** — valid here only; dirty; memory is stale.

Processor operations `PrRd` / `PrWr`; bus transactions `BusRd` (read), `BusRdX` (read with intent to modify, invalidates others), `BusWB` (write-back). Key transitions:
- `PrRd` from I: issue `BusRd`, transition to S.
- `PrWr` from I or S: issue `BusRdX`, transition to M (upgrade from S still requires a bus transaction to invalidate others).
- Snooped `BusRd` while in M: write back, downgrade to S.
- Snooped `BusRdX` while in M/S: invalidate.

Worked trace:
```
Proc action   P1   P2   P3   Bus      Data from
P1 read x      S   --   --   BusRd    Memory
P3 read x      S   --   S    BusRd    Memory
P3 write x     I   --   M    BusRdX   Memory
P1 read x      S   --   S    BusRd    P3 $ (cache-to-cache)
P1 read x      S   --   S    --       P1 $ (hit)
P2 write x     I    M   I    BusRdX   Memory
```
Why does M need to transition back to S (not stay in M forever)? Because another reader eventually shows up; a clean shared state serves multiple readers without extra coherence work.

**SWMR invariant:** at each instant, address X is either read-write owned by exactly one cache (M) or read-only shared by any number (S). The bus itself serializes all transactions, giving write-serialization.

### MESI: Exclusive state
MSI forces two transactions for the common read-then-write pattern (`BusRd` I→S, then `BusRdX` S→M) even when no other cache ever touches the line. MESI adds:
- **Exclusive (E)** — clean, but only this cache has it.

`PrRd` from I with no other sharer → E (one `BusRd` response indicates nobody else has it). `PrWr` from E → M **silently**, no bus transaction. E thus decouples exclusivity from dirtiness, eliminating the upgrade transaction on private-workload writes.

### Directory coherence
Snooping broadcasts every coherence message, which doesn't scale past tens of cores. Directory-based schemes store, per cache line, a list of which caches hold it; coherence messages go point-to-point only to those caches.

Intel Core i7: the shared L3 (banked, one bank per core, connected by a ring) is inclusive, so any line in any L2 is also in L3. The L3 therefore serves as a centralized directory, tracking which L2s hold each line and serializing coherence actions for it. Directory dimensions: P (cores, 4 in the example) × M (number of L3 lines). SWMR and write-serialization are still maintained, just without broadcast.

### AMAT in multiprocessors
Communication time shows up as higher average memory access time. Beyond uniprocessor misses, a multiprocessor adds:
- higher L1/L2 miss rates due to sharing;
- L3 hits that are cheap if the line is unshared, but expensive if it's shared or modified in another core;
- NUMA penalties for remote DRAM.

Xeon 5500 ballpark latencies:
| Level | Latency |
|---|---|
| L1 hit | ~4 cycles |
| L2 hit | ~10 cycles |
| L3 hit, unshared | ~40 cycles |
| L3 hit, shared elsewhere | ~65 cycles |
| L3 hit, modified elsewhere | ~75 cycles |
| Local DRAM | ~30 ns (~120 cycles) |
| Remote DRAM | ~100 ns (~400 cycles) |

Even a fraction of a percent of remote DRAM accesses matters.

### False sharing
Two threads write *different* addresses that happen to land on the *same* cache line. The line ping-pongs between their caches via `BusRdX`/invalidate even though there is no logical sharing. This is purely artifactual communication caused by cache-line-granularity coherence.

```c
// Bad: all counters on one or two cache lines
int myPerThreadCounter[NUM_THREADS];

// Good: pad each counter to a cache line
struct PerThreadState {
    int myPerThreadCounter;
    char padding[CACHE_LINE_SIZE - sizeof(int)];
};
PerThreadState myPerThreadCounter[NUM_THREADS];
```

Demo result on a 4-core system, 8 threads hammering their own counter:
```c
void* worker(void* arg) {
    volatile int* counter = (int*)arg;
    for (int i = 0; i < MANY_ITERATIONS; i++)
        (*counter)++;
    return NULL;
}
```
Unpadded: **14.2 s**. Padded to a full cache line: **4.7 s**. No change to logical work — only the coherence traffic changed.

The line-size tradeoff cuts both ways: larger lines capture more spatial locality but inflate false-sharing miss rates. Simulation of Barnes-Hut, Radiosity, Ocean, Radix Sort on a 1 MB cache shows false-sharing misses rising sharply as line size grows from 8 to 256 B for some workloads (especially Radix Sort).

### From coherence to consistency
Coherence is about *a single address*; consistency is about orderings across *different addresses*. Even with perfect coherence, the ordering between reads and writes to X and Y as seen by another thread is unspecified unless the model says so.

The four relevant orderings (between a pair of operations in one thread's program order):
- `WX → RY`: write to X must commit before a later read of Y.
- `RX → RY`: read of X before later read of Y.
- `RX → WY`: read of X before later write of Y.
- `WX → WY`: write of X before later write of Y.

### Sequential consistency (SC)
Lamport 1976. There is a single global interleaving of all operations such that each thread's operations appear in program order within it, and every read returns the value of the most recent write in that order. SC preserves all four orderings.

**Switch metaphor:** a hypothetical switch picks one processor at random each step, runs one of its in-order operations to completion against a single memory, then picks again. Any execution the real machine produces must be explainable by some schedule of this switch.

SC example with initial A=B=0:
```
Proc 0:    Proc 1:
(1) A=1    (3) B=1
(2) print B (4) print A
```
Under SC, printing "00" or "10" is impossible: the happens-before graph required to produce those outputs contains a cycle (an event would need to happen before itself).

### Why hardware relaxes SC: write buffers
A cache-coherent write can take 100s of cycles (find the line, send invalidations, wait). Under pure SC, a processor must stall until its write completes before issuing the next operation. Every modern CPU adds a **write buffer**: stores go into the buffer and retire quickly; the core issues the following load immediately.

Consequence: the core can read Y before its own write to X has become visible. For the canonical A=B=0 example above, two cores' write buffers can both hold the new value locally while each reads the other's old value from memory, producing `r1 = r2 = 0`. SC forbids this; write buffers allow it.

### Total Store Ordering (TSO) and Processor Consistency (PC)
- **TSO** (x86-ish): a processor may read a later address before its own earlier write is globally visible (its own `WX→RY` is relaxed, but only *within the same processor*'s view). Other processors still cannot see the new value of X until the write is observed by all. Writes from one processor are serialized among themselves (`WW` preserved).
- **PC**: any processor may see the new value of A before all processors do. Weaker than TSO; still preserves WW within a processor.

Both relax only `WX→RY`. Sun SPARC TSO and x86 (a partly-specified TSO variant) are in this family. The performance gain is large because write latency is almost fully hidden.

### Partial Store Ordering (PSO) and weak models
**PSO** additionally relaxes `WX→WY`: writes by the same thread can reach memory out of program order (e.g., one is a cache miss, the other a hit). The classic flag-handoff becomes broken without fences:
```c
// Thread 1 on P1        // Thread 2 on P2
A = 1;                  while (flag == 0);
flag = 1;               print A;       // may print 0!
```
Under PSO, P2 may observe `flag=1` before `A=1`.

**Weak Ordering (WO)** and **Release Consistency (RC)** relax all four orderings — every reordering is legal unless the programmer uses an explicit synchronization instruction. ARM and POWER are in this family. Motivation: maximum freedom to overlap loads/writes to hide memory latency.

Why would reorderings actually happen in hardware? `WW` can be reordered in a write buffer when a hit passes a miss; `RW`/`RR` can be reordered by ordinary out-of-order execution. All of these are legal uniprocessor optimizations; they only become visible in multiprocessor programs with data races.

### Synchronization primitives
Every relaxed architecture exposes **fences / memory barriers**: instructions that force all prior memory operations to complete before any later one starts. On x86:
- `mm_lfence` — wait for prior loads.
- `mm_sfence` — wait for prior stores.
- `mm_mfence` — wait for all prior memory ops.

Richer primitives: atomic read-modify-write / compare-and-swap, transactional memory, load-acquire / store-release pairs (C11/C++11). These only pay the ordering cost at synchronization points, leaving ordinary data accesses unconstrained.

### Data-race-free programs
A **conflict** is two accesses to the same location by different processors, at least one being a write. A program has a **data race** if two conflicting accesses are not ordered by any synchronization operation. Racy programs have nondeterministic results depending on processor speeds; they are almost always buggy.

**SC for DRF:** if a program is data-race-free (all conflicting accesses separated by locks/barriers/fences), then on any of TSO/PSO/WO/RC it will produce results consistent with some SC execution. Intuition: the only places where reorderings matter are conflicting races, and DRF programs have none.

### Language-level memory models
Compilers reorder too (register allocation, common-subexpression elimination, hoisting). A language needs its own memory model to give programmers a contract. C11, C++11, and Java 5 guarantee SC for DRF and give undefined/weakly-defined behavior for races. In practice: use a synchronization library (mutexes, atomics, condition variables); the library inserts the right fences for the target ISA so you don't.

### Clarification
Coherence arises from **replicating** data across caches. Consistency arises from **reordering** memory operations. The two are orthogonal — consistency would still be an issue on a system with no caches at all, because reordering happens in write buffers, out-of-order pipelines, and compilers independently.

## Examples / Worked Problems
- MSI trace on x across P1/P2/P3 showing S/M/I transitions and cache-to-cache transfers (above).
- False-sharing demo: unpadded counters 14.2 s vs cache-line-padded 4.7 s on 4-core, 8-thread workload.
- SC happens-before analysis of the A=B=0 two-thread example: "00" and "10" impossible under SC; "00" possible once write buffers exist.
- PSO flag-handoff: `A=1; flag=1;` on P1 followed by `while(flag==0); print A;` on P2 can print 0 without an `sfence` between the two stores on P1 (or an `lfence` after the spin on P2).

## Takeaways
1. Coherence ≠ consistency. Coherence makes caches invisible (single-address abstraction); consistency defines cross-address ordering.
2. Snooping protocols (MSI/MESI) maintain a Single-Writer-Multiple-Reader invariant with bus-serialized transactions; MESI's Exclusive state eliminates a needless upgrade for private data.
3. Snooping doesn't scale — broadcast grows with core count. Directories (e.g., Intel's inclusive L3) replace broadcast with targeted messages.
4. False sharing is a purely software-visible consequence of coherence granularity. Pad thread-private data to cache-line size.
5. SC is intuitive but too strict for hardware — write buffers alone break it. TSO relaxes W→R, PSO adds W→W, WO/RC relax everything.
6. Fences and synchronization primitives restore ordering at explicit points. Data-race-free programs see SC behavior on any reasonable relaxed model — so use a synchronization library and keep your programs DRF.

## Open Questions / Follow-ups
- Lock-free data structures and correct use of release/acquire atomics — deferred to a later lecture.
- Formal semantics of x86-TSO and ARM's very relaxed model — see Milewski's x86 fences post and the Cambridge weak-memory bibliography linked in-class.
- How directories handle E-state, silent evictions, and three-hop protocols — scalability details beyond the simplified i7 picture.
- Interaction of transactional memory with coherence and consistency — mentioned only briefly.

## Sources
- Slides: `slices/15_consistency.pdf` (Stanford CS149, Fall 2025).
- Raw transcript: `notes/.raw/15_consistency.txt`.
- Referenced: Lamport, "How to Make a Multiprocessor Computer That Correctly Executes Multiprocess Programs" (1979); Culler, Singh, Gupta, *Parallel Computer Architecture*; Bartosz Milewski, "Who Ordered Memory Fences on an x86?" (2008); ARM Barrier Litmus Tests and Cookbook (A08); Cambridge weak-memory papers (https://www.cl.cam.ac.uk/~pes20/weakmemory/).
