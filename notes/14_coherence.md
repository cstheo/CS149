# Lecture 14 — Cache Coherence

## Overview
Shared-memory multiprocessors give the programmer the illusion of a single address space, but each core has its own private cache. Replicating data in multiple caches creates the **cache coherence problem**: without help, different processors can observe different values for the same memory location. This lecture defines coherence formally, explains how hardware maintains it via snooping-based invalidation protocols (MSI, MESI), mentions directory-based approaches for scaling beyond buses, and shows how coherence leaks back into software performance through **false sharing**.

## Key Concepts
- **Coherence problem** — private write-back caches hold replicas of the same memory line; a write by one core can leave stale copies in other caches and in memory.
- **Coherence (definition)** — for each address, there exists a hypothetical serial order of all operations consistent with program order per processor, and every read returns the value of the last write in that order.
- **SWMR invariant** — Single-Writer, Multiple-Reader: every epoch is either one writer or many readers, never both.
- **Data-value invariant** — the value at the start of a new epoch equals the value at the end of the previous read-write epoch.
- **Snooping** — all cache controllers observe a shared interconnect (bus) and react to coherence messages.
- **MSI / MESI** — invalidation-based write-back protocols with 3 or 4 line states.
- **Directory-based coherence** — scalable alternative: a directory records which caches hold each line, so messages are unicast instead of broadcast.
- **False sharing** — two processors write to distinct variables that fall on the same cache line, causing the line to ping-pong through the coherence protocol.

## Detailed Notes

### Why coherence is a hardware problem, not a locking problem
Consider `int foo` at address X initially 0, with four cores and write-back caches:

| Action | P1 $ | P2 $ | P3 $ | P4 $ | mem[X] |
|---|---|---|---|---|---|
| P1 load X (miss) | 0 | | | | 0 |
| P2 load X (miss) | 0 | 0 | | | 0 |
| P1 store X ← 1 | 1 | 0 | | | 0 |
| P3 load X (miss) | 1 | 0 | 0 | | 0 |
| P3 store X ← 2 | 1 | 0 | 2 | | 0 |
| P2 load X (hit!) | 1 | 0 | 2 | | 0 |
| P1 load Y (evicts X, writeback) | — | 0 | 2 | | 1 |

P2's "hit" returns the stale 0 it cached earlier. This is *not* a mutual-exclusion problem — locks around the shared variable would not fix it, because the root cause is replicated storage. It must be solved in hardware (or by coarse-grained software schemes such as OS page-fault replication over clusters, which we skip).

### Formal definition of coherence
A memory system is coherent iff, for every address, there is a hypothetical serial order of all program operations on that address such that:
1. Each processor's operations appear in program order.
2. Every read returns the value written by the most recent write in the serial order.

Equivalent implementation invariants:
- **SWMR** — at any time a line is either in a single read-write epoch (one writer, possibly reading) or a read-only epoch (zero or more readers).
- **Write serialization / data-value** — the value visible at the start of an epoch equals the value at the end of the previous read-write epoch.

Diagram (in prose): a timeline for address x partitions into alternating epochs labeled "Read-Write P0", "Read-Only {P0,P1,P2}", "Read-Write P1", "Read-Only {P0,P1}". Only one writer per epoch; between writers the value is frozen and may be shared.

### Shared caches as a baseline
One giant shared cache serving all cores trivially solves coherence (no replication) and even enables constructive sharing (one core's load prefetches for another). But a cache must be physically close to its core; a monolithic shared cache suffers conflict and contention. SUN Niagara 2 (UltraSPARC T2) used a crossbar-connected shared L2 — the crossbar alone was about the area of a full core.

### Snooping: the broadcast approach
Every cache controller listens to every coherence transaction on the interconnect. It reacts to two event sources:
1. Local CPU loads/stores.
2. Messages broadcast on the bus by other caches.

**Naïve write-through scheme** — on every write, broadcast an invalidation; because writes go to memory, any later miss in another cache pulls fresh data:

| Action | Bus | P0 $ | P1 $ | mem[X] |
|---|---|---|---|---|
| P0 load X | miss | 0 | | 0 |
| P1 load X | miss | 0 | 0 | 0 |
| P0 write X ← 100 | invalidate | 100 | — | 100 |
| P1 load X | miss | 100 | 100 | 100 |

Correct but wasteful — every store hits memory. Real systems use write-back caches that only evict dirty lines on capacity pressure, so coherence protocols get more subtle.

### MSI invalidation protocol (write-back)
Three line states:
- **I (Invalid)** — not present.
- **S (Shared)** — valid in ≥1 cache, memory up to date.
- **M (Modified)** — valid in exactly one cache, memory stale; this cache is the "owner" and must supply the line to others.

Processor actions: `PrRd`, `PrWr`. Bus transactions: `BusRd` (read copy), `BusRdX` (read with intent to modify — invalidates others), `BusWB` (write back dirty line).

State transitions (notation `A / B` = on observing A, take action B):
- **I → S** on `PrRd / BusRd`.
- **I → M** on `PrWr / BusRdX`.
- **S → M** on `PrWr / BusRdX` (an "upgrade"; still needs the bus so others invalidate).
- **S → I** on snooped `BusRdX / --`.
- **M → S** on snooped `BusRd / BusWB` (owner supplies data and writes back).
- **M → I** on snooped `BusRdX / BusWB`.
- **M, S**: local `PrRd` is a silent hit; `PrWr` on M is silent.

Example trace on one line for address x:

| Action | P1 | P2 | P3 | Bus | Data source |
|---|---|---|---|---|---|
| P1 read x | S | — | — | BusRd | Memory |
| P3 read x | S | — | S | BusRd | Memory |
| P3 write x | I | — | M | BusRdX | Memory |
| P1 read x | S | — | S | BusRd | P3 $ (flushes) |
| P1 read x | S | — | S | — | P1 $ (hit) |
| P2 write x | I | M | I | BusRdX | Memory |

Why MSI satisfies coherence:
- SWMR — at most one cache in M; all others invalidated on BusRdX. Multiple caches may sit in S.
- Write serialization — the bus linearizes all transactions; on BusRd/BusRdX, a cache in M supplies the current value.

### MESI: add an "Exclusive clean" state
MSI pays two bus transactions for the common pattern "load then store" of an unshared line: a BusRd for I → S and a BusRdX for S → M, even when no other cache ever had the line.

**MESI** adds:
- **E (Exclusive clean)** — line valid, memory also valid, but this is the only cache with a copy.

On `PrRd` miss, the requesting controller checks whether any other cache asserts the "shared" signal:
- No sharers → load into **E** instead of S.
- Sharers → load into **S** as before.

Then `E → M` on local `PrWr / --` — no bus traffic, because nobody else has the line. If a remote cache does a BusRd, E demotes to S; on BusRdX, to I. This eliminates the wasted upgrade transaction in the no-sharing case (which is the common case for thread-local data).

### Scaling beyond snooping: directories
Snooping requires broadcasting every coherence event to every cache — bandwidth scales with #cores × event rate. A **directory** stores, per cache line, which caches currently hold it and in what state. Coherence messages are then point-to-point to exactly the caches that care.

Intel Core i7 uses the **inclusive shared L3** as the directory. Because every L2 line is also in L3, each L3 line can track which cores' L2s hold it. The ring interconnect carries unicast messages instead of broadcasts. Directory dimensions scale as P (cores) × M (L3 lines). Directories still enforce SWMR and write serialization, just without broadcast.

### Cost of coherence-induced misses (Core i7 Xeon 5500)
Approximate latencies:

| Source | Latency |
|---|---|
| L1 hit | ~4 cycles |
| L2 hit | ~10 cycles |
| L3 hit, unshared | ~40 cycles |
| L3 hit, shared line in another core | ~65 cycles |
| L3 hit, modified in another core | ~75 cycles |
| Local DRAM | ~30 ns (~120 cycles) |
| Remote DRAM (NUMA) | ~100 ns (~400 cycles) |

AMAT on a multiprocessor exceeds the uniprocessor AMAT because of extra coherence misses (communication misses) and NUMA effects. Even a fraction of a percent shift in miss rate matters.

### False sharing: coherence leaks into software
Two threads writing to *different* integers that happen to sit on the same 64-byte cache line cause the line to ping-pong between their caches. No logical communication, but the hardware sees a stream of BusRdX transactions.

```c
// BAD: counters share a cache line
int myPerThreadCounter[NUM_THREADS];

// GOOD: pad each counter to a full cache line
struct PerThreadState {
    int myPerThreadCounter;
    char padding[CACHE_LINE_SIZE - sizeof(int)];
};
PerThreadState myPerThreadCounter[NUM_THREADS];
```

Live demo (8 threads, 4-core box, each incrementing a counter in a tight loop):

```c
void* worker(void* arg) {
    volatile int* counter = (int*)arg;
    for (int i = 0; i < MANY_ITERATIONS; i++)
        (*counter)++;
    return NULL;
}
```

- Unpadded `int counter[MAX_THREADS]` → 14.2 s.
- Padded `padded_t counter[MAX_THREADS]` (one counter per line) → 4.7 s.

~3× slowdown from purely artifactual communication.

Cache-line-size sensitivity (Culler/Singh/Gupta simulation of a 1 MB cache on Barnes-Hut, Radiosity, Ocean Sim, Radix Sort): as line size grows from 8 B to 256 B, cold and capacity misses fall (spatial locality), but *false-sharing* misses rise — especially in apps like Radix Sort with dense fine-grained writes. Line size is a design compromise.

## Examples / Worked Problems
- **Coherence failure trace** (table above): without a protocol, P2 reads a stale hit of 0 while P1 and P3 have written newer values. Demonstrates why locks alone can't fix it.
- **Write-through + invalidate trace**: simple but bandwidth-hostile.
- **MSI trace** for P1/P2/P3 sharing address x: shows M→S transitions with BusWB, upgrades, and ownership handoff.
- **MSI vs MESI** on unshared load-then-store: MSI uses BusRd + BusRdX (two transactions); MESI uses BusRd landing in E and a silent E→M (one transaction).
- **False-sharing demo**: unpadded vs padded per-thread counters, 14.2 s vs 4.7 s.

## Takeaways
1. Coherence is a hardware responsibility created by *replication*. Locks sit on top of coherence; they don't substitute for it.
2. Coherence = SWMR + write serialization. A protocol is correct iff it preserves those invariants for every line, at all times.
3. Snooping (broadcast) MSI/MESI is the standard for bus/ring-scale CMPs; directories (e.g., inclusive L3 in Core i7) replace broadcast with unicast to scale further.
4. MESI's E state exists specifically to eliminate the wasted upgrade transaction for thread-local, load-then-store patterns — a huge fraction of real code.
5. Coherence surfaces in software performance as **communication misses** (real sharing) and **false sharing** (artifactual). Pad or group your per-thread data to cache-line granularity.
6. Cache-line size is a knob: bigger lines help spatial locality and cold/capacity miss rates but amplify false sharing.

## Open Questions / Follow-ups
- Memory *consistency* (program-level ordering across addresses) vs coherence (per-address) — separate lecture.
- MOESI / MESIF variants used by AMD and Intel in practice; owner-state optimizations for cache-to-cache transfers.
- Directory state overheads and alternatives (sparse directories, cuckoo directories) when P or M grow large.
- Software techniques: VTune counters for coherence misses, `alignas(std::hardware_destructive_interference_size)` in C++17.
- Non-coherent shared memory (GPUs across SMs, some NUMA/accelerator fabrics) — how programs cope without hardware coherence.

## Sources
- Slides: `slices/14_coherence.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Raw lecture transcript: `notes/.raw/14_coherence.txt`.
- Intel 64 and IA-32 Architectures Optimization Reference Manual (June 2016) — Skylake/Core i7 cache hierarchy numbers.
- Culler, Singh, and Gupta, *Parallel Computer Architecture* — cache-line-size miss-rate figures.
