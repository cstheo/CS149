# Lecture 18 — Transactional Memory (Part 2) and Course Wrap-Up

## Overview
Picks up where lecture 17 left off: having motivated the `atomic {...}` construct, this lecture dives into *how* transactional memory (TM) is actually implemented. Two design axes — **data versioning** (eager vs. lazy) and **conflict detection** (pessimistic vs. optimistic, plus granularity) — define the implementation space. We walk through a full software TM (STM) algorithm based on Intel's McRT, analyze why STM is slow (read-barrier overhead), and then see how hardware TM (HTM) reuses the cache and coherence protocol to get versioning and conflict detection almost for free. Finishes with Intel Haswell's RTM instructions and a short course wrap-up.

## Key Concepts
- **Memory transaction** — atomic + isolated sequence of memory accesses; commits as a unit or aborts with no visible effect (review from lecture 17).
- **Data versioning** — how uncommitted (new) and committed (old) values coexist.
  - **Eager / undo-log**: write new value in place, log old value for rollback.
  - **Lazy / write-buffer**: buffer new value off to the side, write through to memory on commit.
- **Conflict detection** — how/when the system notices two transactions touch the same address such that their ordering matters.
  - **Pessimistic**: check on every load/store; a *contention manager* decides stall vs. abort.
  - **Optimistic**: check only at commit; committing transaction wins, others may abort.
- **Read set / write set** — addresses a transaction has read / written; required for conflict detection.
- **Detection granularity** — object, word/field, or cache line. Tradeoff: overhead vs. false conflicts.
- **Transaction descriptor** — per-thread record holding read set, write set, undo log / write buffer, status.
- **Transaction record (TxR)** — per-datum guard word storing version number or owner-lock pointer.
- **STM barrier** — compiler-inserted instrumentation (`tmRd`, `tmWr`, `tmTxnBegin/Commit`) around transactional accesses.
- **HTM** — versioning lives in the cache; conflict detection rides on the coherence protocol via R/W bits per cache line.
- **RTM (Haswell)** — `xbegin`/`xend`/`xabort`; L1-tracked, best-effort, with mandatory fallback path.

## Detailed Notes

### TM review
A transaction provides **atomicity** (all-or-nothing on commit/abort), **isolation** (nobody sees partial state), and **serializability** (transactions appear to occur in *some* serial order — the order itself is not specified). Promise: coarse-lock simplicity with fine-lock performance, plus failure atomicity (no lost locks on thread failure) and composability (nested atomic blocks compose safely).

### Two implementation knobs

1. **Data versioning**
   - *Eager*: store new value directly into memory, push the old value into an undo log. Fast commit (just drop the log); abort has to walk the log to undo.
   - *Lazy*: keep new value in a private write buffer. Fast abort (discard buffer); commit has to write the buffer out atomically.

2. **Conflict detection**
   - Track read-set and write-set per transaction. Conflicts = RW (A reads address X that pending B wrote) or WW (both write X).
   - *Pessimistic*: check on every access. Philosophy — "if I'm going to roll back, may as well find out early." Contention manager arbitrates (typical rule: writer wins).
   - *Optimistic*: check only at commit. Philosophy — "assume no conflict, sort it out at the end." Committing txn wins.

### Pessimistic vs. optimistic — the four cases
The slides show 4 time-diagrams each. Assume aggressive contention management (writer wins, reader loses).

Pessimistic, with an "aggressive" writer-wins manager:
1. **Case 1 — success**: disjoint addresses; checks after each op succeed; both commit.
2. **Case 2 — early detect / stall**: T1 writes A while T0 has A in read set; check on T0 stalls T0 until T1 commits, then T0 proceeds.
3. **Case 3 — abort**: T0 reads A, then T1 writes A; writer wins, T0 restarts.
4. **Case 4 — no forward progress (livelock)**: T0 and T1 alternate writes to A, each aborting the other. Motivates contention-manager policies for progress.

Optimistic:
1. **Success** (disjoint), 2. **Abort** at commit (T0 commits first after both read A and T1 wrote A — T1 aborts), 3. **Success** (conflict detected and loser retries cleanly), 4. **Forward progress** where optimistic mode avoids the pessimistic livelock.

Summary: optimistic tends to make forward progress but wastes work on abort; pessimistic detects early but can deadlock/livelock on contention.

### Design-space examples
- **Software TM**
  - Lazy + optimistic RW: Sun TL2
  - Lazy + optimistic-R / pessimistic-W: MS OSTM
  - Eager + optimistic-R / pessimistic-W: Intel STM (McRT variant)
  - Eager + fully pessimistic: Intel STM
- **Hardware TM**
  - Lazy + optimistic: Stanford TCC
  - Lazy + pessimistic: MIT LTM, Intel VTM
  - Eager + pessimistic: Wisconsin LogTM (easiest with standard coherence)

No single design has won; optimal choice depends on SW vs. HW vs. hybrid.

### Software TM (STM)
The compiler rewrites an `atomic { ... }` block into explicit calls:

```
atomic {                     tmTxnBegin()
  a.x = t1                   tmWr(&a.x, t1)
  a.y = t2                   tmWr(&a.y, t2)
  if (a.z == 0) {            if (tmRd(&a.z) != 0) {   // note: compiler flips sense
    a.x = 0                     tmWr(&a.x, 0)
    a.z = t3                    tmWr(&a.z, t3)
  }                          }
}                            tmTxnCommit()
```

Because the same function can be called inside and outside a transaction, STM requires **function cloning** (two versions: instrumented + bare) or dynamic translation.

**Runtime data**:
- **Transaction descriptor** (per thread): read set, write set, undo log or write buffer, status. Used at commit/abort/conflict check.
- **Transaction record** (per datum): one pointer-sized word guarding shared data, either a version number / shared-reader lock (shared state) or a pointer to the owner transaction (exclusive/locked). Structurally identical to how HW cache coherence tracks line ownership.

**Mapping data → TxR** (see the slide diagram):
- Java/C# object: either embed a TxR field in the object header, or hash the object identity into a global TxR table. Can also key on `(obj.hash, field.index)`.
- C/C++: no object header — address-based hash into a global TxR table, at cache-line or word granularity.

**Granularity tradeoffs**:
- *Object*: cheap mapping, exposes optimizations, but false conflicts (two txns touching different fields of the same object conflict).
- *Word/field*: fewer false conflicts, more concurrency, but higher time/space overhead.
- *Cache line*: matches HTM, cheap storage, harder to reason about.
- Real systems mix (array-element for arrays, object-level for scalars).

### An example STM algorithm (Intel McRT: eager versioning, optimistic reads, pessimistic writes)

Timestamp-based versioning:
- **Global timestamp** — incremented when any writing transaction commits.
- **Local timestamp** — value of the global clock when this txn last validated its reads.
- **TxR layout** (32 bits): LSB = lock bit (0 = writer-locked, 1 = unlocked). Upper bits are the version/commit timestamp when unlocked, or a pointer to the owning transaction when locked.

**Read (optimistic)**:
1. Read the memory location directly (eager-ish — no indirection).
2. Validate: is the TxR unlocked and is its version ≤ local timestamp?
3. If not, re-validate the entire read set for consistency.
4. Insert into read set; return value.

**Write (pessimistic)**:
1. Validate: unlocked? version ≤ local timestamp?
2. Acquire the writer lock.
3. Insert into write set.
4. Push undo-log entry.
5. Write in place.

**Read-set validation**:
- Snapshot the global timestamp.
- For each item in the read set: if locked by someone else or version > local timestamp, abort.
- Set local timestamp = snapshotted global timestamp.

**Commit**:
- Atomically increment global timestamp by 2 (LSB is reserved for the write lock bit, so increments of 2 preserve lock semantics).
- If the pre-incremented global timestamp already exceeded local timestamp, re-validate the read set (someone committed concurrently).
- For each item in the write set: release lock and set version to the new global timestamp.

**STM worked example** (from the slides):
```
foo={hdr=3, x=9, y=7}        bar={hdr=5, x=0, y=0}

X1: atomic {                 X2: atomic {
      t = foo.x;                   t1 = bar.x;
      bar.x = t;                   t2 = bar.y;
      t = foo.y;              }
      bar.y = t; }
```
X1 is copying `foo` into `bar`. X2 must observe `bar` as either `[0,0]` or `[9,7]` — never the torn mid-state `[9,0]`.

Execution: X1 locks `bar` (its TxR header flips to "locked, owner=X1"), does eager in-place writes logged in the undo log (`<bar.x,0>, <bar.y,0>`). X2 reads `bar.x` — sees it locked by X1 → **abort** (or wait, depending on contention policy). X1 commits, version bumps (`bar.hdr` goes from 5 to 7). X2 restarts and now reads `[9,7]`. Invariant preserved.

### Why STM is slow
Measured single-thread overhead on `kmeans` and `vacation` is **1.8×–5.6×** over sequential. Most time is in **read barriers** and **commit** (most apps read more than they write). The Vacation speedup curve shows STM trailing ideal by a large margin: 2–8× per-thread overhead.

Two implications:
- **Short term**: demotivates parallel programming — the tax is too high.
- **Long term**: it's energy-wasteful instrumentation.

Plus strong atomicity (transactional and non-transactional accesses interacting correctly) is very expensive purely in software.

### Optimizing STM (compiler)
Monolithic barriers (`tmWr(&a.x, t1)` as one call) hide redundancy from the compiler. Decomposing into `txnOpenForWrite(a); txnLogObjectInt(&a.x, a); a.x = t1;` lets the compiler CSE the `txnOpenForWrite(a)` across the three writes to `a.x`, `a.y`, `a.z`, and similarly collapse logging. After optimization: overhead drops to <40% vs. no concurrency control and <30% vs. lock-based sync on 1 thread — still a tax, but much more palatable.

### Hardware TM (HTM)
Two insights:
1. **Cache already does versioning.** An M-state dirty line is a write-buffer entry; an undo log is a second cache write.
2. **Coherence already does conflict detection.** Remote coherence requests to a line you've read or written *are* the conflicts.

**Cache line annotations** — add to each line:
- `R` bit: line is in this transaction's read set (set by loads).
- `W` bit: line is in this transaction's write set (set by stores).
- On commit or abort, gang-reset R and W bits for all lines.
- Granularity choice: word-level or line-level R/W bits.

**Coherence-based conflict detection**:
- Another core issues a shared/read request for a line whose W bit is set here → RW conflict.
- Another core issues exclusive/upgrade for a line whose R bit is set here → WR conflict.
- Another core issues exclusive for a line whose W bit is set here → WW conflict.

**CPU additions**:
- Register checkpoint at `Xbegin` so context can be restored on abort.
- TM-state registers (status, abort handler pointer).

**Lazy-optimistic HTM walkthrough** (from slides):
```
Xbegin
  Load A       ; R=1 on line A (miss serviced normally)
  Load B       ; R=1 on line B
  Store C ← 5  ; W=1 on line C, value 5 written locally
               ; NOTE: not loaded into exclusive state yet — lazy
Xcommit
```
On commit (two-phase, fast):
1. *Validate*: request exclusive (RdX / upgradeX) on each W-line. If any remote core holds it, conflict → abort.
2. *Commit*: gang-reset R and W bits; W-lines transition to M (dirty-valid). Atomic because no remote can steal those lines mid-sequence.

On remote commit that touches our set:
- Remote `upgradeX A` arrives; A has R=1 here → **abort**: invalidate W-set lines, gang-reset R/W, restore register checkpoint.

**HTM performance**: 2×–7× over STM; within 10% of sequential on one thread; scales nearly ideally on Vacation out to 16 processors.

### Intel Haswell RTM (Restricted Transactional Memory)
- Instructions: `xbegin <fallback-addr>`, `xend`, `xabort`.
- L1 tracks the read/write set. Line eviction from L1 — or a context switch, system call, page fault, many events — forces an abort.
- Progress is *not* guaranteed: programmers must supply a fallback path (typically a spinlock acquisition) at the `xbegin` target.
- Intel optimization guide ch. 12 gives guidance on keeping working sets L1-resident and avoiding abort-inducing instructions.

### HTM example: TCC (Transactional Coherence and Consistency)
TCC replaces ordinary cache coherence with "always-transactional" semantics: every execution is inside some transaction, commits serialize globally, and a successful commit broadcasts updates to all caches.

Worked example on P1/P2/P3 (lazy + optimistic, one commit per step across the system):
```
P1: B T1; R A; W A,1; W C,2; R D; C T1
P2: B T2; R A; W E,3; C T2; B T3; W C,4/5; R A; W E,5/6; C T3
P3: B T4; R E; W B,6; W C,7; R F; C T4
```
Evolution of read/write sets (selected steps):
- After initial ops: P1 read-set {A}, write-set {A:1, C:2}; P2 read-set {A}, write-set {E:3}; P3 read-set {E}, write-set {B:6}.
- P2 commits T2 (writes E). P1's read of A doesn't conflict with P2's write to E → P1 keeps going.
- P1 commits T1 (writes A,C). P3's T4 has read-set {E, F} — no conflict. P2's new T3 currently has empty read-set, still fine.
- P3 starts writing B,C,F-reading; commits T4 with RS {E:3, F:0}, WS {B:6, C:7}.
- P2's T3 had read A and is writing C,E. When it tries to commit, it must serialize against whatever other commits happened. If T4 already committed with C:7, T3's C:4/5 write just overwrites on commit (no read-set conflict on C because T3 didn't read C). If P1 had committed A:1 after T3 read A:0, T3 would abort on commit-time validation of its read set.

The exercise reinforces: with optimistic detection, conflicts are resolved at commit; transactions build up per-processor R/W sets; ordering is imposed by a globally serial commit.

### Course wrap-up
Kayvon's closing themes:
- Performance now comes from parallelism + specialization — CPU cores, integrated GPUs, FPGAs, TPUs, AI accelerators, heterogeneous SoCs like Apple A11.
- Modern software routinely leaves most of a machine's performance on the table; closing that gap matters for compute-intensive and new applications.
- Three through-lines of CS149:
  1. **Identifying parallelism** (and, dually, dependencies).
  2. **Efficient scheduling** — workload balance, communication (bandwidth, latency, synchronization).
  3. **Locality** — efficient state management across memory/cache/network hierarchies.
- Covered scales: mobile SoC, multicore CPU, GPU, CPU+GPU, clusters, AI accelerators.
- Covered abstractions: data-parallel, functional-parallel, transactions, tasks, SPMD.
- Next-step courses at Stanford: CS 217 (HW accelerators for ML, Olukotun), CS/EE 282 (computer systems architecture), CS 348K (visual computing systems, Fatahalian).

## Examples / Worked Problems

### STM question (optimistic read, pessimistic write, eager versioning)
Implement `atomic { obj.f1 = 42; }`:
```
tx = GetTxDescriptor();
OpenForWriteTx(tx, obj);            // validate + acquire writer lock on obj's TxR
LogForUndoIntTx(tx, obj, offset);   // record old value of obj.f1 in undo log
obj.f1 = 42;                        // eager in-place write
// tmTxnCommit() later: release lock, bump version to new global timestamp
```

### STM copy-object example
X1 copies `foo=[9,7]` into `bar`; concurrent X2 reads `bar.x`/`bar.y`. Because X1 takes the writer lock on `bar` before its first in-place write, X2's validated read sees `bar` locked → X2 aborts and retries after X1's commit bumps `bar`'s version. Result for X2 is either `[0,0]` (if it completed before X1 locked) or `[9,7]` (after X1 committed) — never `[9,0]`.

### HTM abort walkthrough
Local transaction has R={A,B}, W={C}. Remote core commits writes to A and D:
- `upgradeX A` arrives → line A has R=1 → local transaction aborts: invalidate C (discard speculative value), gang-reset R/W bits on A,B,C, restore register checkpoint to `Xbegin`.
- `upgradeX D` arrives → D isn't in R or W locally → no effect.

### Pessimistic livelock (Case 4)
Two transactions repeatedly write A. With writer-wins contention, each abort of the loser triggers its restart, which then writes A and aborts the other. Without a fairness mechanism (backoff, priority aging, random delay), neither ever commits. Motivates robust contention managers.

## Takeaways
1. TM implementations are characterized by two orthogonal choices — versioning (eager/lazy) and conflict detection (pessimistic/optimistic) — plus a granularity choice.
2. STM is conceptually clean but slow (1.8×–5.6× single-thread overhead), dominated by read barriers; compiler optimization of decomposed barriers recovers much of the loss but not all.
3. HTM rides the existing cache + coherence machinery: cache = version buffer, coherence messages = conflict signals. Result: near-ideal one-thread performance and good scaling.
4. Real HTM (Haswell RTM) is *best-effort* — L1 capacity and countless microarchitectural events can force aborts, so a software fallback path is mandatory.
5. The CS149 arc — identify parallelism, schedule efficiently, exploit locality — applies from SIMD lanes to datacenter clusters and specialized accelerators.

## Open Questions / Follow-ups
- How to avoid pessimistic-TM livelock in practice — which contention-manager policies (timestamp, priority aging, Polka, Greedy) give both fairness and throughput?
- Strong vs. weak atomicity — cost/benefit of making non-transactional accesses see in-flight transactional state consistently.
- Hybrid TM systems (Sun Rock) — decision rules for switching between HW and SW modes.
- Intel Haswell RTM in anger — what programming patterns keep abort rates low enough to beat a well-tuned spinlock?
- Optimal HTM design point (versioning × detection × granularity) — still open research per the lecture.

## Sources
- Slides: `slices/18_transactionalmem.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Raw lecture text: `notes/.raw/18_transactionalmem.txt`.
- Prior lecture: `notes/17_transactionalmem.md` (TM motivation, semantics, examples).
- Referenced systems: Intel McRT STM (PPoPP'06, PLDI'06, CGO'07); Sun TL2; MS OSTM; Stanford TCC; MIT LTM; Intel VTM; Wisconsin LogTM; Sun Rock (hybrid); Intel Haswell RTM (Intel Optimization Guide ch. 12).
