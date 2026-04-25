# Lecture 17 — Transactional Memory (Part 1)

## Overview
Raises the level of abstraction for synchronization one more step. Prior lectures built locks, barriers, and lock-free structures on top of machine-level atomics (test-and-set, CAS, LL/SC), and saw how easy it is to get those wrong. Transactional memory (TM) replaces lock/unlock with a declarative `atomic { ... }` block: the programmer states *what* must happen atomically, and the system decides *how*. This lecture motivates TM (composability, failure atomicity, performance portability), pins down its semantics (atomicity, isolation, serializability), and opens up the implementation design space: data versioning (eager vs lazy) and conflict detection (pessimistic vs optimistic). Hardware and software implementations themselves are in Part 2.

## Key Concepts
- **Memory transaction** — an atomic and isolated sequence of memory accesses, inspired by database transactions.
- **Atomicity** — all-or-nothing: on commit, every write takes effect at once; on abort, no writes are visible.
- **Isolation** — no other processor observes a transaction's writes until commit.
- **Serializability** — transactions appear to commit in some single serial order (the exact order is not specified).
- **`atomic { }` vs `lock/unlock`** — `atomic` is declarative (intent); locks are an imperative blocking primitive (mechanism). They are *not* interchangeable.
- **Data versioning** — how new (uncommitted) and old (committed) values are kept side-by-side. Eager = write memory now, keep an undo log. Lazy = buffer writes, flush on commit.
- **Conflict detection** — deciding when two concurrent transactions touch the same location in a read/write or write/write conflict. Pessimistic = check on every load/store; Optimistic = check at commit.
- **Granularity of detection** — the unit (word, cache line, object) at which the system tracks the read-set and write-set.

## Detailed Notes

### Between a lock and a hard place
Locks force a trade-off between concurrency and correctness:
- **Coarse-grained locking** (e.g., one lock for the whole data structure) — easy to reason about, low concurrency.
- **Fine-grained locking** (e.g., one lock per bucket, hand-over-hand) — more concurrency, much easier to create races and deadlocks.

Measured on a HashMap and a balanced tree, fine locks scale noticeably better than coarse locks from 1 to 16 processors. But writing correct fine-grained locking code is hard, and the "right" locking scheme for 4 cores may not be right for 64. TM is proposed as a better abstraction.

### From locks to atomic blocks
The canonical bank-deposit example:
```c
void deposit(Acct account, int amount) {            void deposit(Acct account, int amount) {
   lock(account.lock);                                 atomic {
   int tmp = bank.get(account);                          int tmp = bank.get(account);
   tmp += amount;                                        tmp += amount;
   bank.put(account, tmp);                               bank.put(account, tmp);
   unlock(account.lock);                               }
}                                                   }
```
The right-hand version declares *what* (this code is atomic). The system chooses *how* — possibly with locks under the hood, but the lecture focuses on **optimistic concurrency**: serialize only when true read-write or write-write conflicts occur.

### Declarative vs imperative
- **Declarative**: "execute these 1000 independent tasks" / "perform this sequence atomically."
- **Imperative**: "spawn N workers, pull from a shared queue" / "acquire lock, do work, release lock."

TM is the declarative counterpart to lock-based synchronization — analogous to how `parallel_for` is declarative over manual thread management.

### TM semantics in one line
Many of the properties the cache-coherent memory system already guarantees for a *single* address, TM guarantees for *sets* of reads and writes. A transaction that reads {X,Y,Z} and writes {A,X} either all takes effect at once or appears never to have run.

### Motivation 1: concurrency without contention
In a shared tree, two threads updating node 3 and node 4 must coordinate. Hand-over-hand locking walks locks down from the root, so a locker sitting on node 2 en route to node 3 can delay an unrelated update to node 4. With transactions, each thread simply records its read-set and write-set (Transaction A reads 1,2,3 and writes 3; Transaction B reads 1,2,4 and writes 4). There is no write-to-shared-data overlap, so both can commit concurrently — no artificial serialization. If both instead wrote node 3, the system detects the conflict and serializes only those two.

On both HashMap and balanced tree benchmarks, a hardware TM system ("TCC") matches or beats fine-grained locks, while remaining as simple to write as coarse-grained locks.

### Motivation 2: failure atomicity
With locks and exceptions, the programmer is responsible for undoing partial work on every failure path:
```c
void transfer(A, B, amount) {
  synchronized(bank) {
    try {
      withdraw(A, amount);
      deposit(B, amount);
    }
    catch(exception1) { /* undo code 1 */ }
    catch(exception2) { /* undo code 2 */ }
    ...
  }
}
```
This is error-prone: some side effects may already be visible, an uncaught case can leave locks held and deadlock the system. With TM:
```c
void transfer(A, B, amount) {
  atomic {
    withdraw(A, amount);
    deposit(B, amount);
  }
}
```
Any uncaught exception aborts the transaction, discards its writes, and releases its (internal) tracking state. No "lost locks," no half-transferred money.

### Motivation 3: composability
Two independently correct lock-based functions can deadlock when composed:
```c
void transfer(A, B, amount) {             void transfer(B, A, amount) {
  synchronized(A) {                         synchronized(B) {
    synchronized(B) {                         synchronized(A) {
      withdraw(A, amount);                      withdraw(A, 2*amount);
      deposit(B, amount);                       deposit(B, 2*amount);
    }                                         }
  }                                         }
}                                         }
```
Thread 0 calls `transfer(A,B,100)` while Thread 1 calls `transfer(B,A,200)` — classic inverted-order deadlock. Avoiding it requires a *system-wide* lock-ordering policy, which breaks module boundaries.

With TM:
```c
void transfer(A, B, amount) {
  atomic {
    withdraw(A, amount);
    deposit(B, amount);
  }
}
```
Transactions nest cleanly: the outer `atomic` subsumes whatever atomic blocks `withdraw` and `deposit` use. The system serializes `transfer(A,B,100)` and `transfer(B,A,200)` (they conflict) but runs `transfer(A,B,100)` and `transfer(C,D,200)` concurrently (they don't). The programmer declares global intent; the system handles global strategy.

### Caveats: `atomic { } ≠ lock() + unlock()`
Important semantic distinctions:
- Locks can *implement* atomicity but don't *provide* it. Two threads each taking a different lock gain no mutual exclusion.
- Locks are used for things beyond atomicity (signaling, ordering), so you cannot mechanically replace every `lock/unlock` with `atomic { }`. A classic example is a pair of threads each setting a flag and spinning on the other's flag inside locked regions — works with `synchronized`, but a TM system might never let both "transactions" commit because each reads what the other wrote.
- Atomic blocks don't rescue you from *atomicity violations* at the program level. Splitting a logically atomic sequence into two atomic blocks is still a bug:
  ```c
  // Thread 1                         // Thread 2
  atomic { ... ptr = A; ... }         atomic { ... ptr = NULL; ... }
  atomic { B = ptr->field; }          // thread 2 could NULL ptr between the two atomics
  ```
  TM closes data races within a block, not sloppy block boundaries.

### Advantages, recapped
1. **Ease of use** — as easy as coarse-grained locks.
2. **Performance** — often matches fine-grained locks; the runtime provides automatic read-read concurrency and portability across core counts.
3. **Failure atomicity** — no lost locks, abort = rollback.
4. **Composability** — safe to nest and combine modules.

### Implementing TM: the design space
Two core questions any TM system must answer.

#### Data versioning: where does the new value live?
**Eager versioning (undo-log based)** — optimism about commits.
- On write, update memory *immediately*.
- Save the old value into a per-transaction undo log.
- Commit: do nothing (memory is already correct); discard the log.
- Abort: walk the undo log backward, restoring old values.
- Pros: fast commit (memory already updated).
- Cons: slower abort, fault-tolerance issues (a crash mid-transaction leaves memory in a torn state unless the log is durable).

**Lazy versioning (write-buffer based)** — pessimism about commits.
- On write, store into a per-transaction write buffer. Memory is untouched.
- Reads must check the write buffer first, then fall through to memory.
- Commit: flush the buffer to memory atomically.
- Abort: discard the buffer.
- Pros: fast/safe abort, no fault-tolerance issues.
- Cons: slower commit (must drain the buffer), reads pay a buffer-lookup cost.

Picture for eager: a single memory cell `X: 10` becomes `X: 15` on write, with `X: 10` pushed into the undo log; commit erases the log, abort restores from it. Picture for lazy: memory stays `X: 10`, the write buffer holds `X: 15`, commit pushes the buffer into memory, abort throws it away.

#### Conflict detection: when to look for trouble?
The system tracks a **read-set** (addresses read) and a **write-set** (addresses written) per transaction. A **read-write conflict** is one transaction reading an address another has written; a **write-write conflict** is two transactions writing the same address.

**Pessimistic (a.k.a. eager) detection** — check on every load/store.
- Philosophy: "conflicts are likely; detect them now so we don't waste work."
- A *contention manager* decides between stalling the later transaction or aborting it (a common policy is "writer wins": the transaction that already wrote keeps going, readers must abort).
- Four sketched cases on two transactions T0/T1:
  1. *Success* — neither touches the other's set; both commit.
  2. *Early detect (and stall)* — T1 writes A, then T0 tries to read A; T0's read triggers the check, and T0 stalls until T1 commits, then proceeds.
  3. *Abort* — T0 reads A, T1 writes A; the write's check finds the conflict and (writer-wins) T0 is restarted.
  4. *No progress / livelock risk* — T0 and T1 keep writing A and aborting each other; without back-off, they can livelock (hence the open question: how to avoid livelock?).
- Pros: early detection saves wasted work; some potential aborts become mere stalls.
- Cons: no forward-progress guarantee; fine-grained communication (check per access); detection sits on the critical path.

**Optimistic (a.k.a. lazy / commit-time) detection** — check only at commit.
- Philosophy: "hope for the best; sort it out when we try to commit."
- On conflict, priority goes to the committing transaction; others abort later.
- Four sketched cases:
  1. *Success* — disjoint sets, both commit.
  2. *Abort* — T1 writes A, then T0 reads A. T1 reaches commit first, passes its check, commits; T0's later commit check sees the conflict and restarts. T0 re-reads A and commits.
  3. *Success* — T0 reads A, then commits; T1 later writes A and commits; no overlap at commit time.
  4. *Forward progress* — two transactions writing A serialize through their commit checks; whichever commits first wins, the other restarts and eventually succeeds.
- Pros: forward-progress guarantee (at least one transaction always makes it through a commit race); bulk detection (compare sets once, not per access).
- Cons: late detection wastes work; fairness problems remain (a long transaction can starve).

### Failure atomicity and recovery
A consequence of the above: because the runtime already knows how to undo (eager) or discard (lazy) a transaction's effects, any exception or hardware fault inside a transaction can be converted into an abort + restart. That is how TM gives "failure atomicity for free" — as long as the fault is one the runtime can observe.

## Examples / Worked Problems
- **Deposit** — same code, lock version vs `atomic { }` version (above). The point is semantic, not performance: the atomic block leaves the synchronization strategy to the system.
- **Doubly-linked list `PushLeft`** — wrapping the pointer-shuffle (`qn->left`, `qn->right`, `leftSentinel->right`, `oldLeftNode->left`) in a single `atomic { }` block makes it thread-safe without any manual lock ordering between the two sentinels. Locks here would need to guard *at least* both adjacent nodes to avoid torn lists.
- **Tree update (nodes 3 and 4)** — hand-over-hand locking down from root 1 through 2 can make an update to node 4 wait for an update to node 3 to release node 2, even though the two updates touch disjoint data. With TM, each transaction records reads {1,2,3} or {1,2,4} and writes {3} or {4}; the read/write sets do not conflict, so both commit in parallel.
- **Tree update (both modify node 3)** — now the write-sets collide. TM correctly serializes the two transactions (and only those two).
- **Transfer under composition** — `transfer(A,B,100)` and `transfer(B,A,200)` deadlock with fixed-order nested locks; with `atomic { withdraw; deposit; }` the system serializes just these two and runs unrelated transfers concurrently.
- **Eager vs lazy versioning diagrams** — begin, write x←15, commit/abort sequences showing undo-log vs write-buffer state.
- **Pessimistic vs optimistic detection case tables** — the 4×2 grid above; useful to trace through on paper.

## Takeaways
1. `atomic { ... }` is a declarative abstraction: the programmer names an atomicity boundary, the system implements it. It is not a macro for `lock/unlock`.
2. TM's promise is "coarse-lock simplicity with fine-lock performance," plus failure atomicity and composability.
3. Correctness semantics are the database trio: **atomicity, isolation, serializability** — but only for memory operations within a transaction.
4. Any TM implementation must choose a **data versioning** policy (eager/undo-log vs lazy/write-buffer) and a **conflict detection** policy (pessimistic per-access vs optimistic at-commit). Each choice trades commit cost against abort cost and forward-progress guarantees.
5. TM closes data races inside a block, not between blocks. Programmer still owns where the atomic boundaries go — getting that wrong still causes atomicity violations.
6. Locks serve purposes beyond atomicity (signaling, ordering); you cannot mechanically replace every lock with an atomic block.

## Open Questions / Follow-ups
- **Livelock avoidance** under pessimistic detection when two transactions keep aborting each other — teased as an open question in the slides.
- **Fairness and starvation** under optimistic detection (long transactions vs short ones).
- **Granularity** of the read-set/write-set: word, cache line, or object? Affects false conflicts. Deferred to Part 2.
- **Consistency model for TM** — how transactions interleave with non-transactional memory accesses (strong vs weak atomicity). Lecture gestures at it ("What is the consistency model for TM?") without pinning it down.
- **Software vs hardware implementations** — STM (barriers on every access, software-managed logs) and HTM (piggyback on the cache coherence protocol) are the subject of Lecture 18.

## Sources
- Slides: `slices/17_transactionalmem.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Course site: https://gfxcourses.stanford.edu/cs149
- Tree/hand-over-hand figures credited to Austen McDonald; TCC is the Stanford hardware-TM system used for the HashMap / balanced-tree performance comparisons.
