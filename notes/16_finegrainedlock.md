# Lecture 16 — Implementing Locks, Fine-Grained Synchronization, and Lock-Free Programming

## Overview
Covers how locks are actually built on cache-coherent hardware, why naive test-and-set is bad for scalability, and how to design progressively better lock primitives (test-and-test-and-set, ticket lock) using atomic instructions like `cmpxchg` and LL/SC. Then pivots to *using* locks: starts with a coarse per-data-structure lock, moves to fine-grained hand-over-hand locking on a linked list, and finally introduces lock-free data structures (stacks, queues) along with their pitfalls (ABA problem, use-after-free, hazard pointers). Ends by framing when lock-free is actually worth the complexity and previews transactional memory.

## Key Concepts
- **Deadlock** — all threads stuck waiting on resources held by each other. Requires mutual exclusion, hold-and-wait, no preemption, and circular wait.
- **Livelock** — threads execute work but make no meaningful forward progress (e.g., continually abort/retry).
- **Starvation** — system makes progress overall but some thread never does; a fairness problem, usually not permanent.
- **Test-and-set (TS)** — atomic primitive that loads a word and writes 1 if it was 0. Simple spinlock, but every spinner causes invalidation traffic per attempt.
- **Test-and-test-and-set** — spin on a plain read in cache; only issue the atomic RMW when the read sees the lock free. Cuts bus traffic dramatically.
- **Ticket lock** — atomically grab `next_ticket`; spin on `now_serving`; unlock increments `now_serving`. O(P) invalidations per release and provides FIFO fairness.
- **compare-and-swap (CAS)** — atomic `if (*p == old) *p = new`. Universal building block for locks, atomics, and lock-free structures. On x86: `lock cmpxchg` (and `cmpxchg8b`/`cmpxchg16b` for double-wide).
- **LL/SC** — load-linked / store-conditional: SC only commits if the line wasn't touched since the LL. ARM `LDREX`/`STREX`.
- **Fine-grained locking** — per-node locks with hand-over-hand acquisition enable concurrent operations on different parts of a shared structure.
- **Lock-free** — non-blocking; *some* thread is guaranteed to make system-wide progress even if any thread is preempted or crashes. Does not imply starvation-freedom.
- **ABA problem** — a CAS that observes the same pointer value twice can't tell whether it changed and changed back; leads to corruption in lock-free stacks/lists.
- **Hazard pointers** — per-thread "do not free" pointers that let a reclaimer know when a popped node is safe to delete.

## Detailed Notes

### Terminology preliminaries
Deadlock is illustrated by two threads each producing into the other's full work queue: each waits for space, neither can make it. The four conditions (mutual exclusion, hold-and-wait, no preemption, circular wait) must all hold. Livelock vs. starvation: livelock = busy but no progress; starvation = others progress but one victim doesn't.

### Review of MSI and lock coherence traffic
Recapped the MSI state diagram (Modified / Shared / Invalid with `PrRd`, `PrWr`, `BusRd`, `BusRdX` transitions and flushes) because every lock acquire is a coherent memory transaction. The worked sequence for two processors sharing X/Y shows how each `ST` forces `BusRdX` and invalidations, which is exactly the cost you pay when contending for a lock.

### Test-and-set lock
```c
lock:   ts  R0, mem[addr]   // atomic: load, set-to-1 if was 0
        bnz R0, lock        // retry if it was already 1
unlock: st  mem[addr], #0
```
On x86 the equivalent is `lock cmpxchg dst, src`: compares `dst` with `EAX`; if equal, writes `src` and sets `ZF=1`; else loads `dst` into `EAX` and clears `ZF`.

**Coherence trace under contention** (P1 holds lock, P2 and P3 spin): every `ts` from a waiter is a `BusRdX` that invalidates everyone, so releasing the lock also takes a `BusRdX` that fights the waiters for the bus. Result: latency is low with no contention but traffic is O(P) per *attempt*, and the lock holder itself is slowed.

Desirable properties: low latency uncontended, low interconnect traffic contended, scalability, low storage, fairness. Simple TS scores well only on latency and storage.

### Test-and-test-and-set
```c
void Lock(int* lock) {
  while (1) {
    while (*lock != 0);              // spin read-only in local cache (S state)
    if (test_and_set(lock) == 0)     // try RMW once lock looks free
      return;
  }
}
void Unlock(int* lock) { *lock = 0; }
```
While the lock is held, all waiters have the line in S and do no bus traffic. On release: one `BusRdX` invalidates all waiters; each reads to re-share, then one wins the `ts`. That is O(P) invalidations per *release* rather than per *attempt* — roughly O(P²) total traffic if all P processors contend, versus O(P³) for naive TS. Still unfair.

### Ticket lock
```c
struct lock { int next_ticket; int now_serving; };
void Lock(lock* l) {
  int my_ticket = atomic_increment(&l->next_ticket);
  while (my_ticket != l->now_serving);
}
void unlock(lock* l) { l->now_serving++; }
```
Waiters spin on a plain read of `now_serving`. One invalidation per release reaches all waiters. FIFO fairness falls out by construction.

### Building other atomics from CAS
```c
int atomicCAS(int* addr, int compare, int val) {
  int old = *addr;
  *addr = (old == compare) ? val : old;
  return old;
}

void atomic_min(int* addr, int x) {
  int old = *addr;
  int new = min(old, x);
  while (atomicCAS(addr, old, new) != old) {
    old = *addr; new = min(old, x);
  }
}
```
Same pattern yields `atomic_increment`, `lock`, etc. CUDA exposes a full family (`atomicAdd`, `atomicCAS`, `atomicMin`, `atomicExch`, `atomicOr`, …).

A CAS-based lock is more efficient under contention when combined with a read-spin, same idea as TTAS:
```c
void lock(Lock* l) {
  while (1) {
    while (*l == 1);
    if (atomicCAS(l, 0, 1) == 0) return;
  }
}
```

### LL/SC
`load_linked(x)` + `store_conditional(x, value)`: SC succeeds only if nothing has written `x` between it and its LL. Cache-coherent implementations track a reservation that is cleared by any invalidation of the line. ARM provides `LDREX`/`STREX`.

### C++11 atomics
`std::atomic<T>` gives atomic read/write/RMW and memory-ordering semantics (default: sequential consistency, relaxable via `std::memory_order`). For basic `T` it lowers to HW atomics; `is_lock_free()` tells you. Example: `i.compare_exchange_strong(a, 10)`.

### Sorted linked list — what goes wrong without synchronization
With `insert` walking `prev`/`cur` pointers, two simultaneous inserts can land at the same position and one overwrites the other's `prev->next`. Insert vs. delete can leave the new node pointing at a freed cell, corrupting the list.

### Solution 1 — single lock per list
Add a `Lock` in `List` and bracket every operation with `lock(list->lock)` / `unlock(list->lock)`. Correct but fully serializes operations on disjoint parts of the list.

### Solution 2 — hand-over-hand (fine-grained) locking
Move the lock into each `Node`. Traversal acquires the next node's lock before releasing the current one — like a climber swinging hand-over-hand. Two deletes on distant elements then proceed concurrently.

```c
struct Node { int value; Node* next; Lock* lock; };
struct List { Node* head; Lock* lock; };

void delete(List* list, int value) {
  Node* prev, *cur;
  lock(list->lock);
  prev = list->head;
  lock(prev->lock);
  unlock(list->lock);
  cur = prev->next;
  if (cur) lock(cur->lock);
  while (cur) {
    if (cur->value == value) {
      prev->next = cur->next;
      unlock(prev->lock);
      unlock(cur->lock);
      delete cur;
      return;
    }
    Node* old_prev = prev;
    prev = cur;
    cur = cur->next;
    unlock(old_prev->lock);
    if (cur) lock(cur->lock);
  }
  unlock(prev->lock);
}
```
`insert` is symmetric. Deadlock-free because all threads acquire locks in strictly head-to-tail order (no cycles possible). Costs: lock/unlock per traversal step (extra instructions *and* memory writes), a lock word per node, and extra code complexity. A middle ground: lock chunks of nodes — same granularity trade-off as task size.

### Blocking vs. lock-free
A locking algorithm is **blocking**: a preempted / crashed / page-faulting thread holding the lock stops everyone else. **Lock-free** means at least one thread makes system-wide progress no matter what. It does not prevent starvation of individual threads.

### Single-producer / single-consumer bounded queue
Array + `head` + `tail`, modulo N. Empty iff `head == tail`, full iff `tail == head-1 mod N`. Because there is exactly one producer and one consumer, no atomic RMW is required — only appropriate memory ordering (SC or fences). `push` writes the slot and advances `tail`; `pop` reads and advances `head`.

### SPSC unbounded linked-list queue
Maintains `head`, `tail`, and a `reclaim` pointer. `head` points *before* the first real element; `tail` points to the last. The producer also reclaims freed nodes by walking `reclaim` up to `head` and deleting — because the producer alone allocates and deletes, there is no cross-thread free. Walkthrough of push 3, push 10, pop, pop, push 5 shows `reclaim` chasing `head` safely.

### Lock-free stack (first attempt)
```c
void push(Stack* s, Node* n) {
  while (1) {
    Node* old_top = s->top;
    n->next = old_top;
    if (compare_and_swap(&s->top, old_top, n) == old_top) return;
  }
}
Node* pop(Stack* s) {
  while (1) {
    Node* old_top = s->top;
    if (old_top == NULL) return NULL;
    Node* new_top = old_top->next;
    if (compare_and_swap(&s->top, old_top, new_top) == old_top) return old_top;
  }
}
```
Unlike fine-grained locking, threads hold *no* lock on the structure at all.

### The ABA problem
Thread 0 reads `old_top = A`, `new_top = B` and is about to CAS. Thread 1 pops A, pops B, pushes D, mutates A, and pushes A back. Now `s->top == A` again, so Thread 0's CAS succeeds and sets `top = B` — but B is no longer in the stack. The stack is corrupted (D is lost).

### ABA fix — tagged pointer / pop-counter
Store `(top, pop_count)` together and use a **double-wide CAS** that checks both fields.
```c
struct Stack { Node* top; int pop_count; };
// pop:
int pop_count = s->pop_count;
Node* top = s->top;
...
if (double_compare_and_swap(&s->top, top, new_top,
                            &s->pop_count, pop_count, pop_count+1))
  return top;
```
On x86, `cmpxchg8b` (2×32 bit) or `cmpxchg16b` (2×64 bit) handle this by keeping the pair contiguous. Alternatives: careful allocator/reuse policies.

### Use-after-free in lock-free pop
After reading `old.top` but before CAS, another thread may pop and *free* that node — dereferencing `old.top->next` crashes or returns garbage. This is a separate hazard from ABA.

### Hazard pointers (advanced)
Each thread publishes a per-thread `hazard` pointer to the node it is currently inspecting. Instead of freeing a popped node, a thread pushes it on a per-thread `retireList`. Periodically, it scans and deletes only those retired nodes that no thread's hazard pointer references. Sketch:
```c
Node* hazard;                 // per-thread
Node* retireList;             // per-thread
int   retireListSize;

void retire(Node* ptr) {
  push(retireList, ptr); retireListSize++;
  if (retireListSize > THRESHOLD)
    for (Node* n : retireList)
      if (n not pointed to by any thread's hazard pointer) {
        remove n from list; delete n;
      }
}
```
In `pop`, `hazard = s->top` is set before the dereference and cleared on retry/return.

### Lock-free linked-list insert
```c
void insert_after(List* list, Node* after, int value) {
  Node* n = new Node;
  n->value = value;
  Node* prev = list->head;
  while (prev->next) {
    if (prev == after) {
      while (1) {
        Node* old_next = prev->next;
        n->next = old_next;
        if (compare_and_swap(&prev->next, old_next, n) == old_next) return;
      }
    }
    prev = prev->next;
  }
}
```
No locks, no per-node lock storage. Lock-free **delete** is much harder: deleting B concurrent with "insert E after B" can leave B pointing to E while B itself is unlinked, losing E. Real solutions use marked pointers (Harris 2001) or back-links (Fomitchev 2004).

### Performance in practice
Hunt 2011 shows lock-free queue/dequeue/linked-list implementations vs. pthread mutex and fine-grained locks: lock-free wins in some regimes, but well-written fine-grained locks are often close or better on dedicated HPC-style machines. Real motivation for lock-free:
- Many-threaded systems (databases, webservers) where a thread can be preempted or page-fault inside a critical section.
- Avoids priority inversion, lock convoying, and crash-in-critical-section problems.
- Not useful for eliminating contention — CAS can still spin under heavy contention.

## Examples / Worked Problems
- **Coherence trace for TS lock**: three-processor walkthrough showing BusRdX-per-attempt traffic even while one thread holds the lock.
- **Coherence trace for TTAS lock**: waiters read-share the line, only one invalidation per release.
- **Build `atomic_min` from `atomicCAS`** (shown above); exercise: also build `atomic_increment` and `lock`.
- **Hand-over-hand delete of values 10 and 11 concurrently**: T1 deletes 10 while T0 deletes 11, both proceeding without blocking each other because their locked windows don't overlap.
- **ABA walkthrough on a lock-free stack** with nodes A, B, C, D: CAS sees the same top pointer A after interleaved pops/pushes and corrupts the stack — fixed by adding a pop counter and DCAS.
- **Exercise (take-home)**: implement a fine-grained BST with `insert` and `delete`.

## Takeaways
1. Lock implementation is a cache-coherence problem: minimize invalidations, don't just minimize instructions. TS → TTAS → ticket lock trace this progression.
2. Fairness is not free — ticket locks add a second word and a fetch-add but give FIFO ordering.
3. Fine-grained locking unlocks parallelism on shared data structures, at the cost of complexity, extra writes per traversal step, and per-node storage. Ordering the acquires prevents deadlock.
4. Lock-free ≠ contention-free. Its real win is robustness to preemption / faults / crashes inside a critical section, and composability in many-threaded environments.
5. CAS alone isn't enough for lock-free data structures: you also need to defeat ABA (tagged pointers or DCAS) and solve memory reclamation (hazard pointers, RCU, epoch-based schemes).
6. In HPC-style workloads where you own the machine, well-tuned locks often match or beat lock-free — reach for lock-free when the environment demands it.

## Open Questions / Follow-ups
- How exactly does LL/SC map onto MSI? (Teased in lecture — reservation on the cache line is cleared by any invalidation.)
- Full lock-free linked-list *delete*: see Harris 2001 "A Pragmatic Implementation of Non-blocking Linked-Lists" and Fomitchev 2004 "Lock-free linked lists and skip lists".
- Multi-producer/multi-consumer lock-free queues: Michael & Scott 1996.
- **Transactional memory** — previewed as a generalization of CAS's "did anyone else touch this?" check, with hardware/software support for abort/retry of whole critical sections (next lecture).
- Self-check posed in class: insert-only fine-grained list has a further optimization — which lock can be released earlier?

## Sources
- Slides: `slices/16_finegrainedlock.pdf` (Stanford CS149, Fall 2025).
- Raw notes: `notes/.raw/16_finegrainedlock.txt`.
- Culler, Singh, and Gupta — performance figures for TS vs TTAS locks.
- Michael & Scott 1996, "Simple, Fast and Practical Non-Blocking and Blocking Concurrent Queue Algorithms".
- Harris 2001, "A Pragmatic Implementation of Non-Blocking Linked-Lists".
- Fomitchev 2004, "Lock-free linked lists and skip lists".
- Hunt 2011, "Characterizing the Performance and Energy Efficiency of Lock-Free Data Structures".
- Dr. Dobb's: "Lock-Free Code: A False Sense of Security"; MemSQL blog: "Common Pitfalls in Writing Lock-Free Algorithms".
- Michael Sullivan's RMC compiler: https://github.com/msullivan/rmc-compiler
