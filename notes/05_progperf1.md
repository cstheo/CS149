# Lecture 5 — Performance Optimization Part 1: Work Distribution and Scheduling

## Overview
First of two performance-tuning lectures. Frames parallel optimization as iterating on decomposition, assignment, and orchestration to balance three competing goals: distribute work evenly, reduce communication, and minimize overhead. Surveys static, semi-static, and dynamic assignment strategies (including work queues, task granularity, smarter scheduling, and distributed queues with work stealing), then dives deep into the Cilk Plus runtime's scheduler for fork-join parallelism — showing why it chooses *continuation stealing* with a greedy join policy.

## Key Concepts
- **Three competing goals** — balance workload, reduce communication, reduce overhead.
- **TIP #1** — implement the simplest solution first, then measure before optimizing.
- **Load imbalance** — even small imbalances bound speedup (Amdahl-style): a worker with 2× work makes 50% of parallel runtime effectively serial.
- **Static assignment** — work-to-thread mapping determined without runtime observation; near-zero overhead. Applicable when cost is predictable (or predictable on average).
- **Semi-static assignment** — periodically re-profile and re-balance (e.g., particle simulations, adaptive meshes).
- **Dynamic assignment** — runtime assignment via a shared work queue; needed when task cost or count is unknown.
- **Task granularity tradeoff** — many small tasks → good balance, high synchronization cost; few big tasks → low overhead, risk of imbalance. Rule of thumb: many more tasks than processors.
- **Smarter scheduling** — schedule long tasks first; subdivide big tasks to shorten the critical "long pole."
- **Distributed work queues + work stealing** — per-worker deques; idle workers steal from the top of a victim's deque, locals push/pop the bottom.
- **Fork-join** — `cilk_spawn` (fork: caller may continue asynchronously with callee) / `cilk_sync` (join: wait for all spawns in the enclosing function). Implicit `cilk_sync` at function exit.
- **Child stealing vs. continuation stealing** — at a spawn, does the worker run the spawned child (queueing the continuation) or run the continuation (queueing the child)? Cilk uses continuation stealing.
- **Greedy join policy** — threads never wait at a sync; they steal if there's work. The thread that finishes the final spawn resumes the post-sync continuation.
- **Parallel slack** — ratio of independent work to machine parallelism; ~8× is a practical target.

## Detailed Notes

### Programming for high performance
Optimizing parallel programs is iterative: refine decomposition, assignment, and orchestration. The three goals — balance, low communication, low overhead — are at odds. Start with the simplest implementation, measure, and only then pursue more elaborate schemes.

### Load imbalance and Amdahl
If four workers share 12 equal tasks, ideal is `3 tasks × 4 workers`. If P4 gets double work, it runs 2× longer and becomes the bottleneck; in that example ~50% of parallel runtime is serialized on P4 (even though the imbalanced work is only ~1/5 of the total, giving `S = 0.2` in Amdahl's equation).

### Static assignment
Mapping is fixed up front (possibly at runtime, as long as it doesn't react to dynamic behavior). Overhead is essentially indexing math. Applicable when:
- all work has equal cost, or
- per-task cost varies but is known, so you hand out equal total cost, or
- cost is statistically predictable (equal on average) — hand out equal counts.

Programming Assignment 1 prog 1 (grid-cell assignment) explored several row/column/interleaved static layouts.

### Semi-static assignment
Predictability holds over the near future. Periodically profile and re-balance; between re-balances the schedule is static. Examples:
- **Particle simulation** — redistribute particles across workers as they drift spatially (if drift is slow, re-balance is infrequent).
- **Adaptive mesh** — remap mesh regions to processors when the mesh changes with the simulated object/flow.

### Dynamic assignment with a shared work queue
When execution time is unknown (e.g., primality testing of arbitrary integers), threads pull indices from a shared counter under a lock:

```c
LOCK counter_lock;
int counter = 0;

while (1) {
    int i;
    lock(counter_lock);
    i = counter++;          // atomic_incr(counter) also works
    unlock(counter_lock);
    if (i >= N) break;
    is_prime[i] = test_primality(x[i]);
}
```

**Granularity problem** — with one element per task, threads enter the critical section N times. Amortize by claiming `GRANULARITY` elements per lock acquisition:

```c
const int GRANULARITY = 10;
while (1) {
    int i;
    lock(counter_lock);
    i = counter;
    counter += GRANULARITY;
    unlock(counter_lock);
    if (i >= N) break;
    int end = min(i + GRANULARITY, N);
    for (int j = i; j < end; j++)
        is_prime[j] = test_primality(x[j]);
}
```

Choosing task size balances parallel slack against management overhead; the right value depends on workload and machine.

### Smarter dynamic scheduling
Consider 16 tasks where one is much longer. If the scheduler runs the long task last, other workers go idle waiting for it (the "long pole"). Two remedies:
1. **Divide work more finely** — shorter long pole relative to total, at the cost of more sync overhead; sometimes impossible if the long task is intrinsically serial.
2. **Schedule long tasks first** — the worker assigned the long task does fewer tasks but comparable total work; others fill in around it. Requires some predictability of cost.

### Distributed queues and work stealing
One shared queue synchronizes every worker on every dequeue. Instead give each worker its own queue:
- Local push/pop from own deque (no sync).
- When a worker's queue empties, it **steals** from another worker's queue.

This keeps the common case lock-free while still load-balancing under imbalance.

### Dependent tasks
Real task systems let workers submit tasks with dependencies. A task becomes runnable only when its dependencies have all completed:

```c
foo_handle = enqueue_task(foo);                 // independent
bar_handle = enqueue_task(bar, foo_handle);     // runs after foo
```

The scheduler tracks the DAG and dispatches ready tasks.

### Common parallel patterns
- **Data parallelism** — same op over many elements: ISPC `foreach`, ISPC `launch[N] task`, `map`, OpenMP `#pragma omp parallel for`, CUDA `kernel<<<grid, block>>>`.
- **Explicit threading** — create one thread per execution context; `std::thread` with a join loop.
- **Fork-join** — natural for divide-and-conquer. Shown via Cilk Plus.

### Divide-and-conquer and Cilk Plus
Quicksort's two recursive calls are independent — prime fork-join material. Cilk Plus (C++ extension; originally MIT, now in GCC/ICC):

```cpp
cilk_spawn foo(args);   // fork: foo runs, caller may proceed asynchronously
cilk_sync;              // join: wait for all spawns in this function
                        // implicit cilk_sync at function end
```

Basic shapes:

```cpp
// foo and bar may run in parallel
cilk_spawn foo();
bar();
cilk_sync;

// two spawns: same independent work, more overhead (2 spawns vs 1)
cilk_spawn foo();
cilk_spawn bar();
cilk_sync;

// four-way parallel
cilk_spawn foo();
cilk_spawn bar();
cilk_spawn fizz();
buzz();
cilk_sync;
```

Parallel quicksort with a sequential cutoff so spawn overhead doesn't dominate at the leaves:

```cpp
void quick_sort(int* begin, int* end) {
    if (begin >= end - PARALLEL_CUTOFF) {
        std::sort(begin, end);
    } else {
        int* middle = partition(begin, end);
        cilk_spawn quick_sort(begin, middle);
        quick_sort(middle + 1, end);
        // implicit cilk_sync
    }
}
```

Key abstraction point: `cilk_spawn` *allows* parallelism but doesn't mandate it; a conforming implementation could run the spawn as a plain call. Only `cilk_sync` imposes an ordering constraint.

### Writing fork-join programs (rules of thumb)
- At least as much spawned work as machine parallelism.
- Aim for ~8× parallel slack so the scheduler can balance.
- Not so fine that per-task overhead dominates — respect granularity.

### Implementing the Cilk scheduler

**Worker pool** — created once (lazily, in practice). Exactly as many workers as hardware execution contexts. Each runs:
```
while (work_exists()) {
    work = get_new_work();
    work.run();
}
```
A naïve `pthread_create` per `cilk_spawn` is untenable: thread creation is heavy, oversubscription causes context-switch churn and cache pollution.

**Per-thread work deque** — at `cilk_spawn foo()` the worker enqueues *something* and starts executing the other side. When another worker goes idle it steals from a busy worker's queue.

**Child vs continuation stealing — the key design choice.** Given
```cpp
for (int i = 0; i < N; i++) cilk_spawn foo(i);
cilk_sync;
```
- **Child stealing** (run continuation first, queue the child): thread 0 queues `foo(0) … foo(N-1)` *before* running any. Queue grows O(N) — breadth-first traversal. If nothing is stolen, execution order differs from the spawn-free program.
- **Continuation stealing** (run child first, queue the continuation): thread 0 queues "continuation: i=1", starts `foo(0)`. When `foo(0)` returns, it pops the continuation, queues "i=2", starts `foo(1)`. If nothing is stolen, execution is depth-first and equivalent to the serial program. Queue storage with T threads is bounded by T× the single-threaded stack. If an idle thread steals the continuation, it bumps `i` and starts `foo(i)` while the original keeps going. Cilk chooses this.

**Deque mechanics** — work queue is a double-ended queue.
- Local thread pushes/pops the **tail** (bottom).
- Remote thieves steal from the **head** (top).
- Stealing from the top grabs the *largest, oldest* continuation (more work per steal, fewer steals), gives stealers good locality when combined with run-child-first, and avoids contending with the local worker's end — enabling lock-free implementations.
- Victim is chosen randomly.

Example with quicksort on 200 elements: thread 0 working on 0–25 has continuations `26–50`, `51–100`, `101–200` stacked bottom-to-top; thieves steal the fat `101–200` first, leaving coherent local chunks.

### Implementing `cilk_sync`

Two cases:

**No steals** — if nothing from the block was stolen, every spawn was executed locally via normal call/return. `cilk_sync` is a no-op.

**Stealing occurred** — the runtime creates a *descriptor* for the enclosing block (id `A`) with counters `spawn` and `done`:
- When a spawn is stolen, `spawn++`.
- When a stolen spawn finishes, `done++`.
- `cilk_sync` completes once `spawn == done`.

Walkthrough (loop spawning `foo(0)…foo(9)` then `cilk_sync; bar();`):
1. Thread 1 steals the continuation `i=0` from thread 0. Descriptor `A` created with `spawn=1, done=0`. Thread 1 bumps `i` and starts `foo(1)`; thread 0 keeps running `foo(0)`.
2. Thread 2 steals the continuation from thread 1 — `spawn=3, done=0`.
3. `foo(0)` finishes → `done=1`; thread 0 goes idle and steals more (e.g., continuation `i=3`), incrementing `spawn`.
4. Progress continues; eventually one thread is running `foo(9)` and everyone else is idle, descriptor `spawn=10, done=9`.
5. When the last spawn completes (`spawn=10, done=10`), the thread holding the post-sync continuation resumes and executes `bar()`. Descriptor `A` is freed.

Because the thread that initiated the spawns may itself have been stolen from (and is now running something else), the worker that resumes the code after `cilk_sync` is often *not* the same one that entered the block — this is **greedy join** scheduling. Workers never idle at a sync if there's anything to steal.

### Anticipating divide-and-conquer with recursive splitting
A flat `for` loop with `cilk_spawn` per iteration fills the machine slowly under continuation stealing — one thread does all the spawning. A recursive splitter exposes parallelism to stealers fast:

```cpp
void recursive_for(int start, int end) {
    while (start <= end - GRANULARITY) {
        int mid = (start + end) / 2;
        cilk_spawn recursive_for(start, mid);
        start = mid;
    }
    for (int i = start; i < end; i++) foo(i);
}

recursive_for(0, N);
```

Each level doubles the stealable continuations, so the work tree fans out logarithmically and all workers find something to do quickly.

### Cost model of steals
Bookkeeping (descriptors, atomic updates) is paid **only on steals**. When large continuations are stolen, steals are rare — most of the time, workers push/pop their own deque, which is cheap.

## Examples / Worked Problems
- **Primality dynamic scheduling** — contrast fine-grained (1 element, high lock overhead) vs coarse-grained (10 elements, 10× fewer critical-section entries) work-queue loops.
- **Long-pole scheduling** — with 16 tasks where one is big, running long last causes idle time; running long first balances total work per thread.
- **Quicksort under Cilk** — `PARALLEL_CUTOFF` prevents spawn overhead from dominating leaves; the recursive DAG (partition → two quicksorts → four → std::sort leaves) is the canonical fork-join tree.
- **200-element quicksort steal order** — thread 0 working on 0–25 holds continuations for 26–50, 51–100, 101–200; thieves steal from the top (101–200) first.
- **Loop spawn sync walkthrough** — full descriptor trace from first steal through last completion, showing how `spawn`/`done` counters drive `cilk_sync` and how the post-sync `bar()` is resumed by whichever thread closes the count.
- **`recursive_for` vs flat loop** — recursive splitter fills the machine in O(log N) steals; flat loop relies on serial spawning.

## Takeaways
1. Balance, communication, and overhead pull against each other — the right mix depends on workload predictability and machine parameters.
2. Prefer static assignment when you can predict cost; fall back to semi-static, then dynamic queues, only as predictability erodes.
3. Granularity is a dial, not a fixed choice: many small tasks enable balancing, few large tasks reduce overhead. Aim for ~8× parallel slack.
4. Distributed deques with random-victim work stealing give lock-free common-case performance plus load balancing under imbalance.
5. Fork-join (Cilk) is the natural fit for divide-and-conquer. Continuation stealing preserves serial execution order when no steals happen and bounds queue storage to O(T · stack).
6. Steal from the top, work from the bottom — locality for both parties and cheap lock-free concurrency.
7. Greedy join: threads never wait at a sync. Bookkeeping is paid only when stealing actually occurs, and descriptor counters gate `cilk_sync` completion.
8. Expose parallelism early with recursive splitters so stealers fill the machine fast.

## Open Questions / Follow-ups
- What's the formal argument that continuation stealing bounds queue storage to T× serial stack, and when does it fail in practice?
- How do lock-free deque algorithms (Arora-Blumofe-Plaxton, Chase-Lev) actually handle the head/tail race?
- Correctness criterion for treating `cilk_spawn` as a plain call — what properties of the program guarantee this is safe?
- How do dependency-aware task systems (e.g., TBB `flow_graph`, OpenMP depend clauses, Legion) compare to pure fork-join at scale?
- Performance Part 2 (next lecture) — communication cost and its interaction with assignment.

## Sources
- Slides: `slices/05_progperf1.pdf` (Stanford CS149, Fall 2025).
- Raw transcript: `notes/.raw/05_progperf1.txt`.
- Referenced system: Cilk Plus (originally MIT; now GCC / Intel ICC).
- Related: Amdahl's law from Lecture 4; Programming Assignment 1 progs 1 and 3.
