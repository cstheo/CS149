

# Env

System: macOS 26.01

Chip: Apple M4

Cores: 10 (4 performance and 6 efficiency)

# Prog1

| threadNum | 1    | 2     | 4     | 8     | 16    |
| --------- | ---- | ----- | ----- | ----- | ----- |
| Speedup   | 1x   | 1.64x | 2.31x | 3.40x | 5.57x |

![image-20251119221138264](./prog1_mandelbrot_threads/view1_speedup.png)

![image-20251119221138264](./prog1_mandelbrot_threads/view2_speedup.png)

Unable to optimize on macOS...
Transfer to Arch Linux

After optimization(Speedup):
view1: 10.28x
view2: 9.3x 

# Prog3

## Part 1

What is the maximum speedup you expect given what you know about these CPUs? 

  - AVX2 executes 8-wide SIMD vectors, so the ceiling for a single-core ISPC kernel mapped directly to SIMD lanes is roughly 8× vs.

Why might the number you observe be less than this ideal?

- Mandelbrot iteration counts vary dramatically per pixel: points inside the set run the loop to maxIterations, points far outside exit.The pixels in the image are unevenly distributed.

## Part 2

Why 256 Tasks Work Best

  - make sure the runtime can keep every worker thread busy

What are differences between the thread abstraction and the ISPC task abstraction? 

  - thread is scheduled by kernel, ispc task is scheduled by ispc runtime
  - ispc task is more lightweight 

- thread is MIMD but ispc is SIMD
- the cost of ispc schedule is much lower(ispc task schedule no need to context switch)
- ispc runtime is just user program

# Prog4

**Init**

3.9x speed up due to SIMD parallelization

45x speed up due to multicore parallelization

**best**

All set to 0.0f

6.65x speedup from ISPC
73.81x speedup from task ISPC

**worst**

All set to 1.0f

1.75x speedup from ISPC
2.93x speedup from task ISPC

**AVX2** with best speedup

6.58x speedup from ISPC
62.15x speedup from task ISPC
8.98x speedup from AVX2

# Prog5

**Init**

[saxpy serial] :	     [10.512] ms	[28.350] GB/s	[3.805] GFLOPS
[saxpy ispc] :		[9.191] ms	[32.425] GB/s	[4.352] GFLOPS
[saxpy task ispc] :	[5.097] ms	[58.474] GB/s	[7.848] GFLOPS
(1.80x speedup from use of tasks)

could you rewrite the code to achieve near linear speedup? Yes or No? Please justify your answer.

- For extremely large vectors like N=20M, which is far greater than the cache capacity, SAXPY is basically a pipeline of "streaming data from memory → performing a few calculations → writing it back"; the bottleneck is not computing power but DRAM bandwidth. More cores/more tasks can only make "full bandwidth" faster, but once the memory bandwidth is saturated, adding more cores will not make it linearly faster.

Note that the total bandwidth memory consumed computation in main.cpp is TOTAL_BYTES = 4 * N * sizeof(float);. Even though saxpy loads one element from X, one element from Y, and writes one element to result the multiplier by 4 is correct. Why is this the case?

- We are handling large vectors which is far greater than the cache capacity. So we will encounter cache miss, which mean we will get the data from lower storage (disk). a typical CPU cache is write-back + write-allocate. If the destination cache line isn’t already in cache, the first store triggers a Read-
    For-Ownership (RFO): the CPU must read the whole cache line from memory into cache before it can modify any bytes in it. Later, when that dirty line is evicted, it’s written back to memory.
