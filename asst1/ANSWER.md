

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
