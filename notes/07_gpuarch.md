# Lecture 7 — GPU Architecture & CUDA Programming

## Overview
Traces how GPUs evolved from fixed-function 3D rendering chips into general-purpose, highly parallel compute engines, and then introduces CUDA as the programming interface that exposes that hardware. Covers the CUDA abstractions (thread/block/grid hierarchy, distributed host/device memory, shared memory, synchronization), the NVIDIA V100 SM microarchitecture (warps, SIMT, sub-cores, register files), and the dynamic thread-block scheduler that maps CUDA programs onto any GPU without code changes.

## Key Concepts
- **GPU origin story** — designed to run *shader programs* per pixel/vertex in parallel. Shaders are pure functions over streams, which looks like data parallelism, and motivated the move to general compute (GPGPU, then CUDA).
- **Compute mode (NVIDIA Tesla, 2007)** — non-graphics interface: allocate device buffers, copy data in/out, launch `launch(kernel, N)` in SPMD.
- **CUDA thread hierarchy** — grid of *thread blocks*, each block a group of *CUDA threads*. IDs can be 1/2/3-D (`blockIdx`, `threadIdx`, `blockDim`).
- **Distributed address spaces** — host memory (CPU) vs device global memory (GPU); must `cudaMalloc` + `cudaMemcpy`. Inside the device: per-thread private, per-block `__shared__`, per-program global.
- **Synchronization** — `__syncthreads()` barrier within a block; atomic ops on shared/global memory; implicit barrier at kernel return.
- **Block independence assumption** — the runtime may schedule blocks in any order on any SM; threads *within* a block run concurrently and may cooperate.
- **Warp** — NVIDIA implementation detail: a group of 32 CUDA threads executed in lockstep on SIMD ALUs (SIMT). Not part of the CUDA programming model but decisive for performance (divergence ⇒ masked lanes).
- **SM (Streaming Multiprocessor)** — one GPU core. V100 has 80 SMs, each split into 4 sub-cores, each sub-core holds up to 16 warps, 64 warps per SM, 64 KB registers/sub-core (256 KB/SM), 128 KB shared/L1 per SM.
- **Dynamic block scheduler** — hardware unit that maps pending blocks to SMs whenever resources (threads, shared mem, registers) are free. Lets the same binary run on a 6-core or 80-core GPU.

## Detailed Notes

### From 3D rendering to compute
The rendering workload is: given 3D triangle meshes + camera + lights, compute the color of each output pixel. This splits into a pipeline — *vertex generation → vertex processing → primitive generation → rasterization (fragment generation) → fragment processing → pixel ops*. The fragment-processing stage runs a programmer-supplied *shader* function once per covered pixel, e.g. in GLSL:

```glsl
uniform sampler2D myTexture;
uniform float3 lightDir;
varying vec3 norm;
varying vec2 uv;

void myShader() {
    vec3 kd = texture2D(myTexture, uv);
    kd *= clamp(dot(lightDir, norm), 0.0, 1.0);
    return vec4(kd, 1.0);
}
```

A shader is a *pure function invoked over a stream of inputs* — the same observation that motivated data-parallel supercomputers in the 90s. GPUs therefore packed many SIMD, multi-threaded cores to run shaders in parallel, with huge memory bandwidth (~1 TB/s on high-end GPUs today).

### GPGPU hacks and Brook (2002–2004)
Before compute APIs existed, researchers mapped general computation onto graphics by setting the output to an N×N image and drawing two screen-covering triangles so each pixel became one "kernel" invocation (ray tracing, sparse solvers, lattice simulations). Stanford's Brook language (Buck 2004) abstracted this as streams + kernels and compiled down to shader + drawTriangles calls:

```c
kernel void scale(float amount, float a<>, out float b<>) { b = amount * a; }
float input_stream<1000>, output_stream<1000>;
scale(scale_amount, input_stream, output_stream);
```

### Compute mode and CUDA (2007)
NVIDIA's Tesla architecture added a non-graphics path: the driver loads a single kernel binary, allocates device buffers, and the application issues `launch(kernel, N)`. CUDA is a C-like language over this interface. Design goal: *low abstraction distance* — constructs map directly to hardware capabilities.

### CUDA host vs device code
Partitioning is static. Host code runs on the CPU (serial C/C++), device code is marked `__global__` (kernel entry) or `__device__` (device-only helper). Kernel launch uses triple-angle syntax with a grid size and block size:

```c
const int Nx = 12, Ny = 6;
dim3 threadsPerBlock(4, 3);
dim3 numBlocks(Nx/threadsPerBlock.x, Ny/threadsPerBlock.y);
matrixAdd<<<numBlocks, threadsPerBlock>>>(A, B, C);  // launches 72 threads

__global__ void matrixAdd(float A[Ny][Nx], float B[Ny][Nx], float C[Ny][Nx]) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    C[j][i] = A[j][i] + B[j][i];
}
```

The call is bulk and blocks until all threads finish. If grid size is not a multiple of block size, round up and guard with `if (i<Nx && j<Ny)` in the kernel.

### Memory model
Host and device have separate address spaces. `cudaMalloc` allocates on the device; `cudaMemcpy(..., cudaMemcpyHostToDevice)` moves bytes. Dereferencing a `deviceA[i]` pointer from host code is undefined.

Within a kernel, three address spaces are visible:
- **Per-thread private** (registers / local memory) — thread-local variables.
- **Per-block shared** (`__shared__`) — on-chip, fast, all threads in the block see it; lifetime = block.
- **Global** — device DRAM, visible to every thread and persists across kernels.

### 1D convolution — shared memory optimization
Version 1, one thread per output, reads inputs directly from global memory:

```c
#define THREADS_PER_BLK 128

__global__ void convolve(int N, float* input, float* output) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float result = 0.0f;
    for (int i=0; i<3; i++) result += input[index + i];
    output[index] = result / 3.f;
}
```

Each block issues `3 * 128 = 384` loads. Version 2 cooperatively stages inputs into `__shared__` memory so the block only does 130 global loads:

```c
__global__ void convolve(int N, float* input, float* output) {
    __shared__ float support[THREADS_PER_BLK+2];
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    support[threadIdx.x] = input[index];
    if (threadIdx.x < 2)
        support[THREADS_PER_BLK + threadIdx.x] = input[index + THREADS_PER_BLK];

    __syncthreads();  // wait for all loads

    float result = 0.0f;
    for (int i=0; i<3; i++) result += support[threadIdx.x + i];
    output[index] = result / 3.f;
}
```

`__syncthreads()` is a per-block barrier; without it later threads might read uninitialized slots of `support`. Launch: `convolve<<<N/THREADS_PER_BLK, THREADS_PER_BLK>>>(N, devInput, devOutput);`

### Compiled binary carries resource requirements
Each compiled kernel ships with (i) instructions, (ii) threads per block, (iii) bytes of per-thread local data, (iv) bytes of shared memory per block. For `convolve` v2: 128 threads, 520 B shared, some local mem per thread. The scheduler uses these to decide placement.

### Dynamic block scheduling — walk-through
Assume a fictitious 2-core GPU where each core holds 384 thread contexts and 1.5 KB of shared memory. A kernel launch creates 1000 blocks; the work-scheduler pulls them off and assigns to cores in order, reserving 128 contexts + 520 B per block. Only two blocks fit per core (3 × 520 > 1.5 KB). When a block finishes, its resources free up and the scheduler drops in the next pending block. The same binary runs on a 6-core or 80-core chip with no changes — there is no `num_cores` in CUDA. This is analogous to ISPC's task pool, or a web-server thread pool sized to cores.

### V100 SM microarchitecture
Each V100 SM has four sub-cores. A sub-core has:
- One warp selector (fetch/decode) picking among up to 16 resident warps.
- 16-wide SIMD fp32 ALUs (32-wide op every 2 clocks), 16-wide int ALUs (same), 8-wide fp64, 2 tensor cores, load/store units.
- 64 KB of scalar register file partitioned across its warps.

Per SM: 256 KB registers, 128 KB shared/L1 combined, up to 64 warps (= 2048 CUDA threads). Per chip: 80 SMs × 4 × 16 fp32 lanes = 5120 fp32 MAD ALUs at 1.245 GHz = 12.7 TFLOPs; up to 163,840 interleaved CUDA threads; 6 MB L2; 16 GB HBM at 900 GB/s (4096-bit bus).

### Warps and SIMT
When a block launches, threads 0–31 become warp 0, 32–63 warp 1, etc. A 128-thread block = 4 warps. Each clock, a sub-core picks one runnable warp and issues its next instruction; because the 16-wide ALU runs a 32-wide warp over 2 clocks, the issue rate is effectively one warp instruction every 2 clocks per sub-core. If threads in the same warp diverge (take different branches), the lanes serialize and some are masked off — the "single instruction, multiple thread" model.

Unlike ISPC gangs, the CUDA program is *not* compiled to SIMD instructions. The hardware dynamically detects that 32 threads share an instruction and executes them as a SIMD group.

### Why all block threads must be co-resident
CUDA semantics say threads in a block run concurrently — they can `__syncthreads()` and pass data through shared memory. The implementation therefore allocates register state and shared memory for *every* thread in the block before the block starts, so no thread can be stuck waiting for a resource held by a sibling. Blocks, in contrast, are independent — the runtime is free to reorder or serialize them.

### Atomics and block-ordering pitfalls
Histograms are fine: each block atomically bumps shared counters; block order doesn't matter.

```c
atomicAdd(&counts[A[i]], 1);
```

But code that assumes block 0 runs before block 1 (e.g., a spin-wait on a flag set by block 0) can deadlock on a GPU that runs only one block at a time — block 1 spins forever because block 0 never gets scheduled.

### Persistent-thread style (bonus)
Launch exactly enough blocks to fill the GPU (e.g., `80 * (32*64/128) = 1280` blocks for V100), then each block loops pulling work indices from a global `atomicInc` counter. Gives the programmer explicit control but bakes machine-specific assumptions (that *all* blocks are concurrently resident) into the program.

## Examples / Worked Problems
- **1D convolution v1 vs v2** — shared memory reduces global loads from `3 * 128 = 384` to `130` per block (~3× bandwidth reduction).
- **Two-core GPU scheduling** — 1000 blocks, 2 cores, 2 blocks/core fit: trace blocks 0,2 to core 0 and 1,3 to core 1, block 0 finishes, block 4 takes its slot (contexts 0–127), then block 2 finishes and block 5 moves into contexts 128–255.
- **Block independence gotcha** — two-block flag signaling; whether it deadlocks depends on whether the implementation can run both concurrently. CUDA guarantees nothing here.
- **Histogram with atomics** — legal regardless of block order; atomics used only for mutual exclusion.

## Takeaways
1. GPUs started as shader engines for streams of pixels; the compute mode (CUDA) just exposes the same SIMD + multi-threaded cores through a general-purpose API.
2. CUDA = SPMD thread hierarchy + distributed host/device memory + per-thread/per-block/global device address spaces + barrier/atomic primitives.
3. Threads in a block are concurrent and cooperating (shared memory + `__syncthreads`); blocks are independent and scheduled in any order — this is what lets a single binary scale across GPU sizes.
4. Warps (32 threads, SIMT) are an implementation detail, but divergence and memory-access patterns at warp granularity dominate real performance.
5. V100 SM = 4 sub-cores × (16-wide fp32 SIMD + 16 resident warps) with 256 KB regs and 128 KB shared/L1; 80 SMs/chip give 12.7 TFLOPs and 163 K interleaved threads.
6. Use shared memory to amortize global loads across a block, and size blocks/shared usage so the scheduler can keep many blocks resident per SM (latency hiding via warp interleaving).

## Open Questions / Follow-ups
- Detailed treatment of memory coalescing, bank conflicts, and occupancy tuning — later in the course.
- Tensor cores for deep learning (hundreds of TFLOPs at reduced precision) — deferred to later lecture.
- Graphics pipeline fixed-function hardware — covered in CS248a / CS348K.
- How CUDA threads really differ from pthreads (stack allocation, scheduling, blocking calls) — promised in class, details to come.

## Sources
- Slides: `slices/07_gpuarch.pdf` (Stanford CS149, Fall 2025, Profs. Kayvon Fatahalian & Kunle Olukotun).
- Raw text: `/Users/anekoique/cources/CS149/notes/.raw/07_gpuarch.txt`.
- Referenced: Buck et al., Brook for GPUs (SIGGRAPH 2004); Harris 2002 (Coupled Map Lattice); Bolz 2003 (Sparse Solvers); Purcell 2002 (Ray Tracing on Programmable GPUs); NVIDIA V100 whitepaper.
