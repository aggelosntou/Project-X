# Project-X

**Author:** Angelos Ntousis  
**Background:** Mathematician  
**Target:** Top 0.1% ML Systems Engineer / Scientist  
**Timeline:** Pre-masters  

---

> *"The biggest lesson from 70 years of AI research is that general methods that leverage computation are ultimately the most effective — and by a large margin."*
> — Richard Sutton, The Bitter Lesson

This is not a collection of exercises.  
This is a structured, documented engineering transformation — from mathematician to someone who understands computing at the deepest level, before entering a masters program in ML/AI.

The goal is not to finish this repository. The goal is to become the kind of engineer who could have written it from first principles.

---

## What this is

Project-X is a pre-masters curriculum built around one conviction:

**The engineers who will define the next decade of AI are those who understand the full stack — from transistors to transformers — not just how to call an API.**

This repository tracks the journey of building that understanding, one project at a time, with every decision documented and every concept understood — not just implemented.

---

## The Target

To become a top 0.1% ML Systems engineer means being able to:

- Implement a neural network from scratch in C, understanding every byte
- Understand why a transformer runs slowly and fix it at the hardware level
- Design and run a distributed training experiment from first principles
- Read a systems paper (CUDA kernel optimization, memory bandwidth analysis, quantization) and implement the ideas
- Enter a masters program already thinking like a researcher, not a student
- Build ambitious systems that other engineers consider hard

This repository is the documented path to that level.

---

## Structure of the Journey

The journey is organized into **six phases**, each building on the last.  
No phase is skipped. No phase is rushed.

```
Phase 0 → Language & Tooling Foundations        [C, Python, tooling]
Phase 1 → Systems Programming                   [memory, OS, networks, concurrency]
Phase 2 → Computer Architecture                 [hardware, CPUs, GPUs, compilers, databases]
Phase 3 → AI & ML from Scratch                  [math → code → systems]
Phase 4 → ML Systems Engineering                [the book comes alive]
Phase 5 → Frontier ML Systems                   [LLM inference, GPU kernels, distributed training, alignment]
```

Each phase has concrete projects, clear completion criteria, and documented learning.

---

## Phase 0 — Language and Tooling Foundations
**Goal:** Be dangerous in C and Python. Understand the tools of the trade.

This phase is about removing the friction between an idea and an implementation.  
A mathematician who cannot code fluently is a mathematician. This phase ends that.

### Projects

| Project | Language | What it teaches |
|---|---|---|
| `hello-systems` | C | Compilation pipeline, Makefile, GCC flags, linking, undefined behavior |
| `pointers-and-memory` | C | Pointers, pointer arithmetic, stack vs heap, segfaults, valgrind |
| `string-library` | C | Manual memory management, buffer handling, null termination |
| `python-numpy-from-scratch` | Python | NumPy internals intuition, vectorization, broadcasting |
| `data-structures-c` | C | Linked list, hash map, dynamic array — by hand, with tests |

### Completion criteria
- Can write a C program from scratch, compile it, debug it with GDB, and check it with valgrind
- Understands the difference between a pointer and a value, a stack frame and a heap allocation
- Can use NumPy without documentation for standard operations
- Has read: *The C Programming Language* (K&R), chapters 1–6

---

## Phase 1 — Systems Programming
**Goal:** Understand how an operating system works by building pieces of one. Understand how concurrent programs fail and how to make them correct.

This is where most ML engineers have a blind spot.  
Understanding processes, files, sockets, and memory at the OS level is what separates people who debug production systems from people who restart them.  
Concurrency is not optional for ML Systems — distributed training, data pipelines, and inference servers are all concurrent programs. This phase builds the foundation to reason about them correctly.

### Projects

| Project | Language | What it teaches |
|---|---|---|
| `memory-allocator` | C | `malloc`/`free` from scratch, free lists, fragmentation, `sbrk` |
| `unix-shell` | C | `fork`, `exec`, `wait`, pipes, file descriptors, signals |
| `http-server` | C | TCP sockets, HTTP/1.1 parsing, concurrent connections, `select`/`epoll` |
| `file-system-toy` | C | Inodes, directories, block allocation — a tiny FS on a file |
| `tcp-server` | Rust | Ownership model, async I/O, safe systems programming |
| `thread-pool` | C | Pthreads, mutexes, condition variables, work queues, thread lifecycle |
| `lock-free-queue` | C | Atomics, memory ordering, CAS operations, ABA problem |
| `parallel-matrix-multiply` | C | OpenMP, thread scaling, Amdahl's Law measured in practice |

### Why concurrency here
Every ML Systems bottleneck is ultimately a concurrency problem. GPU utilization at 60% instead of 95% is a data loading concurrency problem. A distributed training run that hangs is a synchronization problem. An inference server with poor throughput is a batching and threading problem. Building a thread pool and a lock-free queue from scratch gives you the mental model to diagnose all of these — not just follow a tutorial to fix them.

### Completion criteria
- Can explain what happens when you run `./program` from the shell, step by step
- Understands file descriptors, process isolation, and virtual memory
- Has implemented `malloc` and understands why it's hard to do well
- Can write a server that handles concurrent connections without crashing
- Understands mutexes, condition variables, and when each is appropriate
- Can explain what a race condition is, produce one deliberately, and fix it
- Understands memory ordering: what `acquire`, `release`, and `seq_cst` mean and why it matters
- Has measured Amdahl's Law directly — seen parallelism hit its ceiling on real hardware
- Has read: *Computer Systems: A Programmer's Perspective* (CS:APP), chapters 1–12
- Has read: *The Art of Multiprocessor Programming* — Herlihy & Shavit, chapters 1–7

---

## Phase 2 — Computer Architecture
**Goal:** Understand what hardware actually executes. Think in memory, latency, and bandwidth. Understand how languages become machine code. Understand how data systems store and retrieve information.

The ML Systems book makes one thing clear: **the bottleneck is memory bandwidth, not compute**.  
You cannot optimize what you do not understand.  
This phase builds hardware intuition — what is fast, what is slow, and why — and adds two critical capabilities: understanding how ML frameworks and compilers work underneath, and how data systems are built, since every ML pipeline is ultimately a data management problem.

### Projects

| Project | Language | What it teaches |
|---|---|---|
| `cpu-emulator` | C | Fetch-decode-execute cycle, registers, ALU, instruction encoding |
| `cache-experiments` | C | L1/L2/L3 behavior, cache lines, false sharing, memory access patterns |
| `matrix-multiply-optimized` | C | SIMD, loop tiling, cache-oblivious algorithms, `perf` profiling |
| `bytecode-vm` | C++ | Stack machine, bytecode design, interpreter loop, JIT concepts |
| `gpu-intro` | CUDA/C | Thread hierarchy, shared memory, memory coalescing, first CUDA kernel |
| `mini-compiler` | C | Lexer, parser, AST, IR design, code generation for a small language |
| `mini-database` | C | B-tree storage engine, simple query execution, disk I/O, indexing |

### Why a compiler here
`torch.compile()`, TVM, XLA, Triton — every modern ML optimization tool is a compiler. When you write a custom CUDA kernel in Phase 4 or read the FlashAttention paper, you are doing compiler work. The mini-compiler project teaches lexing, parsing, IR design, and code generation — the four concepts that underpin everything from PyTorch's graph compiler to LLVM-based model optimization. Without this, those tools are black boxes. With it, they are legible.

### Why a database here
Every ML pipeline is a data management problem. Feature stores, vector databases, training data versioning, streaming data loaders — these are all built on the same ideas as a storage engine. Understanding B-trees, storage engines, and disk I/O patterns directly informs how to build efficient data pipelines, why certain data formats (Parquet, Arrow) are fast, and how vector databases work for retrieval-augmented generation. The mini-database project makes all of this concrete.

### Completion criteria
- Can read `perf stat` output and understand what each counter means
- Understands why a naive matrix multiply is 10× slower than BLAS
- Has written a CUDA kernel and understands the GPU memory hierarchy
- Can explain: cache lines, SIMD, branch prediction, memory coalescing
- Can implement a lexer and parser for a simple grammar from scratch
- Understands what an IR is and why compilers use one
- Understands B-tree structure and why it is the dominant storage data structure
- Can explain the difference between a sequential scan and an index scan and when each wins
- Has read: *Computer Organization and Design* (Patterson & Hennessy), parts I–III
- Has read: *Crafting Interpreters* — Robert Nystrom (both parts: tree-walk and bytecode)
- Has read: *Database Internals* — Alex Petrov, chapters 1–6

---

## Phase 3 — AI and ML from Scratch
**Goal:** Implement every important idea in ML without using a framework.

This is where the mathematics becomes code.  
Every concept here is something most ML engineers use as a black box.  
Building it from scratch means you will never be confused by it again.

### Projects

| Project | Language | What it teaches |
|---|---|---|
| `matrix-library-c` | C | Dense matrix ops, memory layout (row vs col major), BLAS interface design |
| `autograd-engine` | Python | Automatic differentiation from scratch, computation graphs, backprop |
| `neural-net-c` | C | Forward pass, backprop, weight updates — no libraries, pure C |
| `neural-net-numpy` | Python | Same network in NumPy — connecting C intuition to Python workflow |
| `transformer-from-scratch` | Python | Attention, positional encoding, layer norm — every line understood |
| `conv-net-from-scratch` | Python | Convolutions, pooling, backprop through conv layers |
| `optimizer-zoo` | Python | SGD, Momentum, Adam, AdamW — implement and compare |
| `loss-functions` | Python | Cross-entropy, MSE, focal loss — derive gradients by hand first |

### Completion criteria
- Can implement backpropagation on paper and then in code
- Understands what a computation graph is and why it matters
- Has trained a transformer that actually converges, written from scratch
- Can explain the vanishing gradient problem and why residual connections fix it
- Has read: *Deep Learning* (Goodfellow, Bengio, Courville), all chapters
- Has read: *ML Systems* (Reddi), Parts I and II

---

## Phase 4 — ML Systems Engineering
**Goal:** Bridge deep ML knowledge with production systems thinking.

This is the target. This is the *ML Systems* book made real.  
By this phase, every concept in the book has a project behind it.

### Projects

| Project | Language | What it teaches |
|---|---|---|
| `quantization-engine` | C/Python | INT8/FP16 quantization from scratch, accuracy vs speed tradeoffs |
| `inference-server` | Python/C++ | Model serving, batching, latency vs throughput, REST API |
| `distributed-training-toy` | Python | Data parallelism, gradient synchronization, `torch.distributed` |
| `model-compression` | Python | Pruning, knowledge distillation — implement and benchmark |
| `profiler` | Python/C | Custom profiler for a training loop — FLOPS, memory, time |
| `data-pipeline` | Python | Streaming data loader, data versioning, augmentation pipeline |
| `mlops-mini` | Python | Experiment tracking, model registry, automated retraining |
| `edge-deployment` | C/Python | Export model to ONNX, run on CPU with optimized kernels |
| `cuda-kernel-custom` | CUDA | Write a fused attention kernel, benchmark vs naive PyTorch |
| `benchmark-suite` | C/Python | Reproduce MLPerf-style benchmarking methodology from scratch |

### Completion criteria
- Can take a trained model and make it 4× faster without meaningful accuracy loss
- Understands memory bandwidth constraints and designs around them
- Has run distributed training and debugged a gradient synchronization bug
- Can read a paper like FlashAttention and implement the key ideas
- Has read: *ML Systems* (Reddi), all parts
- Has read: 10+ systems papers (see reading list below)

---

## Phase 5 — Frontier ML Systems

**Goal:** Build the systems that frontier labs are running today. Close the gap between academic ML knowledge and what production research engineering actually requires in 2026.

Phase 4 teaches ML Systems as a discipline. Phase 5 is where you execute at the frontier — the algorithms, kernels, and training pipelines that define how models at the scale of GPT-4, Claude, and Gemini are built and served. Every project in this phase mirrors a real system at a top lab. Every benchmark number you produce here is the kind of number that appears in a research engineer interview.

### Projects

| Project | Language | What it mirrors |
|---|---|---|
| `llm-inference-engine` | Python, CUDA | vLLM — PagedAttention, continuous batching, speculative decoding |
| `flash-attention-cuda` | CUDA C++ | FlashAttention 1 & 2 — tiled attention, online softmax, IO analysis |
| `triton-kernels` | Triton, Python | Production kernel engineering — fused softmax, LayerNorm, cross-entropy, attention |
| `distributed-training-internals` | Python, C | Megatron-LM + ZeRO — Ring-AllReduce, tensor parallelism, optimizer sharding |
| `rlhf-pipeline` | Python | InstructGPT + DPO — reward model, PPO, direct preference optimization |

### Why Phase 5 exists

Phases 0–4 build every prerequisite. Phase 5 is where they converge. A distributed training bug is simultaneously a concurrency problem (Phase 1), a memory bandwidth problem (Phase 2), and a gradient synchronization problem (Phase 3). A fused CUDA kernel requires compiler knowledge (Phase 2), hardware understanding (Phase 2), and ML intuition (Phase 3). Phase 5 does not teach new concepts — it demands that you apply all of them simultaneously, at the level the problems require.

The projects here are not exercises. They are the systems that matter.

### Phase 5 Completion Criteria

- Can implement a paged KV cache from scratch and explain the memory fragmentation analysis
- Can write a tiled attention kernel in both CUDA and Triton and explain the performance difference
- Can implement Ring-AllReduce from sockets and trace a distributed training hang to its root cause
- Understands DPO at the mathematical level: can derive the loss from the RLHF objective by hand
- Every project has a benchmark with measurable numbers and hardware specs sufficient to reproduce them

---

## Reading List

These are not optional. They are part of the curriculum.

### Books (in order)
1. *The C Programming Language* — Kernighan & Ritchie
2. *Computer Systems: A Programmer's Perspective* — Bryant & O'Hallaron
3. *The Art of Multiprocessor Programming* — Herlihy & Shavit (chapters 1–7)
4. *Computer Organization and Design* — Patterson & Hennessy
5. *Crafting Interpreters* — Robert Nystrom (free online)
6. *Database Internals* — Alex Petrov (chapters 1–6)
7. *Deep Learning* — Goodfellow, Bengio, Courville
8. *Machine Learning Systems* — Vijay Janapa Reddi (Harvard)
9. *Designing Data-Intensive Applications* — Kleppmann
10. *The Art of Doing Science and Engineering* — Hamming

### Papers (build up to these)
- *Attention Is All You Need* — Vaswani et al. (2017)
- *FlashAttention* — Dao et al. (2022)
- *Efficient Memory Management for Large Language Model Serving* — vLLM (2023)
- *Scaling Laws for Neural Language Models* — Kaplan et al. (2020)
- *The Bitter Lesson* — Sutton (2019)
- *BERT: Pre-training of Deep Bidirectional Transformers* — Devlin et al. (2018)
- *An Image is Worth 16×16 Words* — Dosovitskiy et al. (2020)
- *Roofline: An Insightful Visual Performance Model* — Williams et al. (2009)
- *MLSys: The New Frontier of Machine Learning Systems* — Ratner et al.
- *Megatron-LM: Training Multi-Billion Parameter Language Models* — Shoeybi et al.

### Phase 5 Papers (required before each project)

- *Efficient Memory Management for Large Language Model Serving with PagedAttention* — Kwon et al. (2023) `[llm-inference-engine]`
- *Fast Inference from Transformers via Speculative Decoding* — Leviathan et al. (2023) `[llm-inference-engine]`
- *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* — Dao et al. (2022) `[flash-attention-cuda]`
- *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning* — Dao (2024) `[flash-attention-cuda, triton-kernels]`
- *Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations* — Tillet et al. (2019) `[triton-kernels]`
- *ZeRO: Memory Optimizations Toward Training Trillion Parameter Models* — Rajbhandari et al. (2020) `[distributed-training-internals]`
- *PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel* — Zhao et al. (2023) `[distributed-training-internals]`
- *GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism* — Huang et al. (2019) `[distributed-training-internals]`
- *Training language models to follow instructions with human feedback* — Ouyang et al. / InstructGPT (2022) `[rlhf-pipeline]`
- *Direct Preference Optimization: Your Language Model is Secretly a Reward Model* — Rafailov et al. (2023) `[rlhf-pipeline]`
- *Proximal Policy Optimization Algorithms* — Schulman et al. (2017) `[rlhf-pipeline]`
- *Sarathi-Serve: Efficient LLM Serving by Piggybacking Decodes with Chunked Prefills* — Agrawal et al. (2024) `[llm-inference-engine]`

---

## Repository Structure

```
Project-X/
│
├── README.md                         ← this document
│
├── phase-0-foundations/
│   ├── hello-systems/
│   ├── pointers-and-memory/
│   ├── string-library/
│   ├── python-numpy-from-scratch/
│   └── data-structures-c/
│
├── phase-1-systems/
│   ├── memory-allocator/
│   ├── unix-shell/
│   ├── http-server/
│   ├── file-system-toy/
│   ├── tcp-server-rust/
│   ├── thread-pool/
│   ├── lock-free-queue/
│   └── parallel-matrix-multiply/
│
├── phase-2-architecture/
│   ├── cpu-emulator/
│   ├── cache-experiments/
│   ├── matrix-multiply-optimized/
│   ├── bytecode-vm/
│   ├── gpu-intro/
│   ├── mini-compiler/
│   └── mini-database/
│
├── phase-3-ai-from-scratch/
│   ├── matrix-library-c/
│   ├── autograd-engine/
│   ├── neural-net-c/
│   ├── neural-net-numpy/
│   ├── transformer-from-scratch/
│   ├── conv-net-from-scratch/
│   ├── optimizer-zoo/
│   └── loss-functions/
│
├── phase-4-ml-systems/
│   ├── quantization-engine/
│   ├── inference-server/
│   ├── distributed-training-toy/
│   ├── model-compression/
│   ├── profiler/
│   ├── data-pipeline/
│   ├── mlops-mini/
│   ├── edge-deployment/
│   ├── cuda-kernel-custom/
│   └── benchmark-suite/
│
├── phase-5-frontier/
│   ├── llm-inference-engine/         ← PagedAttention, continuous batching, speculative decoding
│   ├── flash-attention-cuda/         ← FlashAttention 1 & 2 in raw CUDA
│   ├── triton-kernels/               ← Fused softmax, LayerNorm, attention, cross-entropy
│   ├── distributed-training-internals/ ← Ring-AllReduce, tensor parallelism, ZeRO
│   └── rlhf-pipeline/               ← Reward model + PPO + DPO from scratch
│
├── docs/
│   ├── notes/                        ← concept notes per topic
│   ├── paper-reviews/                ← written reviews of papers read
│   ├── design-decisions/             ← why certain implementations were chosen
│   └── learning-log/                 ← weekly progress entries
│
└── assets/
```

### Every project must contain
- `README.md` — what it is, why it exists, how to build and run it
- `src/` — the code
- `LEARNINGS.md` — what concepts this project taught, what was hard, what tradeoffs were made
- `BENCHMARKS.md` — where relevant: measurements, profiling results, comparisons

---

## Standards

This repository holds itself to one standard:

**Every line of code here was understood before it was written.**

This means:
- No copy-pasting without being able to delete it and rewrite it from memory
- No using a library function without knowing what it does underneath
- No "it works, I'll move on" — if it works but you cannot explain why, it is not finished
- Every project has a written explanation a smart non-expert could follow

The mathematics background is an advantage here. The instinct to prove things before accepting them applies directly.

---

## Philosophy

**Why C first?**  
Because C does not hide anything. Every allocation is explicit. Every pointer is visible. The programmer is responsible for everything. Learning ML systems in Python without this is like doing calculus without understanding limits — you can follow the rules, but you cannot reason about the edges.

**Why concurrency in Phase 1?**  
Because every ML Systems bottleneck — slow data pipelines, underutilized GPUs, hanging distributed training runs — is a concurrency problem at its root. Building a thread pool and a lock-free queue from scratch gives you the mental model to diagnose these problems, not just restart the job and hope.

**Why a compiler in Phase 2?**  
Because `torch.compile`, TVM, Triton, and XLA are all compilers. The engineers who can extend, debug, and reason about these tools are the ones who built something similar first. A mini-compiler turns the most important ML optimization tools from black boxes into readable systems.

**Why a database in Phase 2?**  
Because every ML pipeline is a data management problem. Feature stores, vector databases, streaming data loaders, training data versioning — these are all built on the same ideas as a storage engine. Understanding B-trees and disk I/O directly informs every data engineering decision in Phase 4.

**Why build from scratch instead of using frameworks?**  
Because the engineers who will matter in ten years are not the ones who learned the PyTorch API. They are the ones who could have written parts of it. A from-scratch neural network in C is not useful for production. It is useful for the engineer who wrote it.

**Why document everything?**  
Because writing forces clarity. You do not understand something until you can explain it simply. Every `LEARNINGS.md` is a forcing function: if you cannot write it down, you did not learn it.

**Why this before a masters?**  
Because the difference between a masters student who knows this material and one who does not is not one semester. It is the difference between someone who can immediately contribute to research and someone who spends the first year catching up. The goal is to arrive already thinking at the level the program expects to teach.

---

## Completion Definition

Project-X is complete when:

1. Every phase has at least its core projects finished and documented
2. The reading list is done — books read, papers reviewed and written up
3. A systems paper can be read, understood, and partially implemented within a week
4. The Phase 4 projects demonstrate measurable, benchmarked results
5. The repository tells a coherent story: a mathematician who became an engineer, one project at a time

That is the portfolio. That is the proof.

---

## Status

> Phase 0 — In progress  
> Phase 1 — Not started  
> Phase 2 — Not started  
> Phase 3 — Not started  
> Phase 4 — Not started  
> Phase 5 — Scaffolded (READMEs complete, implementation pending)  

*This repository is actively being built. Each phase will be updated as projects are completed.*

---

*Built by Angelos Ntousis — mathematician, engineer in progress.*
