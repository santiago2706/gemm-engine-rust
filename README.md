# Rust GEMM Lab

> Learning Computer Architecture by Building the Core of Modern AI.

## Overview

**Rust GEMM Lab** is a project focused on understanding modern AI systems by implementing a high-performance matrix multiplication engine entirely in Rust.

Instead of relying on optimized libraries, every optimization is built from scratch to learn how hardware accelerates artificial intelligence.

The project follows the same optimization path used by real AI libraries such as cuBLAS, oneDNN and BLIS.

---

## Goals

* Learn Rust through a real systems project.
* Understand Computer Architecture concepts.
* Explore CPU cache optimization.
* Implement SIMD (AVX2/FMA).
* Build multithreaded matrix multiplication.
* Learn quantization techniques.
* Understand GPU compute and tiled algorithms.
* Compare against industrial BLAS libraries.

---

## Roadmap

### Phase 1 — Core Engine

* [ ] Project setup
* [ ] Matrix data structure
* [ ] Matrix indexing
* [ ] Matrix constructors
* [ ] Unit tests

### Phase 2 — Basic Matrix Operations

* [ ] Addition
* [ ] Subtraction
* [ ] Transpose
* [ ] Naive Matrix Multiplication

### Phase 3 — Performance Optimizations

* [ ] SIMD (AVX2)
* [ ] Cache Blocking
* [ ] Multithreading (Rayon)

### Phase 4 — Advanced Topics

* [ ] Quantization (INT8)
* [ ] GPU Compute
* [ ] Roofline Analysis
* [ ] BLAS Comparison

---

## Technologies

* Rust
* Cargo
* Rayon
* Criterion
* GitHub Actions

---

## Repository Structure

```text
src/
tests/
benches/
docs/
scripts/
results/
```

---

## Current Status

🚧 Phase 1 — Project Initialization

---

## Learning Objectives

By the end of this project I expect to understand:

* Computer Architecture
* Cache Locality
* SIMD Programming
* Parallel Programming
* High Performance Computing
* GPU Programming
* AI Inference Optimization

---

## License

MIT License
