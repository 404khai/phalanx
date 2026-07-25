//! Microbenchmarks for Phase 2 reference kernels.
//!
//! These establish a baseline before SIMD / threading work in later phases.
//! Run with `cargo bench --bench tensor_ops`.

// Criterion macros expand helpers without doc comments; this file is not a public API.
#![allow(missing_docs)]

use criterion::{Criterion, criterion_group, criterion_main};
use phalanx::{Shape, Tensor};

fn bench_elemwise_add(c: &mut Criterion) {
    let n = 1_024 * 1_024; // ~1M elements — stress bandwidth more than ALU
    let a = Tensor::full([n], 1.0).expect("alloc");
    let b = Tensor::full([n], 2.0).expect("alloc");

    c.bench_function("tensor_add_1m", |bencher| {
        bencher.iter(|| a.add(&b).expect("add"));
    });
}

fn bench_matmul(c: &mut Criterion) {
    // 256³ mul-adds: large enough to see trends, small enough for CI laptop runs.
    let n = 256usize;
    let a = Tensor::full([n, n], 1.0).expect("alloc");
    let b = Tensor::full([n, n], 1.0).expect("alloc");

    c.bench_function("tensor_matmul_256", |bencher| {
        bencher.iter(|| a.matmul(&b).expect("matmul"));
    });
}

fn bench_transpose(c: &mut Criterion) {
    let rows = 512usize;
    let cols = 512usize;
    let data = vec![1.0f32; rows * cols];
    let t = Tensor::from_vec(data, Shape::new([rows, cols]).expect("shape")).expect("tensor");

    c.bench_function("tensor_transpose_512", |bencher| {
        bencher.iter(|| t.transpose().expect("transpose"));
    });
}

criterion_group!(benches, bench_elemwise_add, bench_matmul, bench_transpose);
criterion_main!(benches);
