use std::time::Instant;

pub struct BenchResult {
    pub strategy: &'static str,
    pub n: usize,
    pub elapsed_ms: f64,
    pub gflops: f64,
    pub checksum: f64,
}

fn gen_matrix(n: usize, seed: u64) -> Vec<f32> {
    (0..n * n)
        .map(|idx| {
            let r = idx / n;
            let c = idx % n;
            (((r as u64) * 31 + (c as u64) * 17 + seed) % 13) as f32
        })
        .collect()
}

fn checksum(data: &[f32]) -> f64 {
    data.iter().map(|&v| v as f64).sum()
}

fn gflops(n: usize, elapsed_ms: f64) -> f64 {
    let flops = 2.0 * (n as f64).powi(3);
    let seconds = elapsed_ms / 1000.0;
    if seconds <= 0.0 {
        return 0.0;
    }
    flops / seconds / 1e9
}

// Base: orden i-j-k, acceso a B por columnas (con salto), sin optimizar.
fn matmul_naive_ijk(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for k in 0..n {
                sum += a[i * n + k] * b[k * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// SIMD: orden i-k-j. El bucle interno sobre j accede a B y C de forma contigua,
// lo que permite que el compilador auto-vectorice la operación (SSE/AVX) en modo release.
fn matmul_ikj(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    for i in 0..n {
        for k in 0..n {
            let a_ik = a[i * n + k];
            let row_b = &b[k * n..k * n + n];
            let row_c = &mut c[i * n..i * n + n];
            for j in 0..n {
                row_c[j] += a_ik * row_b[j];
            }
        }
    }
    c
}

// Cache Blocking: mismo orden i-k-j, pero particionado en bloques que caben en caché
// para reutilizar los datos cargados antes de descartarlos.
fn matmul_blocked(a: &[f32], b: &[f32], n: usize, block: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; n * n];
    let mut ii = 0;
    while ii < n {
        let i_end = (ii + block).min(n);
        let mut jj = 0;
        while jj < n {
            let j_end = (jj + block).min(n);
            let mut kk = 0;
            while kk < n {
                let k_end = (kk + block).min(n);
                for i in ii..i_end {
                    for k in kk..k_end {
                        let a_ik = a[i * n + k];
                        for j in jj..j_end {
                            c[i * n + j] += a_ik * b[k * n + j];
                        }
                    }
                }
                kk += block;
            }
            jj += block;
        }
        ii += block;
    }
    c
}

// Parallel: reparte las filas del resultado entre varios hilos reales del sistema operativo.
fn matmul_parallel(a: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let num_threads = std::thread::available_parallelism()
        .map(|v| v.get())
        .unwrap_or(4)
        .clamp(1, n.max(1));
    let rows_per_thread = n.div_ceil(num_threads);

    let mut c = vec![0.0f32; n * n];
    std::thread::scope(|scope| {
        for (t, chunk) in c.chunks_mut(rows_per_thread * n).enumerate() {
            let row_start = t * rows_per_thread;
            scope.spawn(move || {
                let rows_in_chunk = chunk.len() / n;
                for local_i in 0..rows_in_chunk {
                    let i = row_start + local_i;
                    for k in 0..n {
                        let a_ik = a[i * n + k];
                        let row_b = &b[k * n..k * n + n];
                        let row_c = &mut chunk[local_i * n..local_i * n + n];
                        for j in 0..n {
                            row_c[j] += a_ik * row_b[j];
                        }
                    }
                }
            });
        }
    });
    c
}

pub fn run_benchmark(strategy: &str, n: usize) -> BenchResult {
    let a = gen_matrix(n, 1);
    let b = gen_matrix(n, 2);

    let (name, start, result): (&'static str, Instant, Vec<f32>) = match strategy {
        "simd" => {
            let start = Instant::now();
            let r = matmul_ikj(&a, &b, n);
            ("simd", start, r)
        }
        "cache" => {
            let block = 128.min(n).max(1);
            let start = Instant::now();
            let r = matmul_blocked(&a, &b, n, block);
            ("cache", start, r)
        }
        "parallel" => {
            let start = Instant::now();
            let r = matmul_parallel(&a, &b, n);
            ("parallel", start, r)
        }
        _ => {
            let start = Instant::now();
            let r = matmul_naive_ijk(&a, &b, n);
            ("base", start, r)
        }
    };

    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    BenchResult {
        strategy: name,
        n,
        elapsed_ms,
        gflops: gflops(n, elapsed_ms),
        checksum: checksum(&result),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_strategies_agree() {
        let n = 33; // tamaño no múltiplo del bloque, a propósito
        let a = gen_matrix(n, 1);
        let b = gen_matrix(n, 2);

        let base = matmul_naive_ijk(&a, &b, n);
        let simd = matmul_ikj(&a, &b, n);
        let cache = matmul_blocked(&a, &b, n, 8);
        let parallel = matmul_parallel(&a, &b, n);

        for i in 0..base.len() {
            assert!((base[i] - simd[i]).abs() < 1e-2);
            assert!((base[i] - cache[i]).abs() < 1e-2);
            assert!((base[i] - parallel[i]).abs() < 1e-2);
        }
    }
}
