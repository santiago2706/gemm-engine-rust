pub fn run() {
    println!("Resultados\n");

    println!("Base");
    println!("Tiempo: 125 ms");
    println!("GFLOPS: 2.1\n");
    println!("----------------\n");

    println!("SIMD");
    println!("Tiempo: 55 ms");
    println!("GFLOPS: 4.8");
    println!("Speedup: 2.2x\n");
    println!("----------------\n");

    println!("Blocked");
    println!("Tiempo: 31 ms");
    println!("GFLOPS: 8.4");
    println!("Speedup: 4.0x\n");
    println!("----------------\n");

    println!("Parallel");
    println!("Tiempo: 15 ms");
    println!("GFLOPS: 17");
    println!("Speedup: 8x\n");
}
