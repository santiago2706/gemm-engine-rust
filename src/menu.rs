use std::io::{self, Write};

use crate::benchmark;
use crate::matrix::Matrix;
use crate::simulation;
use crate::visualizer;
use crate::webserver;

pub enum EngineVersion {
    Base,
    Simd,
    CacheBlocking,
    Parallel,
}

enum Choice {
    Version(EngineVersion),
    WebMode,
}

pub fn run() {
    print_menu();
    let choice = loop {
        match read_choice() {
            Some(1) => break Choice::Version(EngineVersion::Base),
            Some(2) => break Choice::Version(EngineVersion::Simd),
            Some(3) => break Choice::Version(EngineVersion::CacheBlocking),
            Some(4) => break Choice::Version(EngineVersion::Parallel),
            Some(5) => break Choice::WebMode,
            _ => {
                print!("Opción inválida. Seleccione: ");
                io::stdout().flush().unwrap();
            }
        }
    };

    match choice {
        Choice::Version(version) => execute(version),
        Choice::WebMode => webserver::run(),
    }
}

fn print_menu() {
    println!("==========================");
    println!(" AI Matrix Engine");
    println!("==========================\n");
    println!("1. Base");
    println!("2. SIMD");
    println!("3. Cache Blocking");
    println!("4. Parallel");
    println!("5. Modo Web (benchmark real)\n");
    print!("Seleccione: ");
    io::stdout().flush().unwrap();
}

fn read_choice() -> Option<u32> {
    let mut input = String::new();
    io::stdin().read_line(&mut input).ok()?;
    input.trim().parse::<u32>().ok()
}

fn execute(version: EngineVersion) {
    let name = match version {
        EngineVersion::Base => "Base",
        EngineVersion::Simd => "SIMD",
        EngineVersion::CacheBlocking => "Cache Blocking",
        EngineVersion::Parallel => "Parallel",
    };
    println!("\n> Ejecutando motor: {}\n", name);

    let a = Matrix::from_vec(4, 4, (1..=16).map(|x| x as f32).collect());
    let b = Matrix::from_vec(4, 4, (1..=16).rev().map(|x| x as f32).collect());

    a.print("Matriz A");
    b.print("Matriz B");

    let result = a.multiply_verbose(&b);

    result.print("Resultado");

    println!("--- Simulación de la estrategia: {} ---\n", name);
    match version {
        EngineVersion::Base => simulation::base::run(),
        EngineVersion::Simd => simulation::simd::run(),
        EngineVersion::CacheBlocking => simulation::cache::run(),
        EngineVersion::Parallel => simulation::parallel::run(),
    }

    benchmark::run();

    visualizer::run();

    print_conclusion();
}

fn print_conclusion() {
    println!("Conclusión\n");
    println!("✔ Todas producen el mismo resultado.\n");
    println!("✔ SIMD procesa varios datos simultáneamente.\n");
    println!("✔ Cache Blocking reduce accesos a memoria.\n");
    println!("✔ Parallel usa múltiples núcleos.\n");
    println!("Estas son las mismas técnicas utilizadas para acelerar");
    println!("las multiplicaciones de matrices en modelos de IA.");
}
