# Motor de multiplicación de matrices en Rust — aprender arquitectura de computadoras construyendo el corazón de la IA

> **Documento único de construcción.** Fusiona y reemplaza a `estructura_demo.md` (el *qué*) y `plan_construccion_demo.md` (el *cómo*). Reescrito en **Rust** y reencuadrado alrededor de una idea: la multiplicación de matrices densas (GEMM) **es** la operación que domina el cómputo de las redes neuronales y los transformers. Optimizarla a mano, peldaño a peldaño, es la forma más concreta de entender *cómo corre la IA sobre el hardware*.
>
> **Filosofía de uso (no cambió):** no le pases esto a un agente de golpe. Pídele **una tarea con su test**, verifícala, y **pídele que te explique el porqué** (sobre todo en SIMD, bloqueo, hilos y roofline). Eso es lo que vas a defender en la exposición y lo que convierte la demo en aprendizaje real.

---

## 0. Por qué este proyecto enseña IA de verdad

Una capa lineal de una red neuronal es `Y = X·W + b`: una multiplicación de matrices. Un bloque de atención de un transformer son varias multiplicaciones de matrices (Q·Kᵀ, luego ·V). Entrenar e inferir un LLM es, en >90% del tiempo de cómputo, **GEMM** (*General Matrix Multiply*). Por eso empresas enteras (NVIDIA con cuBLAS, Intel con oneDNN) llevan décadas afinando esta única operación.

La escalera que vas a construir —escalar → vectorizado → bloqueado → multihilo → (cuantizado) → (GPU)— es **exactamente** cómo escala el cómputo de IA real: del paralelismo de datos dentro de un núcleo, a la jerarquía de memoria, a los muchos núcleos, a la precisión reducida, a las miles de hebras de una GPU. Cada versión añade un concepto de arquitectura y un concepto de cómo se acelera la IA hoy.

**Qué vas a aprender (mapa):** localidad y líneas de caché · SIMD / paralelismo de datos · jerarquía de memoria y *tiling* · paralelismo de hilos y seguridad de Rust · no asociatividad del punto flotante (clave para la reproducibilidad en ML) · el modelo *roofline* e intensidad operacional (cómo se decide si un kernel está limitado por cómputo o por memoria) · cuantización · jerarquía de memoria de la GPU.

---

## 1. La escalera de optimización (la columna vertebral del proyecto)

| # | Versión | Concepto de arquitectura | Conexión con la IA de hoy | Núcleo / Opc |
|---|---|---|---|---|
| 1 | `matmul_base` (escalar, orden `ikj`) | Localidad espacial, líneas de caché | La referencia honesta; denominador del speedup | Núcleo |
| 2 | `matmul_simd` (AVX2 + FMA) | **DLP**: paralelismo a nivel de datos | Las unidades vectoriales que la CPU usa para inferir | Núcleo |
| 3 | `matmul_blocked` (*tiling* + SIMD) | Jerarquía de memoria, reúso, *working set* | Por qué el ancho de banda manda; antesala del *tiling* en GPU | Núcleo |
| 4 | `matmul_parallel` (rayon) | **TLP**: paralelismo de hilos | Cómo escala el entrenamiento entre núcleos/máquinas | Núcleo |
| 5 | `matmul_quant_i8` (int8) | Precisión vs. throughput | Cuantización de LLMs (int8/fp16/bf16) | Opcional |
| 6 | `matmul_gpu` (wgpu, *naive* → *tiled*) | Paralelismo masivo, memoria compartida | Dónde corre la IA de verdad | Opcional |
| — | Comparación contra BLAS real | Límite práctico del afinado a mano | cuBLAS/oneDNN: décadas de optimización | Opcional |

> **Idea que se repite:** el *bloqueo de caché* en CPU y el *tiling con memoria compartida* en GPU son **el mismo truco** —traer un sub-bloque a la memoria rápida y reutilizarlo al máximo antes de devolverlo—. Si entiendes la versión 3, entiendes el 80% de la versión 6.

---

## 2. Stack tecnológico (Rust)

Versiones = mínimos recomendados; ajústalos a lo que tengas.

| Capa | Herramienta | Versión mín. | Para qué se usa aquí |
|---|---|---|---|
| Lenguaje | **Rust** (edición 2021) | 1.75 | Las versiones de matmul; seguridad de memoria sin recolector de basura |
| Toolchain | rustup + cargo | — | Compilar, testear y *benchmarkear* con un solo comando |
| Vectorización | `std::arch::x86_64` (AVX2 + FMA) | — | Intrínsecos explícitos: ves el registro de 256 bits y el FMA |
| Hilos | `rayon` | 1.x | Paralelismo de datos sobre filas/bloques, sin *data races* posibles |
| Tests | `cargo test` (integrados) | — | Verificación de equivalencia numérica, sin dependencias |
| Benchmark | `criterion` + un binario propio | 0.5 | Rigor estadístico (warm-up, outliers) + CSV para el roofline |
| Medición | `std::time::Instant` | — | Reloj monotónico de alta resolución **incluido en std** |
| Perfilado | `cargo flamegraph`, `perf`, Cachegrind | — | Fallos de caché y puntos calientes para la Discusión |
| Gráficos | Python + pandas + matplotlib | 3.10 | Siguen siendo el estándar para Figura 1 (speedup) y 2 (roofline) |
| Calidad | `clippy` + `rustfmt` | — | Linter y formateo; entran en el CI |
| GPU (opc) | `wgpu` (WGSL) o `candle` | — | Kernel propio (aprender) o GEMM real (comparar) |
| CI | GitHub Actions | — | `build` + `test` + `clippy` + `fmt` en cada push |

**Por qué Rust encaja con IA y con este curso:** te da el control de bajo nivel que necesitas para *enseñar* el hardware (intrínsecos, bloqueo manual), pero con un sistema de tipos que **demuestra** que tu código paralelo no tiene carreras de datos. Y el ecosistema de inferencia en Rust es real y creciente (Candle de HuggingFace, Burn, llama.cpp portado). Aprendes el hardware y un lenguaje empleable a la vez.

**Flujo de datos (sin cambios respecto al original):** la biblioteca `matmul` (las versiones) no depende de nada; el binario `benchmark` la consume, mide y escribe `results/results.csv`; ese CSV alimenta `plot.py` (figuras); en paralelo, los tests de integración la validan y son lo que corre el CI. Esta separación es la que permite construir y verificar por tareas independientes.

---

## 3. Estructura de archivos (layout de un crate de Rust)

```
matmul-demo/
├── Cargo.toml                 # paquete, dependencias, perfiles, benches
├── .cargo/
│   └── config.toml            # RUSTFLAGS: target-cpu=native (≈ -march=native)
├── README.md                  # cómo compilar y ejecutar
├── src/
│   ├── lib.rs                 # interfaz pública: reexporta las versiones
│   ├── base.rs                # matmul_base (escalar, ikj)
│   ├── simd.rs                # matmul_simd (AVX2 + FMA, con respaldo)
│   ├── blocked.rs             # matmul_blocked (tiling + SIMD)
│   ├── parallel.rs            # matmul_parallel (rayon)
│   ├── quant.rs               # (opc) matmul_quant_i8
│   └── bin/
│       └── benchmark.rs       # banco de pruebas: corre todo y exporta CSV
├── tests/
│   └── correctness.rs         # verifica que todas dan el mismo resultado (cargo test)
├── benches/
│   └── matmul.rs              # (opc) criterion: medición con rigor estadístico
├── scripts/
│   └── plot.py                # gráficos (speedup + roofline) desde el CSV
├── results/
│   └── results.csv            # salida de las mediciones (se genera)
└── .github/
    └── workflows/
        └── ci.yml             # build + test + clippy + fmt al hacer push
```

> Diferencias con la versión C: ya **no hace falta** un módulo `timer` (lo da `std::time::Instant`) ni `CMakeLists.txt` (lo da `cargo`). Los tests viven en `tests/` y los corre `cargo test`; los benchmarks en `benches/`. Menos *plumbing*, más concepto.

---

## 4. Interfaz común (`src/lib.rs`)

Todas las versiones comparten **la misma firma** para que el banco las trate como intercambiables. Matrices *row-major*, `f32`, de tamaño `n × n`.

```rust
//! C = A · B   (todas n×n, row-major, f32)
//! `c` debe venir inicializada a 0 por el llamador.

pub fn matmul_base(a: &[f32], b: &[f32], c: &mut [f32], n: usize);
pub fn matmul_simd(a: &[f32], b: &[f32], c: &mut [f32], n: usize);
pub fn matmul_blocked(a: &[f32], b: &[f32], c: &mut [f32], n: usize);
pub fn matmul_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize);
```

**Notas de diseño para el agente:**
- Los *slices* `&[f32]` ya llevan su longitud; aun así pasamos `n` por claridad y para validar con `assert_eq!(a.len(), n * n)` al entrar (un *panic* temprano es mejor que un cálculo silenciosamente corrupto).
- `matmul_simd` es una función **segura** por fuera: hace detección de CPU en tiempo de ejecución y, solo si hay AVX2+FMA, llama a un núcleo `unsafe` marcado con `#[target_feature(enable = "avx2,fma")]`; si no, cae a `matmul_base`. Este patrón —API segura envolviendo un núcleo inseguro verificado— es idiomático y es un concepto que vas a explicar.
- `matmul_blocked` recibe el tamaño de bloque como constante de compilación (`const BLOCK: usize = 64;`) ajustable para experimentar con la caché.

---

## 5. Preparación del entorno

Antes de la primera tarea, deja el entorno listo y **documenta tu máquina** (va a la Tabla 1 de la monografía).

```bash
# 1. Rust (toolchain estable) — en Linux/macOS/WSL2
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustc --version && cargo --version

# 2. Confirma que tu CPU soporta AVX2 y FMA (Linux/WSL2: deben aparecer)
lscpu | grep -i -E "avx2|fma"
#   macOS Apple Silicon: NO hay AVX2 (es ARM/NEON). Ver la nota de portabilidad abajo.

# 3. Documenta la plataforma para la Tabla 1 (guarda la salida)
lscpu > entorno_lscpu.txt          # en macOS: sysctl -a | grep machdep.cpu

# 4. Entorno Python para los gráficos
pip install pandas matplotlib --break-system-packages
```

`.cargo/config.toml` (equivalente a `-march=native`):

```toml
[build]
rustflags = ["-C", "target-cpu=native"]
```

> **Nota de portabilidad (importante hoy):** mucha IA corre en **ARM** (Apple Silicon, móviles, servidores Graviton), que **no tiene AVX2** sino **NEON**. Dos caminos:
> 1. Quédate en x86 (Linux/WSL2) y usa AVX2 explícito, como prevé el syllabus.
> 2. Usa **`std::simd`** (SIMD portable, hoy en *nightly*): el mismo código vectoriza a AVX2 en x86 y a NEON en ARM. Más moderno, menos "ves la instrucción concreta". Recomendación: aprende primero con AVX2 explícito (entiendes qué pasa), y menciona `std::simd` como la evolución portable.

---

## 6. Plan de construcción por tareas

Formato de cada tarea: **objetivo**, **depende de**, **entregable**, **criterio de aceptación** (cómo sabes que está bien), **qué aprendes** (el concepto) y **prompt** (qué pedirle al agente). Las firmas y rutas salen de §3 y §4. Tareas **T0–T9 = núcleo**; **T10–T13 = opcionales/avanzadas** (el horizonte moderno).

---

### T0 — Esqueleto del repositorio
- **Objetivo:** crear el crate y el repositorio Git inicial.
- **Depende de:** nada.
- **Entregable:** estructura de §3 creada, `Cargo.toml` y commit inicial.
- **Criterio de aceptación:** `cargo build` compila un esqueleto vacío; `git status` limpio tras el primer commit.
- **Qué aprendes:** anatomía de un crate de Rust (lib + bins + tests + benches) y cómo `cargo` reemplaza al *build system* manual.
- **Prompt:** *"Crea un crate de Rust llamado `matmul-demo` con la estructura de carpetas [pega §3]. Configura `Cargo.toml` (edición 2021, dependencias `rayon`, dev-dep `criterion`, un `[[bench]]` con `harness = false`) y `.cargo/config.toml` con `target-cpu=native`. Inicializa Git con un `.gitignore` para `/target` y `results/*.png`. Que `cargo build` pase."*

### T1 — Interfaz y validación
- **Objetivo:** definir la interfaz pública y el esqueleto de cada versión.
- **Depende de:** T0.
- **Entregable:** `src/lib.rs` con las firmas (§4), módulos vacíos que devuelven `unimplemented!()`.
- **Criterio de aceptación:** `cargo build` compila; cada función valida `a.len() == n*n` con `assert_eq!`.
- **Qué aprendes:** por qué medir tiempo en Rust **no necesita** un módulo aparte (`std::time::Instant::now()` + `.elapsed()` es un reloj monotónico de alta resolución), a diferencia de C.
- **Prompt:** *"Crea `src/lib.rs` con las cuatro firmas públicas [pega §4], cada una con `assert_eq!(a.len(), n*n)`. Declara los módulos `base`, `simd`, `blocked`, `parallel` con cuerpos `unimplemented!()`. Muéstrame además cómo cronometrar una llamada con `std::time::Instant` y explícame por qué es monotónico y mide tiempo de pared, no de CPU."*

### T2 — Versión base (escalar)
- **Objetivo:** implementar `matmul_base` en orden `ikj`.
- **Depende de:** T1.
- **Entregable:** `src/base.rs`.
- **Criterio de aceptación:** para casos conocidos (identidad · A = A) el resultado es correcto.
- **Qué aprendes:** **localidad espacial**. En `row-major`, recorrer una fila es recorrer memoria contigua (una *línea de caché* trae ~16 floats de una vez). El orden `ikj` hace que el bucle más interno avance por columnas contiguas de B y de C → aprovecha cada línea; el orden `ijk` salta por la columna de B (zancada `n`) → desperdicia caché.
- **Prompt:** *"Implementa `matmul_base` (row-major, f32, triple bucle en orden ikj) según la firma. Explícame, con el concepto de línea de caché, por qué ikj tiene mejor localidad espacial que ijk."*

### T3 — Prueba de correctitud
- **Objetivo:** test de equivalencia que de momento valida la base contra una referencia independiente.
- **Depende de:** T2.
- **Entregable:** `tests/correctness.rs`.
- **Criterio de aceptación:** `cargo test` pasa (verde).
- **Qué aprendes:** **no asociatividad del punto flotante**. Sumar en distinto orden da resultados *ligerísimamente* distintos en `f32`. Por eso comparas con **tolerancia** (`1e-3`), no con igualdad exacta. Esto es exactamente el problema de **reproducibilidad** en ML: la misma red puede dar números distintos según el orden de acumulación.
- **Prompt:** *"Escribe `tests/correctness.rs` que llene A y B de forma determinista (un generador simple con semilla fija), calcule C con `matmul_base`, y lo compare contra una referencia ingenua e independiente con tolerancia `1e-3` usando `assert!`. Explícame por qué uso tolerancia y no `==`, y qué tiene que ver con la reproducibilidad en deep learning."*

### T4 — Versión SIMD (AVX2 + FMA)
- **Objetivo:** implementar `matmul_simd` vectorizando el bucle interno.
- **Depende de:** T2, T3.
- **Entregable:** `src/simd.rs`; test extendido a base vs. SIMD.
- **Criterio de aceptación:** `cargo test` pasa comparando base vs. SIMD; maneja `n` no múltiplo de 8 (**cola escalar**).
- **Qué aprendes:** **DLP (paralelismo de datos)**: un registro de 256 bits guarda **8 floats**; `_mm256_fmadd_ps` hace `a*b+c` sobre los 8 a la vez, en una instrucción. Y en Rust: **por qué SIMD es `unsafe`** (el compilador no puede garantizar que tu CPU tenga la instrucción) y cómo se hace seguro de verdad con `#[target_feature]` + `is_x86_feature_detected!`.

  Esqueleto del patrón seguro:
  ```rust
  pub fn matmul_simd(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
      #[cfg(target_arch = "x86_64")]
      {
          if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
              // SAFETY: comprobamos AVX2+FMA en tiempo de ejecución justo arriba.
              unsafe { return matmul_avx2(a, b, c, n) }
          }
      }
      matmul_base(a, b, c, n) // respaldo portable
  }

  #[cfg(target_arch = "x86_64")]
  #[target_feature(enable = "avx2,fma")]
  unsafe fn matmul_avx2(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
      use std::arch::x86_64::*;
      for i in 0..n {
          for k in 0..n {
              let a_ik = _mm256_set1_ps(a[i*n + k]);   // difunde A[i,k] a las 8 lanes
              let mut j = 0;
              while j + 8 <= n {
                  let bv = _mm256_loadu_ps(b.as_ptr().add(k*n + j));
                  let cv = _mm256_loadu_ps(c.as_ptr().add(i*n + j));
                  let r  = _mm256_fmadd_ps(a_ik, bv, cv); // 8 × (a*b+c) de un golpe
                  _mm256_storeu_ps(c.as_mut_ptr().add(i*n + j), r);
                  j += 8;
              }
              while j < n { c[i*n + j] += a[i*n + k] * b[k*n + j]; j += 1; } // cola
          }
      }
  }
  ```
- **Prompt:** *"Implementa `matmul_simd` con el patrón de detección en tiempo de ejecución + núcleo `#[target_feature(enable=\"avx2,fma\")]`, usando `_mm256_set1_ps`, `_mm256_loadu_ps`, `_mm256_fmadd_ps`, `_mm256_storeu_ps` y cola escalar para el remanente. Extiende el test a base vs. SIMD. Explícame qué hace `_mm256_fmadd_ps` y por qué el bloque tiene que ir en `unsafe` aunque sea seguro."*

### T5 — Versión con bloqueo de caché
- **Objetivo:** `matmul_blocked` (*tiling* + SIMD), con `BLOCK` configurable.
- **Depende de:** T4.
- **Entregable:** `src/blocked.rs`; test extendido a las tres versiones.
- **Criterio de aceptación:** `cargo test` pasa para las tres; `const BLOCK` ajustable; maneja bordes no múltiplos de `BLOCK`.
- **Qué aprendes:** **jerarquía de memoria y reúso**. La matmul ingenua, para `n` grande, vuelve a leer toda B desde RAM una y otra vez: el *working set* no cabe en L1/L2 y el ancho de banda se vuelve el cuello de botella. El bloqueo procesa sub-bloques `BLOCK×BLOCK` que **sí caben en caché** y los reutiliza al máximo antes de avanzar → menos viajes a RAM. Este es el concepto que más se transfiere a la GPU.
- **Prompt:** *"Implementa `matmul_blocked` con bloqueo `BLOCK×BLOCK` (`const BLOCK: usize = 64`) y el bucle interno vectorizado (reusa el núcleo AVX2), cuidando los bordes. Extiende el test a las tres versiones. Explícame, con la idea de *working set* y ancho de banda, por qué el bloqueo reduce los fallos de caché."*

### T6 — Versión multihilo (rayon)
- **Objetivo:** `matmul_parallel`, repartiendo filas/bloques entre núcleos.
- **Depende de:** T5.
- **Entregable:** `src/parallel.rs`; test extendido a las cuatro versiones.
- **Criterio de aceptación:** `cargo test` pasa para las cuatro; usa todos los núcleos disponibles.
- **Qué aprendes:** **TLP (paralelismo de hilos)** y la **garantía de Rust**: cada hilo escribe una **fila distinta** de C (slices mutables disjuntos), así que es *imposible* que haya una carrera de datos —y el compilador lo **demuestra**, no lo confías—. En C/OpenMP tendrías que razonarlo a mano. También verás **false sharing** si dos hilos tocan la misma línea de caché.
  ```rust
  use rayon::prelude::*;
  pub fn matmul_parallel(a: &[f32], b: &[f32], c: &mut [f32], n: usize) {
      c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
          for k in 0..n {
              let a_ik = a[i*n + k];
              let b_row = &b[k*n..k*n + n];
              for j in 0..n { c_row[j] += a_ik * b_row[j]; }   // (vectoriza este bucle)
          }
      });
  }
  ```
- **Prompt:** *"Implementa `matmul_parallel` con `rayon`, paralelizando por filas de C con `par_chunks_mut(n)`, y vectoriza el bucle interno. Explícame por qué el sistema de tipos de Rust garantiza que no hay *data races* aquí, y qué es el *false sharing*."*

### T7 — Banco de pruebas y CSV
- **Objetivo:** medir todas las versiones y exportar resultados.
- **Depende de:** T6.
- **Entregable:** `src/bin/benchmark.rs` que escribe `results/results.csv`.
- **Criterio de aceptación:** genera el CSV con cabecera `n,version,time_ms,gflops,speedup`; aplica **warm-up** y promedia repeticiones; semilla fija.
- **Qué aprendes:** cómo se mide bien (descartar la corrida fría, promediar, ruido del sistema) y la métrica **GFLOP/s** = `2·n³ / t / 1e9` (cada elemento de C son `n` multiplicaciones + `n` sumas).
- **Prompt:** *"Escribe `src/bin/benchmark.rs` que, para n en {256,512,1024,2048} con warm-up y 10 repeticiones (`std::time::Instant`), mida las cuatro versiones, calcule GFLOP/s y el speedup respecto a base, y vuelque todo a `results/results.csv` con la cabecera exacta `n,version,time_ms,gflops,speedup`."*

### T8 — Benchmark con rigor estadístico (criterion)
- **Objetivo:** medición robusta (warm-up automático, detección de outliers, intervalos).
- **Depende de:** T6.
- **Entregable:** `benches/matmul.rs`.
- **Criterio de aceptación:** `cargo bench` corre y reporta tiempos con su varianza.
- **Qué aprendes:** por qué un cronómetro casero engaña (ruido del SO, escalado de frecuencia) y cómo una herramienta estadística lo corrige. El CSV de T7 alimenta el roofline; criterion te da el número "de verdad" para el texto.
- **Prompt:** *"Crea `benches/matmul.rs` con `criterion`: un grupo que compare las cuatro versiones para n=512 y n=1024. Explícame qué problemas de medición resuelve criterion frente a cronometrar a mano."*

### T9 — Gráficos: speedup y roofline
- **Objetivo:** generar las figuras desde el CSV.
- **Depende de:** T7.
- **Entregable:** `scripts/plot.py`.
- **Criterio de aceptación:** produce `results/speedup.png` y `results/roofline.png`; el roofline usa el **pico de cómputo y el ancho de banda de TU CPU** (documenta de dónde salen).
- **Qué aprendes (el concepto estrella): el modelo *roofline***. Eje X = **intensidad operacional** (FLOP por byte leído de RAM); eje Y = GFLOP/s, en log-log. Hay dos techos: una recta inclinada (ancho de banda de memoria) y una horizontal (pico de cómputo). Un kernel a la izquierda está **limitado por memoria**; a la derecha, **por cómputo**. El bloqueo sube la intensidad operacional (más reúso por byte) → mueve tu punto a la derecha, hacia el techo de cómputo.
  - **Pico de cómputo (f32, AVX2+FMA):** `núcleos × frecuencia × FLOPs_por_ciclo`. Con 2 puertos FMA: `2 (FMA) × 8 (lanes) × 2 (mul+suma) = 32 FLOPs/ciclo/núcleo` (varía por microarquitectura, documenta el tuyo).
  - **Ancho de banda:** de las specs de tu RAM (p. ej. DDR4-3200 dual-channel ≈ 51 GB/s) o medido.
  - **Conexión IA (potente y real):** en un LLM, el *prefill* (procesar el prompt) son GEMMs grandes → **limitado por cómputo**, a la derecha del roofline. El *decode* autoregresivo (generar token a token) es matriz×vector → **limitado por ancho de banda**, a la izquierda. Por eso el decode es lento y por eso se hace *batching*: junta varias secuencias para subir la intensidad operacional y empujar el punto hacia el techo de cómputo. Ese es **el mismo movimiento** que hace tu versión bloqueada.
- **Prompt:** *"Escribe `scripts/plot.py` con pandas+matplotlib: (1) speedup vs n desde `results.csv`; (2) roofline log-log con techo de cómputo y de memoria más los puntos de cada versión. Deja `pico_gflops` y `ancho_banda_gbps` como variables comentadas para que ponga las de mi máquina, y comenta cómo estimé el pico."*

---

### T10 — Integración continua *(opcional, lo pide el syllabus)*
- **Objetivo:** que cada push compile, testee y revise estilo.
- **Depende de:** T7.
- **Entregable:** `.github/workflows/ci.yml`.
- **Criterio de aceptación:** badge en verde; el job corre `cargo build --release`, `cargo test`, `cargo clippy -- -D warnings` y `cargo fmt --check`.
- **Qué aprendes:** higiene moderna de un proyecto Rust (linter + formateo como puerta de calidad). Los runners de GitHub soportan AVX2, así que tu SIMD se ejecuta de verdad en CI.
- **Prompt:** *"Crea `.github/workflows/ci.yml` para ubuntu-latest que instale Rust estable y corra `cargo build --release`, `cargo test`, `cargo clippy -- -D warnings` y `cargo fmt --all -- --check`."*

### T11 — Comparación contra un BLAS real *(opcional, da humildad y contexto)*
- **Objetivo:** medir cuánto te falta para una librería tuneada profesionalmente.
- **Depende de:** T7.
- **Entregable:** una columna/serie extra en el benchmark usando el crate `matrixmultiply` (o `candle` en CPU).
- **Criterio de aceptación:** el CSV incluye la versión "biblioteca" y se ve la brecha contra tu mejor versión.
- **Qué aprendes:** la diferencia entre tu mejor esfuerzo y lo que logran cuBLAS/oneDNN/BLIS con décadas de afinado (empaquetado de bloques, prefetch, micro-kernels en ensamblador). Te ubica: entender el porqué vale más que ganarle.
- **Prompt:** *"Añade al benchmark una versión que multiplique con el crate `matrixmultiply`. Compárala con mi `matmul_parallel` y explícame qué hace una BLAS de producción que yo no (packing, micro-kernels, prefetch)."*

### T12 — Cuantización a int8 *(opcional, muy actual)*
- **Objetivo:** una matmul en enteros de 8 bits.
- **Depende de:** T4.
- **Entregable:** `src/quant.rs` con `matmul_quant_i8` (acumula en `i32`).
- **Criterio de aceptación:** el resultado, *descuantizado*, se aproxima al de `f32` dentro de una tolerancia mayor (la cuantización introduce error controlado).
- **Qué aprendes:** **precisión vs. throughput**, el motor de la IA eficiente de hoy. Con 8 bits, un registro de 256 bits guarda **32 enteros** (vs. 8 floats) → 4× más elementos por instrucción y **4× menos tráfico de memoria** → el punto sube en el roofline. Es literalmente cómo los LLMs cuantizados (int8/int4) corren más rápido y con menos RAM. (En CPUs con VNNI, `_mm256_dpbusd_epi32` hace el producto-punto int8→int32 en una instrucción.)
- **Prompt:** *"Implementa `matmul_quant_i8`: cuantiza A y B de f32 a i8 con escala simétrica, multiplica acumulando en i32, y descuantiza. Compárala con la versión f32 (tolerancia mayor) y explícame por qué menos bits dan más throughput y cómo se relaciona con la cuantización de LLMs."*

### T13 — Versión GPU *(opcional, el horizonte)*
- **Objetivo:** llevar la misma escalera a la GPU.
- **Depende de:** T5 (la idea de *tiling*).
- **Entregable:** `src/gpu/` con un *compute shader* WGSL vía `wgpu`: primero *naive*, luego con **memoria compartida** (*tiled*).
- **Criterio de aceptación:** el resultado coincide con la CPU (tolerancia f32) y se ve el salto del naive al *tiled*.
- **Qué aprendes:** la **jerarquía de memoria de la GPU** (registros → memoria compartida/L1 del *workgroup* → memoria global) y que el *tiling con memoria compartida* es **el mismo bloqueo de caché de T5**, pero explícito: cada *workgroup* trae un sub-bloque a la memoria compartida y lo reutiliza. Aquí es donde "corre la IA de verdad".
- **Prompt:** *"Crea un kernel de matmul en WGSL ejecutado con `wgpu`: versión 1 naive (cada hilo calcula un C[i,j]); versión 2 con `var<workgroup>` cargando tiles a memoria compartida. Valida contra la CPU. Explícame el paralelo entre la memoria compartida del workgroup y el bloqueo de caché de la versión CPU."*

### T14 — Perfilado de caché *(opcional, refuerza la Discusión)*
- **Objetivo:** medir fallos de caché por versión y datos para la sección de discusión.
- **Depende de:** T7.
- **Entregable:** tabla de fallos (texto/CSV) y, si quieres, un *flamegraph*.
- **Criterio de aceptación:** conteos comparables entre base y bloqueo para `n` grande.
- **Prompt:** *"Dame los comandos para perfilar el binario `benchmark` en release: `perf stat -e cache-misses,LLC-load-misses`, `valgrind --tool=cachegrind`, y `cargo flamegraph`. Explícame cómo leer que el bloqueo reduce los fallos de último nivel."*

---

## 7. Definición de "hecho" (checklist de cierre)

- [ ] Las cuatro versiones núcleo compilan en *release* y pasan `cargo test`.
- [ ] `benchmark` genera `results.csv` con datos reales de tu máquina.
- [ ] `plot.py` produce speedup y roofline con los parámetros reales de tu CPU.
- [ ] El CI está en verde (build + test + clippy + fmt).
- [ ] La plataforma está documentada (`entorno_lscpu.txt`) y volcada a la Tabla 1.
- [ ] Sabes **explicar** cada versión: localidad (base), DLP (SIMD), jerarquía de memoria (bloqueo), TLP (paralelo), roofline (gráfico).
- [ ] (Opcional) int8, GPU, comparación BLAS y/o conteo de fallos de caché para la Discusión.

---

## 8. Riesgos comunes y solución (versión Rust)

| Síntoma | Causa probable | Solución |
|---|---|---|
| `SIGILL` / instrucción ilegal al ejecutar | CPU sin AVX2, o llamaste al núcleo `unsafe` sin detección | Usa el patrón `is_x86_feature_detected!` con respaldo a `base` |
| El *speedup* de SIMD parece pequeño | LLVM ya **auto-vectorizó** la base en *release* | Es esperable y honesto; mídelo, y si quieres una base "limpia" desactiva la vectorización solo en `base.rs` (atributo de optimización) y documéntalo |
| SIMD difiere de base más de la tolerancia | Mal manejo de la **cola** (n no múltiplo de 8) | Revisa el bucle escalar final; recuerda que una diferencia *pequeña* es normal (punto flotante) |
| El bloqueo no acelera | `BLOCK` mal dimensionado para tu caché | Prueba 32, 64, 128 y compara fallos de caché |
| El paralelo no escala / va peor | *Overhead* de hilos en `n` pequeño, o *false sharing* | Paraleliza por bloques de filas, no fila a fila, para `n` chico |
| Tiempos ruidosos | Sin warm-up, o escalado de frecuencia (turbo) | Descarta la 1ª corrida; usa `criterion`; fija frecuencia si puedes |
| `target-cpu=native` rompe en CI o en otra máquina | El binario usa instrucciones que el otro CPU no tiene | En CI usa un target genérico + detección en runtime (que ya tienes) |

---

## 9. Trazabilidad: tarea → sección de la monografía

| Tarea(s) | Alimenta la sección |
|---|---|
| Entorno (§5) | 3.1 Tabla 1 |
| T0–T1, T10 | 3.4 Reproducibilidad / Anexos (CI) |
| T2, T4, T5, T6 | 4. Implementación / Anexos |
| T3 | 4.4 Verificación (y nota sobre reproducibilidad en punto flotante) |
| T7, T8 | 5. Resultados (Tabla 2) / Anexo C |
| T9 | 5. Resultados (Figuras 1 y 2: speedup y roofline) |
| T11, T12, T14 | 6. Discusión (límite del afinado, cuantización, fallos de caché) |
| T13 | 6/7. Discusión y trabajo futuro (GPU) |

---

## 10. Hacia dónde sigue (el horizonte de la IA moderna)

Cuando termines el núcleo, ya entenderás los cuatro ejes con los que se razona *todo* el rendimiento de IA: **datos (SIMD), memoria (bloqueo), hilos (paralelo) y el roofline** que los une. Los pasos naturales, en orden de impacto:

1. **Precisión reducida (T12):** bf16/fp16/int8/int4. Es la palanca nº1 de la inferencia de LLMs hoy.
2. **GPU (T13):** la misma escalera, con miles de hebras y memoria compartida explícita. De aquí a entender cómo funciona un kernel de cuBLAS o de **FlashAttention** (que es, en el fondo, *tiling* de la atención para no salir de la memoria rápida).
3. **Frameworks reales en Rust:** mira cómo **Candle** o **Burn** orquestan estas operaciones; tu matmul a mano te deja *leer su código* con criterio en vez de como magia.

> En cada paso, pídele al agente que te explique el porqué. El objetivo no es ganarle a cuBLAS —no vas a—, sino **entender por qué cuBLAS hace lo que hace**. Eso es lo que convierte esta demo en arquitectura de computadoras aplicada a la IA, y no en código que solo funciona.
