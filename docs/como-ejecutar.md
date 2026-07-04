# Cómo ejecutar el proyecto

## Requisitos

- [Rust y Cargo](https://www.rust-lang.org/tools/install) instalados.

## Compilar

Desde la raíz del proyecto:

```bash
cargo build
```

## Ejecutar

```bash
cargo run
```

Al iniciar, el programa muestra un menú:

```
==========================
 AI Matrix Engine
==========================

1. Base
2. SIMD
3. Cache Blocking
4. Parallel
5. Modo Web (benchmark real)

Seleccione:
```

Escribe el número de la estrategia que quieras simular (`1`, `2`, `3` o `4`) y presiona Enter, o elige `5` para el modo web (ver más abajo).

## Qué muestra cada ejecución

Tras elegir una opción, el programa ejecuta en orden:

1. **Matrices de ejemplo (4x4)** — imprime la Matriz A y la Matriz B.
2. **Multiplicación paso a paso** — muestra cada lectura, multiplicación y acumulado, y el resultado final.
3. **Simulación visual de la estrategia elegida** — el diagrama conceptual de cómo trabaja Base, SIMD, Cache Blocking o Parallel.
4. **Benchmark simulado** — tiempos, GFLOPS y speedup de las 4 estrategias.
5. **Gráficos ASCII** — barras comparando tiempo y rendimiento.
6. **Conclusión** — resumen de las técnicas y su relación con la aceleración de IA.

## Demo visual (navegador) con benchmark real

Hay una versión gráfica e interactiva del proyecto en [`docs/visual-demo/`](visual-demo/). A diferencia de una página suelta, ahora se sirve desde el propio programa Rust y el benchmark que muestra **corre de verdad**, no son números fijos.

Para usarla:

```bash
cargo run --release
```

y elige la opción `5. Modo Web (benchmark real)`. La terminal mostrará:

```
Servidor web activo en http://127.0.0.1:7878
```

Abre esa dirección en el navegador (no abras `index.html` directamente con doble clic — necesita el servidor para responder al benchmark). Ahí encontrarás:

- La explicación conceptual de GEMM y una tabla comparativa de las 4 estrategias.
- La definición de cada estrategia, con diagrama, "qué simula esta demo" y "qué es en hardware real".
- La multiplicación 4×4 paso a paso (esta parte sigue siendo una simulación pedagógica: anima cómo se leen y acumulan los datos).
- Un **benchmark real**: eliges el tamaño de matriz (128 a 1024) y el servidor ejecuta ahí mismo, en Rust (`src/engine.rs`), las 4 implementaciones —Base (bucle ingenuo), SIMD (reordenado para auto-vectorización), Cache Blocking (bloqueado por tiles) y Parallel (repartido en hilos reales con `std::thread::scope`)— midiendo el tiempo real con `Instant::now()` y verificando que las 4 dan el mismo resultado (checksum).
- Usa `--release` para ver diferencias de velocidad marcadas: en modo debug el compilador no vectoriza ni optimiza, así que SIMD/Cache Blocking no muestran su ventaja real.

Detén el servidor con `Ctrl+C` en la terminal donde corre `cargo run`.

## Pruebas

Para correr las pruebas unitarias (por ejemplo, las de `matrix.rs`):

```bash
cargo test
```
