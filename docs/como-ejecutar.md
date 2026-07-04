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

Seleccione:
```

Escribe el número de la estrategia que quieras simular (`1`, `2`, `3` o `4`) y presiona Enter.

## Qué muestra cada ejecución

Tras elegir una opción, el programa ejecuta en orden:

1. **Matrices de ejemplo (4x4)** — imprime la Matriz A y la Matriz B.
2. **Multiplicación paso a paso** — muestra cada lectura, multiplicación y acumulado, y el resultado final.
3. **Simulación visual de la estrategia elegida** — el diagrama conceptual de cómo trabaja Base, SIMD, Cache Blocking o Parallel.
4. **Benchmark simulado** — tiempos, GFLOPS y speedup de las 4 estrategias.
5. **Gráficos ASCII** — barras comparando tiempo y rendimiento.
6. **Conclusión** — resumen de las técnicas y su relación con la aceleración de IA.

## Demo visual (navegador)

Además de la terminal, hay una versión gráfica e interactiva en [`docs/visual-demo/index.html`](visual-demo/index.html). Ábrela directamente con doble clic (o "Abrir con navegador") para:

- Elegir una estrategia y leer su definición con un diagrama.
- Avanzar paso a paso (o reproducir automáticamente) la multiplicación 4x4, viendo qué celdas de A y B se leen y cómo se acumula el resultado.
- Ver el benchmark simulado y los gráficos de barras de tiempo/rendimiento.
- Leer la conclusión final.

No requiere servidor ni dependencias, es HTML/CSS/JS autocontenido.

## Pruebas

Para correr las pruebas unitarias (por ejemplo, las de `matrix.rs`):

```bash
cargo test
```
