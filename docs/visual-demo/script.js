const N = 4;
const A = Array.from({ length: 16 }, (_, i) => i + 1);
const B = Array.from({ length: 16 }, (_, i) => 16 - i);

// Tiempos simulados (ms) tomados del mismo benchmark que muestra la app de terminal.
const STRATEGY_TOTAL_MS = { base: 125, simd: 55, cache: 31, parallel: 15 };

function idx(r, c) {
  return r * N + c;
}

// --- Generadores de pasos: cada estrategia agrupa el mismo cálculo de forma distinta ---

function buildBase() {
  const frames = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let acc = 0;
      for (let k = 0; k < N; k++) {
        const a = A[idx(i, k)];
        const b = B[idx(k, j)];
        const product = a * b;
        acc += product;
        const isLast = k === N - 1;
        frames.push({
          activeA: [[i, k]],
          activeB: [[k, j]],
          writes: isLast ? [{ r: i, c: j, value: acc }] : [],
          weight: 1,
          text:
            `Leyendo A[${i}][${k}]\n` +
            `Leyendo B[${k}][${j}]\n` +
            `<span class="op">${a} × ${b} = ${product}</span>\n` +
            `Acumulado = ${acc}` +
            (isLast ? `\n→ Resultado[${i}][${j}] = ${acc}` : ""),
        });
      }
    }
  }
  return frames;
}

function buildSimd() {
  const frames = [];
  for (let i = 0; i < N; i++) {
    for (let j = 0; j < N; j++) {
      let acc = 0;
      const terms = [];
      for (let k = 0; k < N; k++) {
        const a = A[idx(i, k)];
        const b = B[idx(k, j)];
        const product = a * b;
        acc += product;
        terms.push(`${a}×${b}=${product}`);
      }
      frames.push({
        activeA: [[i, 0], [i, 1], [i, 2], [i, 3]],
        activeB: [[0, j], [1, j], [2, j], [3, j]],
        writes: [{ r: i, c: j, value: acc }],
        weight: 1,
        text:
          `Cargando fila A[${i}][0..3] y columna B[0..3][${j}] en un registro vectorial\n` +
          `<span class="op">${terms.join("  |  ")}</span>\n` +
          `Suma horizontal = ${acc}\n→ Resultado[${i}][${j}] = ${acc}`,
      });
    }
  }
  return frames;
}

function buildCache() {
  const blockSize = 2;
  const acc = Array.from({ length: N }, () => Array(N).fill(0));
  const frames = [];
  let ramReads = 0;
  let cacheHits = 0;

  for (let bi = 0; bi < N; bi += blockSize) {
    for (let bj = 0; bj < N; bj += blockSize) {
      for (let bk = 0; bk < N; bk += blockSize) {
        let firstAccessInBlock = true;
        for (let i = bi; i < bi + blockSize; i++) {
          for (let j = bj; j < bj + blockSize; j++) {
            for (let k = bk; k < bk + blockSize; k++) {
              const a = A[idx(i, k)];
              const b = B[idx(k, j)];
              const product = a * b;
              acc[i][j] += product;

              const isLoad = firstAccessInBlock;
              firstAccessInBlock = false;
              if (isLoad) ramReads++;
              else cacheHits++;

              const isLastForCell = bk + blockSize >= N && k === bk + blockSize - 1;

              frames.push({
                activeA: [[i, k]],
                activeB: [[k, j]],
                writes: isLastForCell ? [{ r: i, c: j, value: acc[i][j] }] : [],
                weight: isLoad ? 3 : 1,
                isBlockLoad: isLoad,
                ramReadsSoFar: ramReads,
                cacheHitsSoFar: cacheHits,
                text:
                  (isLoad
                    ? `⏳ Cargando bloque A[${bi}..${bi + 1}][${bk}..${bk + 1}] y B[${bk}..${bk + 1}][${bj}..${bj + 1}] desde RAM\n`
                    : `⚡ Reutilizando bloque ya cargado en caché\n`) +
                  `Leyendo A[${i}][${k}]\nLeyendo B[${k}][${j}]\n` +
                  `<span class="op">${a} × ${b} = ${product}</span>\n` +
                  `Acumulado parcial = ${acc[i][j]}` +
                  (isLastForCell ? `\n→ Resultado[${i}][${j}] = ${acc[i][j]}` : ""),
              });
            }
          }
        }
      }
    }
  }
  return frames;
}

function buildParallel() {
  const frames = [];
  for (let j = 0; j < N; j++) {
    const acc = [0, 0, 0, 0];
    for (let k = 0; k < N; k++) {
      const lines = [];
      for (let i = 0; i < N; i++) {
        const a = A[idx(i, k)];
        const b = B[idx(k, j)];
        acc[i] += a * b;
        lines.push(`CPU${i + 1}: A[${i}][${k}] × B[${k}][${j}] = ${a}×${b} → acumulado=${acc[i]}`);
      }
      const isLast = k === N - 1;
      frames.push({
        activeA: [[0, k], [1, k], [2, k], [3, k]],
        activeB: [[k, j]],
        coreActive: [0, 1, 2, 3],
        writes: isLast ? [0, 1, 2, 3].map((i) => ({ r: i, c: j, value: acc[i] })) : [],
        weight: 1,
        text: lines.join("\n") + (isLast ? `\n→ Columna ${j} completa (4 núcleos a la vez)` : ""),
      });
    }
  }
  return frames;
}

const BUILDERS = { base: buildBase, simd: buildSimd, cache: buildCache, parallel: buildParallel };

// --- Estado y render ---

let strategyKey = "base";
let framesData = [];
let current = -1;
let playing = false;
let playTimeout = null;

const gridA = document.getElementById("grid-a");
const gridB = document.getElementById("grid-b");
const gridR = document.getElementById("grid-r");
const stepExplanation = document.getElementById("step-explanation");
const stepCounter = document.getElementById("step-counter");
const strategyMeta = document.getElementById("strategy-meta");
const timeBarFill = document.getElementById("time-bar-fill");
const timeBarLabel = document.getElementById("time-bar-label");
const corePanel = document.getElementById("core-panel");
const btnNext = document.getElementById("btn-next");
const btnPrev = document.getElementById("btn-prev");
const btnReset = document.getElementById("btn-reset");
const btnPlay = document.getElementById("btn-play");

function renderStaticGrid(container, values, prefix) {
  container.innerHTML = "";
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const div = document.createElement("div");
      div.className = "cell";
      div.textContent = values[idx(r, c)];
      div.id = `${prefix}-${r}-${c}`;
      container.appendChild(div);
    }
  }
}

function renderEmptyResultGrid() {
  gridR.innerHTML = "";
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      const div = document.createElement("div");
      div.className = "cell empty";
      div.textContent = "·";
      div.id = `r-${r}-${c}`;
      gridR.appendChild(div);
    }
  }
}

function clearHighlights() {
  document.querySelectorAll(".cell.active-a, .cell.active-b").forEach((el) => {
    el.classList.remove("active-a", "active-b");
  });
}

function renderResultFromFrames(uptoIndex) {
  renderEmptyResultGrid();
  for (let s = 0; s <= uptoIndex; s++) {
    const f = framesData[s];
    f.writes.forEach((w) => {
      const cell = document.getElementById(`r-${w.r}-${w.c}`);
      cell.textContent = w.value;
      cell.classList.remove("empty");
      cell.classList.add("filled");
    });
  }
  if (uptoIndex >= 0) {
    framesData[uptoIndex].writes.forEach((w) => {
      document.getElementById(`r-${w.r}-${w.c}`).classList.add("just-set");
    });
  }
}

function computeTotalWeight() {
  return framesData.reduce((sum, f) => sum + (f.weight || 1), 0);
}

function elapsedMsUpTo(index) {
  if (index < 0) return 0;
  const totalWeight = computeTotalWeight();
  const totalMs = STRATEGY_TOTAL_MS[strategyKey];
  let total = 0;
  for (let s = 0; s <= index; s++) {
    total += ((framesData[s].weight || 1) / totalWeight) * totalMs;
  }
  return total;
}

function updateCoreHighlights(frame) {
  if (strategyKey !== "parallel") return;
  [0, 1, 2, 3].forEach((i) => {
    const box = document.getElementById(`core-box-${i}`);
    if (!box) return;
    box.classList.toggle("working", !!(frame && frame.coreActive && frame.coreActive.includes(i)));
  });
}

function renderCorePanel() {
  if (strategyKey !== "parallel") {
    corePanel.style.display = "none";
    corePanel.innerHTML = "";
    return;
  }
  corePanel.style.display = "flex";
  const boxes = [0, 1, 2, 3]
    .map((i) => `<div class="core-box" id="core-box-${i}">CPU ${i + 1}<br>Fila ${i}</div>`)
    .join("");
  corePanel.innerHTML = `<p class="core-panel-label">Estado de los núcleos:</p><div class="core-boxes">${boxes}</div>`;
}

function metaTextFor(index) {
  if (index < 0) return "";
  const f = framesData[index];
  if (strategyKey === "simd") return "Vector: 4 elementos a la vez (simulando 8 en hardware real)";
  if (strategyKey === "cache") return `RAM reads: ${f.ramReadsSoFar} · Cache hits: ${f.cacheHitsSoFar}`;
  if (strategyKey === "parallel") return "4 núcleos trabajando en paralelo";
  return "";
}

function applyStep(index) {
  current = index;
  clearHighlights();
  renderResultFromFrames(index);

  const total = framesData.length;
  stepCounter.textContent = `Paso ${Math.max(index + 1, 0)} / ${total}`;
  strategyMeta.textContent = metaTextFor(index);

  const totalMs = STRATEGY_TOTAL_MS[strategyKey];
  const elapsed = elapsedMsUpTo(index);
  const pct = Math.min(100, (elapsed / totalMs) * 100);
  timeBarFill.style.width = `${pct}%`;
  timeBarLabel.textContent = `${elapsed.toFixed(2)} ms / ${totalMs} ms simulados`;

  if (index < 0) {
    stepExplanation.textContent = "Presiona Siguiente paso para comenzar.";
    updateCoreHighlights(null);
  } else {
    const f = framesData[index];
    stepExplanation.innerHTML = f.text;
    f.activeA.forEach(([r, c]) => document.getElementById(`a-${r}-${c}`)?.classList.add("active-a"));
    f.activeB.forEach(([r, c]) => document.getElementById(`b-${r}-${c}`)?.classList.add("active-b"));
    updateCoreHighlights(f);
  }

  btnPrev.disabled = index <= -1;
  btnNext.disabled = index >= total - 1;
}

function stopPlay() {
  playing = false;
  if (playTimeout) {
    clearTimeout(playTimeout);
    playTimeout = null;
  }
  btnPlay.textContent = "▶ Reproducir automático";
}

function nextDelayMs(nextIndex) {
  const f = framesData[nextIndex];
  if (strategyKey === "cache") return f.isBlockLoad ? 260 : 70;
  return { base: 220, simd: 170, parallel: 170 }[strategyKey] || 200;
}

function scheduleNextPlayTick() {
  const nextIndex = current + 1;
  if (nextIndex >= framesData.length) {
    stopPlay();
    return;
  }
  playTimeout = setTimeout(() => {
    applyStep(nextIndex);
    if (playing) scheduleNextPlayTick();
  }, nextDelayMs(nextIndex));
}

btnNext.addEventListener("click", () => {
  if (current < framesData.length - 1) applyStep(current + 1);
});

btnPrev.addEventListener("click", () => {
  applyStep(Math.max(current - 1, -1));
});

btnReset.addEventListener("click", () => {
  stopPlay();
  applyStep(-1);
});

btnPlay.addEventListener("click", () => {
  if (playing) {
    stopPlay();
    return;
  }
  if (current >= framesData.length - 1) applyStep(-1);
  playing = true;
  btnPlay.textContent = "⏸ Pausar";
  scheduleNextPlayTick();
});

// --- Definiciones por estrategia ---

const definitions = {
  base: {
    title: "Base",
    text: "Procesa un dato a la vez, de forma estrictamente secuencial: lee, multiplica, guarda y repite.",
    demo:
      "Cada paso lee un elemento de A y uno de B, los multiplica y acumula. Son 64 pasos en total: uno por cada multiplicación individual de la matriz 4×4.",
    real:
      "Así trabaja una CPU sin ninguna optimización: una instrucción escalar a la vez, sin aprovechar registros vectoriales, caché adicional ni otros núcleos.",
    diagram: () => `
      <div class="box">CPU</div><div class="arrow">→</div>
      <div class="box">Lee dato</div><div class="arrow">→</div>
      <div class="box">Multiplica</div><div class="arrow">→</div>
      <div class="box">Guarda</div><div class="arrow">→</div>
      <div class="box">Repite</div>
    `,
  },
  simd: {
    title: "SIMD",
    text: 'Un registro vectorial (por ejemplo AVX2) carga varios datos a la vez en "lanes" y los procesa todos en una sola instrucción.',
    demo:
      "Cada paso procesa una fila completa de A y una columna completa de B a la vez (4 multiplicaciones simultáneas), como si fueran los lanes de un registro vectorial. Solo 16 pasos: 4 veces menos que Base.",
    real:
      "Instrucciones como AVX2 cargan hasta 8 números en un solo registro y los multiplican en un único ciclo, en vez de uno por uno.",
    diagram: () => `
      <div class="lane">Lane0</div>
      <div class="lane">Lane1</div>
      <div class="lane">Lane2</div>
      <div class="lane">...</div>
      <div class="arrow">→</div>
      <div class="box">8 datos cargados</div><div class="arrow">→</div>
      <div class="box">8 multiplicaciones</div><div class="arrow">→</div>
      <div class="box">8 resultados</div>
      <div class="note">Procesando 8 elementos simultáneamente...</div>
    `,
  },
  cache: {
    title: "Cache Blocking",
    text: "Un bloque de datos se mantiene en caché mientras se reutiliza varias veces, evitando volver a leerlo desde RAM.",
    demo:
      "Los datos se procesan en bloques de 2×2. La primera vez que se necesita un bloque se \"carga desde RAM\" (lento); las siguientes veces que se reutiliza dentro del mismo bloque son \"aciertos de caché\" (rápidos). Se hacen las mismas 64 multiplicaciones que en Base, pero con muchas menos lecturas a RAM.",
    real:
      "La memoria caché es mucho más rápida que la RAM. Reorganizar los bucles para reutilizar los datos que ya están en caché reduce drásticamente los accesos lentos a memoria principal.",
    diagram: () => `
      <div class="box">RAM<br>□□□□□□□□</div><div class="arrow">→</div>
      <div class="box">CACHE<br>■■■■</div><div class="arrow">→</div>
      <div class="box">CPU</div><div class="arrow">→</div>
      <div class="box">CACHE reutilizada</div><div class="arrow">→</div>
      <div class="box">RAM</div>
      <div class="note">El bloque permanece en caché. No vuelve a leerse desde RAM.</div>
    `,
  },
  parallel: {
    title: "Parallel",
    text: "Varios núcleos de CPU trabajan sobre distintos rangos de filas al mismo tiempo.",
    demo:
      "Cada una de las 4 filas se asigna a un núcleo distinto. En cada paso los 4 núcleos avanzan a la vez sobre su fila. Solo 16 pasos, porque las 4 filas se calculan simultáneamente en vez de una tras otra.",
    real:
      "Con varios núcleos (o hilos), se reparten distintas filas de la matriz entre ellos y se calculan literalmente al mismo tiempo, dividiendo el trabajo total entre el número de núcleos disponibles.",
    diagram: () => `
      <div class="core">CPU 1<br>Filas 0-63</div>
      <div class="core">CPU 2<br>Filas 64-127</div>
      <div class="core">CPU 3<br>Filas 128-191</div>
      <div class="core">CPU 4<br>Filas 192-255</div>
      <div class="note">Todos trabajan al mismo tiempo.</div>
    `,
  },
};

const defTitle = document.getElementById("def-title");
const defText = document.getElementById("def-text");
const defDiagram = document.getElementById("def-diagram");
const defDemo = document.getElementById("def-demo");
const defReal = document.getElementById("def-real");

function selectStrategy(key) {
  stopPlay();
  strategyKey = key;
  framesData = BUILDERS[key]();

  const def = definitions[key];
  defTitle.textContent = def.title;
  defText.textContent = def.text;
  defDiagram.innerHTML = def.diagram();
  defDemo.textContent = def.demo;
  defReal.textContent = def.real;

  renderCorePanel();

  document.querySelectorAll(".strategy-btn").forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.strategy === key);
  });

  applyStep(-1);
}

document.getElementById("strategy-menu").addEventListener("click", (e) => {
  const btn = e.target.closest(".strategy-btn");
  if (!btn) return;
  selectStrategy(btn.dataset.strategy);
});

renderStaticGrid(gridA, A, "a");
renderStaticGrid(gridB, B, "b");
selectStrategy("base");

// --- Benchmark real (fetch al servidor Rust) ---

const STRATEGY_LABELS = { base: "Base", simd: "SIMD", cache: "Cache Blocking", parallel: "Parallel" };

const benchGrid = document.getElementById("bench-grid");
const benchCharts = document.getElementById("bench-charts");
const chartTime = document.getElementById("chart-time");
const chartPerf = document.getElementById("chart-perf");
const benchStatus = document.getElementById("bench-status");
const btnRunBenchmark = document.getElementById("btn-run-benchmark");
const matrixSizeSelect = document.getElementById("matrix-size");

function renderBenchResults(results, n) {
  const baseResult = results.find((r) => r.strategy === "base") || results[0];
  const allMatch = results.every((r) => r.matches_base !== false);

  benchGrid.innerHTML = results
    .map((r) => {
      const speedup = baseResult.elapsed_ms > 0 ? baseResult.elapsed_ms / r.elapsed_ms : 1;
      const speedupRow = r.strategy === "base" ? "" : `<p>Speedup: <span>${speedup.toFixed(1)}x</span></p>`;
      return `
        <div class="bench-card">
          <h3>${STRATEGY_LABELS[r.strategy] || r.strategy}</h3>
          <p>Tiempo: <span>${r.elapsed_ms.toFixed(2)} ms</span></p>
          <p>GFLOPS: <span>${r.gflops.toFixed(2)}</span></p>
          ${speedupRow}
        </div>
      `;
    })
    .join("");

  const maxTime = Math.max(...results.map((r) => r.elapsed_ms), 0.001);
  const maxGflops = Math.max(...results.map((r) => r.gflops), 0.001);

  chartTime.innerHTML = results
    .map(
      (r) => `
      <div class="bar-row">
        <span class="bar-label">${STRATEGY_LABELS[r.strategy] || r.strategy}</span>
        <div class="bar" style="width:${Math.max(4, (r.elapsed_ms / maxTime) * 100)}%">${r.elapsed_ms.toFixed(1)} ms</div>
      </div>
    `
    )
    .join("");

  chartPerf.innerHTML = results
    .map(
      (r) => `
      <div class="bar-row">
        <span class="bar-label">${STRATEGY_LABELS[r.strategy] || r.strategy}</span>
        <div class="bar perf" style="width:${Math.max(4, (r.gflops / maxGflops) * 100)}%">${r.gflops.toFixed(2)}</div>
      </div>
    `
    )
    .join("");

  benchCharts.style.display = "grid";
  benchStatus.textContent = allMatch
    ? `✔ Medición real completada (N=${n}). Las 4 estrategias produjeron el mismo resultado (checksum verificado).`
    : `⚠ Medición completada (N=${n}), pero los checksums no coinciden exactamente (posible acumulación de error de punto flotante).`;
  benchStatus.className = allMatch ? "bench-status ok" : "bench-status error";
}

async function runRealBenchmark() {
  const n = matrixSizeSelect.value;
  btnRunBenchmark.disabled = true;
  benchStatus.className = "bench-status";
  benchStatus.textContent = `Ejecutando multiplicación real de matrices ${n}×${n} en las 4 estrategias (esto corre en Rust, puede tardar unos segundos)...`;
  benchCharts.style.display = "none";

  try {
    const res = await fetch(`/api/benchmark/all?n=${encodeURIComponent(n)}`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const results = await res.json();
    renderBenchResults(results, n);
  } catch (err) {
    benchStatus.className = "bench-status error";
    benchStatus.textContent =
      'No se pudo conectar con el servidor Rust. Ejecuta "cargo run" (o "cargo run --release"), elige la opción 5 y abre http://127.0.0.1:7878 en vez de abrir este archivo directamente.';
  } finally {
    btnRunBenchmark.disabled = false;
  }
}

btnRunBenchmark.addEventListener("click", runRealBenchmark);
