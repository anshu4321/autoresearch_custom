const REFRESH_MS = 5000;
let perfChart = null;
let lastSeriesKey = "";
let lastRunsKey = "";

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function fmtNum(v, digits = 3) {
  if (typeof v !== "number" || Number.isNaN(v)) return "--";
  return v.toFixed(digits);
}

function fmtPct(v, digits = 1) {
  if (typeof v !== "number" || Number.isNaN(v)) return "--";
  return `${(v * 100).toFixed(digits)}%`;
}

function fmtDuration(seconds) {
  if (typeof seconds !== "number" || Number.isNaN(seconds)) return "--";
  if (seconds < 60) return `${seconds.toFixed(0)}s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

function applyLivePill(state) {
  const pill = document.getElementById("livePill");
  const normalized = (state || "idle").toLowerCase();
  pill.className = `pill ${normalized}`;
  pill.textContent = normalized.toUpperCase();
}

function renderKPIs(snapshot) {
  const stats = snapshot.stats || {};
  const best = stats.best || null;
  const latest = stats.latest || null;

  document.getElementById("bestNdcg").textContent = best ? fmtNum(best.val_ndcg_at_10, 6) : "--";
  document.getElementById("bestCommit").textContent = best ? `commit ${best.commit}` : "commit --";

  document.getElementById("latestNdcg").textContent = latest ? fmtNum(latest.val_ndcg_at_10, 6) : "--";
  if (latest && best) {
    const delta = latest.val_ndcg_at_10 - best.val_ndcg_at_10;
    const sign = delta > 0 ? "+" : "";
    document.getElementById("latestDelta").textContent = `vs best ${sign}${delta.toFixed(6)}`;
  } else {
    document.getElementById("latestDelta").textContent = "delta --";
  }

  document.getElementById("totalRuns").textContent = `${stats.total_runs ?? 0}`;
  document.getElementById("keepRatio").textContent = `keep ratio ${fmtPct(stats.keep_ratio ?? 0, 1)} | crashes ${stats.crashes ?? 0}`;

  document.getElementById("projectedRuns").textContent = `${stats.projected_runs_7h ?? "--"}`;
  document.getElementById("cycleTime").textContent = `cycle ${fmtDuration(stats.cycle_seconds)}`;

  const git = snapshot.git || {};
  document.getElementById("gitInfo").textContent = `Branch: ${git.branch || "--"} | Head: ${git.head || "--"}`;
  document.getElementById("lastUpdated").textContent = `Last update: ${snapshot.generated_at || "--"}`;
}

function renderLive(snapshot) {
  const live = snapshot.live || {};
  applyLivePill(live.state || "idle");

  const progress = live.progress || {};
  const summary = live.summary || {};

  document.getElementById("liveStep").textContent = progress.step ?? "--";
  document.getElementById("liveProgress").textContent = progress.pct != null ? `${fmtNum(progress.pct, 1)}%` : "--";
  document.getElementById("liveLoss").textContent = progress.loss != null ? fmtNum(progress.loss, 4) : "--";
  document.getElementById("liveRemaining").textContent = progress.remaining_s != null ? `${progress.remaining_s}s` : "--";

  document.getElementById("liveVram").textContent =
    summary.peak_vram_mb != null ? `${fmtNum(summary.peak_vram_mb / 1024, 2)} GB` : "--";
  document.getElementById("liveTrainSec").textContent =
    summary.training_seconds != null ? `${fmtNum(summary.training_seconds, 1)}s` : "--";
}

function initOrUpdateChart(snapshot) {
  const labels = snapshot.series?.labels || [];
  const ndcg = snapshot.series?.ndcg || [];
  const best = snapshot.series?.best_so_far || [];
  const seriesKey = JSON.stringify({ labels, ndcg, best });

  const ctx = document.getElementById("perfChart").getContext("2d");
  if (!perfChart) {
    const grad = ctx.createLinearGradient(0, 0, 0, 320);
    grad.addColorStop(0, "rgba(56, 213, 255, 0.45)");
    grad.addColorStop(1, "rgba(56, 213, 255, 0.02)");

    perfChart = new Chart(ctx, {
      type: "line",
      data: {
        labels,
        datasets: [
          {
            label: "nDCG@10",
            data: ndcg,
            borderColor: "#38d5ff",
            backgroundColor: grad,
            borderWidth: 2.2,
            pointRadius: 2.4,
            pointHoverRadius: 4,
            tension: 0.28,
            fill: true,
          },
          {
            label: "Best so far",
            data: best,
            borderColor: "#ff9f43",
            borderWidth: 2,
            pointRadius: 0,
            tension: 0.1,
            borderDash: [8, 6],
            fill: false,
          },
        ],
      },
      options: {
        maintainAspectRatio: false,
        responsive: true,
        animation: false,
        plugins: {
          legend: {
            labels: {
              color: "#dcecff",
              usePointStyle: true,
              boxWidth: 8,
              font: { family: "IBM Plex Mono", size: 11 },
            },
          },
          tooltip: {
            backgroundColor: "rgba(9, 23, 33, 0.95)",
            borderColor: "rgba(56, 213, 255, 0.35)",
            borderWidth: 1,
            titleColor: "#fff",
            bodyColor: "#cde6ff",
            padding: 10,
          },
        },
        scales: {
          x: {
            ticks: { color: "#a2bccc", maxTicksLimit: 10 },
            grid: { color: "rgba(117, 170, 196, 0.14)" },
          },
          y: {
            ticks: { color: "#a2bccc" },
            grid: { color: "rgba(117, 170, 196, 0.14)" },
          },
        },
      },
    });
    lastSeriesKey = seriesKey;
    return;
  }

  if (seriesKey === lastSeriesKey) {
    return;
  }
  lastSeriesKey = seriesKey;
  perfChart.data.labels = labels;
  perfChart.data.datasets[0].data = ndcg;
  perfChart.data.datasets[1].data = best;
  perfChart.update("none");
}

function renderRunsTable(snapshot) {
  const body = document.getElementById("runsBody");
  const runs = [...(snapshot.runs || [])].reverse();
  const runsKey = JSON.stringify(runs);

  if (runsKey === lastRunsKey) {
    return;
  }
  lastRunsKey = runsKey;

  if (!runs.length) {
    body.innerHTML =
      '<tr><td colspan="6" class="mono" style="color:#9eb7c9">No runs yet. Start the experiment loop.</td></tr>';
    return;
  }

  const rows = runs
    .map((run) => {
      const statusClass = escapeHtml(run.status || "unknown");
      const index = escapeHtml(run.index ?? "--");
      const commit = escapeHtml(run.commit || "--");
      const desc = run.description || "--";
      const safeDesc = escapeHtml(desc);
      return `
        <tr>
          <td class="mono">${index}</td>
          <td class="mono">${commit}</td>
          <td class="mono">${fmtNum(run.val_ndcg_at_10, 6)}</td>
          <td class="mono">${fmtNum(run.memory_gb, 1)}</td>
          <td><span class="badge ${statusClass}">${statusClass}</span></td>
          <td title="${safeDesc}">${safeDesc}</td>
        </tr>
      `;
    })
    .join("");

  body.innerHTML = rows;
}

async function refresh() {
  try {
    const res = await fetch(`/api/snapshot?ts=${Date.now()}`, { cache: "no-store" });
    if (!res.ok) {
      throw new Error(`snapshot request failed: ${res.status}`);
    }
    const snapshot = await res.json();

    renderKPIs(snapshot);
    renderLive(snapshot);
    initOrUpdateChart(snapshot);
    renderRunsTable(snapshot);
  } catch (err) {
    console.error(err);
    applyLivePill("stalled");
    document.getElementById("lastUpdated").textContent = "Last update: fetch error";
  }
}

refresh();
setInterval(refresh, REFRESH_MS);
