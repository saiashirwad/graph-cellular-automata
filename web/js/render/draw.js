import { findView } from "./views.js";
import { findLayout } from "../layout/index.js";

const RIPPLE_MS = 900;
const DIE_MS = 1100;

export function createRenderer(cv) {
  const ctx2d = cv.getContext("2d");
  let CW = 600, CH = 600, S = 600, OX = 0, OY = 0, DPR = 1;
  let zoom = 1, zoomT = 1;

  function resize() {
    DPR = Math.min(window.devicePixelRatio || 1, 2);
    CW = Math.round((cv.clientWidth || 600) * DPR);
    CH = Math.round((cv.clientHeight || 600) * DPR);
    cv.width = CW; cv.height = CH;
    const top = 96 * DPR, bottom = 175 * DPR;
    const band = Math.max(CH - top - bottom, CH * 0.4);
    S = Math.min(CW * 0.92, band);
    OX = (CW - S) / 2;
    OY = top + (band - S) / 2;
  }
  new ResizeObserver(resize).observe(cv);
  cv.addEventListener("wheel", e => {
    e.preventDefault();
    zoomT = Math.min(1.6, Math.max(0.7, zoomT * Math.exp(-e.deltaY * 0.0014)));
  }, { passive: false });

  const px = p => OX + S * 0.5 + (p - 0.5) * S * zoom;
  const py = p => OY + S * 0.5 + (0.5 - p) * S * zoom;

  function canvasPos(ev) {
    const r = cv.getBoundingClientRect();
    return {
      x: 0.5 + ((ev.clientX - r.left) * DPR - OX - S * 0.5) / (S * zoom),
      y: 0.5 - ((ev.clientY - r.top) * DPR - OY - S * 0.5) / (S * zoom),
    };
  }

  function easeOut(u) {
    const t = 1 - u;
    return 1 - t * t * t;
  }

  function drawRipples(app, now) {
    for (let i = app.ripples.length - 1; i >= 0; i--) {
      const r = app.ripples[i];
      const raw = (now - r.t0) / (r.ms || RIPPLE_MS);
      if (raw >= 1) { app.ripples.splice(i, 1); continue; }
      const u = easeOut(raw);
      const a = (1 - raw) * (r.alpha ?? 0.28);
      ctx2d.strokeStyle = `rgba(${r.col},${a.toFixed(3)})`;
      ctx2d.lineWidth = (r.width ?? 1.1) * DPR;
      ctx2d.beginPath();
      ctx2d.arc(
        px(r.x), py(r.y),
        ((r.r0 ?? 0.01) + (r.r1 ?? 0.07) * u) * S * zoom,
        0, 6.2832,
      );
      ctx2d.stroke();
    }
  }

  /** Soft dissolve of nodes that just died — shrink, drift, fade. No pop. */
  function drawDying(app, now) {
    if (!app.dying?.length) return;
    ctx2d.globalCompositeOperation = "source-over";
    for (let i = app.dying.length - 1; i >= 0; i--) {
      const d = app.dying[i];
      const raw = (now - d.t0) / DIE_MS;
      if (raw >= 1) { app.dying.splice(i, 1); continue; }
      // hold a beat, then ease away
      const u = raw < 0.12 ? 0 : easeOut((raw - 0.12) / 0.88);
      const fade = 1 - u;
      const drift = 0.012 * u; // gentle lift in display space
      const X = px(d.x + d.dx * u * 0.4);
      const Y = py(d.y + drift + d.dy * u * 0.25);
      const rad = d.r * DPR * (1 - 0.75 * u) * (0.85 + 0.15 * Math.sin(raw * 6 + d.phase));

      // soft ember core
      ctx2d.fillStyle = `rgba(${d.R},${d.G},${d.B},${(0.55 * fade * d.a).toFixed(3)})`;
      ctx2d.beginPath();
      ctx2d.arc(X, Y, Math.max(0.4 * DPR, rad), 0, 6.2832);
      ctx2d.fill();

      // faint ash ring expanding as the core shrinks
      if (u > 0.05) {
        ctx2d.strokeStyle = `rgba(${d.R},${d.G},${d.B},${(0.22 * fade).toFixed(3)})`;
        ctx2d.lineWidth = 0.8 * DPR;
        ctx2d.beginPath();
        ctx2d.arc(X, Y, rad * (1.4 + 1.8 * u), 0, 6.2832);
        ctx2d.stroke();
      }

      // two tiny motes drifting off
      for (let m = 0; m < 2; m++) {
        const ang = d.phase + m * 2.1;
        const mx = X + Math.cos(ang) * rad * (1.2 + 2.5 * u);
        const my = Y + Math.sin(ang) * rad * (1.2 + 2.5 * u) - 4 * DPR * u;
        ctx2d.fillStyle = `rgba(${d.R},${d.G},${d.B},${(0.35 * fade * (1 - u)).toFixed(3)})`;
        ctx2d.fillRect(mx, my, Math.max(1, 1.2 * DPR), Math.max(1, 1.2 * DPR));
      }
    }
  }

  function draw(app, now) {
    zoom += (zoomT - zoom) * 0.18;
    const { n, c, alphaIdx: A, x, off, src, tgt, dim } = app;
    const { dispPos, dispD, showEdges, showGhost, viewId, views } = app;

    ctx2d.clearRect(0, 0, CW, CH);

    if (showEdges) {
      ctx2d.lineWidth = Math.max(1, 0.5 * DPR);
      ctx2d.strokeStyle = "rgba(255,255,255,0.09)";
      ctx2d.beginPath();
      for (let i = 0; i < n; i++)
        for (let k = off[i]; k < off[i + 1]; k++) {
          const j = src[k];
          if (j <= i) continue;
          ctx2d.moveTo(px(dispPos[i * 2]), py(dispPos[i * 2 + 1]));
          ctx2d.lineTo(px(dispPos[j * 2]), py(dispPos[j * 2 + 1]));
        }
      ctx2d.stroke();
    }

    if (showGhost && viewId === "rgb" && (app.layoutId === "space" || dim > 2)) {
      ctx2d.globalCompositeOperation = "source-over";
      for (let i = 0; i < n; i++) {
        const a = tgt[i * 4 + 3];
        if (a < 0.05) continue;
        ctx2d.fillStyle = `rgba(${(tgt[i * 4] * 255) | 0},${(tgt[i * 4 + 1] * 255) | 0},${(tgt[i * 4 + 2] * 255) | 0},0.14)`;
        ctx2d.beginPath();
        ctx2d.arc(px(dispPos[i * 2]), py(dispPos[i * 2 + 1]), 2.4 * DPR, 0, 6.2832);
        ctx2d.fill();
      }
    }

    if (app.layoutId === "map" && app.umap?.pts) {
      ctx2d.fillStyle = "rgba(255,255,255,0.10)";
      const P = app.umap.pts;
      for (let k = 0; k < P.length; k += 2)
        ctx2d.fillRect(px((P[k] + 1) / 2), py((P[k + 1] + 1) / 2), DPR, DPR);
    }

    const geom = {
      px, py, dispPos, dispD, dpr: DPR,
      rCore: 3.2 * DPR, rHalo: 9 * DPR,
    };
    const view = findView(views, viewId);
    if (view?.paint) view.paint(ctx2d, app, geom);

    drawDying(app, now);
    drawRipples(app, now);

    let hovered = -1;
    if (app.ptr && !app.drawing) {
      let hi = -1, hd = 0.035 * 0.035;
      for (let i = 0; i < n; i++) {
        const dx = dispPos[i * 2] - app.ptr.x, dy = dispPos[i * 2 + 1] - app.ptr.y;
        const d = dx * dx + dy * dy;
        if (d < hd) { hd = d; hi = i; }
      }
      if (hi >= 0) {
        const X = px(dispPos[hi * 2]), Y = py(dispPos[hi * 2 + 1]);
        ctx2d.strokeStyle = "rgba(255,255,255,0.30)";
        ctx2d.lineWidth = 1 * DPR;
        ctx2d.beginPath();
        for (let e = off[hi]; e < off[hi + 1]; e++) {
          const j = src[e];
          ctx2d.moveTo(X, Y);
          ctx2d.lineTo(px(dispPos[j * 2]), py(dispPos[j * 2 + 1]));
        }
        ctx2d.stroke();
        ctx2d.strokeStyle = "rgba(255,255,255,0.75)";
        for (let e = off[hi]; e < off[hi + 1]; e++) {
          const j = src[e];
          ctx2d.beginPath();
          ctx2d.arc(px(dispPos[j * 2]), py(dispPos[j * 2 + 1]), 3 * DPR, 0, 6.2832);
          ctx2d.stroke();
        }
        ctx2d.strokeStyle = "rgba(255,255,255,0.9)";
        ctx2d.lineWidth = 1.2 * DPR;
        ctx2d.beginPath();
        ctx2d.arc(X, Y, 4.5 * DPR, 0, 6.2832);
        ctx2d.stroke();
        hovered = hi;
      }
    }

    const hud = document.getElementById("hud");
    if (hud) {
      hud.textContent = hovered >= 0
        ? "node " + hovered + " · deg " + (off[hovered + 1] - off[hovered]) +
          " · α " + x[hovered * c + A].toFixed(2)
        : typeof viewId === "number"
          ? (viewId === 3 ? "α · aliveness" : "channel " + viewId + " · meaning learned in training")
          : "";
    }

    if (app.ptr) {
      ctx2d.strokeStyle = app.drawing ? "rgba(255,111,145,0.55)" : "rgba(255,255,255,0.35)";
      ctx2d.lineWidth = 1.2 * DPR;
      ctx2d.setLineDash([4 * DPR, 4 * DPR]);
      ctx2d.beginPath();
      ctx2d.arc(px(app.ptr.x), py(app.ptr.y), app.brushR * S * zoom, 0, 6.2832);
      ctx2d.stroke();
      ctx2d.setLineDash([]);
      ctx2d.fillStyle = app.drawing ? "rgba(255,111,145,0.7)" : "rgba(255,255,255,0.7)";
      ctx2d.fillRect(px(app.ptr.x) - DPR, py(app.ptr.y) - DPR, 2 * DPR, 2 * DPR);
    }
  }

  function paintThumbs(app) {
    if (!app.dispPos || !app.thumbs?.length) return;
    const r = Math.max(1, Math.round(app.thumbs[0].s / 54));
    const { n, c, alphaIdx: A, x, dispPos } = app;
    for (const th of app.thumbs) {
      const g = th.g, Sz = th.s;
      g.clearRect(0, 0, Sz, Sz);
      let m = 1e-6;
      for (let i = 0; i < n; i++)
        if (x[i * c + A] > 0.05) {
          const v = Math.abs(x[i * c + th.c]);
          if (v > m) m = v;
        }
      for (let i = 0; i < n; i++) {
        if (x[i * c + A] < 0.05) continue;
        const v = x[i * c + th.c] / m;
        const a = Math.min(Math.abs(v), 1);
        if (a < 0.04) continue;
        g.fillStyle = v > 0 ? `rgba(255,111,145,${a.toFixed(3)})` : `rgba(90,209,232,${a.toFixed(3)})`;
        g.fillRect(dispPos[i * 2] * Sz - r, (1 - dispPos[i * 2 + 1]) * Sz - r, 2 * r, 2 * r);
      }
    }
  }

  resize();
  return { resize, draw, paintThumbs, canvasPos, get zoom() { return zoom; }, get S() { return S; }, get dpr() { return DPR; } };
}

export function loss(app) {
  let s = 0;
  const { n, c, x, tgt } = app;
  for (let i = 0; i < n; i++)
    for (let ch = 0; ch < 4; ch++) {
      const d = x[i * c + ch] - tgt[i * 4 + ch];
      s += d * d;
    }
  return s / (n * 4);
}

export function alivePct(app) {
  let a = 0;
  for (let i = 0; i < app.n; i++) if (app.x[i * app.c + app.alphaIdx] > 0.1) a++;
  return a / app.n * 100;
}

export function updateLegend(app) {
  const legendEl = document.getElementById("legend");
  if (!legendEl) return;
  const view = findView(app.views, app.viewId);
  const layout = findLayout(app.layoutId, app);
  const parts = [...(view?.legParts?.(app) || [])];
  parts.push(layout.leg, "hover a node to see its neighbors");
  legendEl.innerHTML = parts.join(" · ");
}
