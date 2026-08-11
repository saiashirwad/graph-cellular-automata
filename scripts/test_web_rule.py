"""Check the browser rule math against GraphNCA (update_rate=1, no stochasticity).

Ports the JS step in web/js/ca/step-cpu.js to NumPy and compares one step to
the PyTorch module on the same state. Exit non-zero on mismatch.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from gnca.model import GraphNCA, alive_mask, load_rule


def step_numpy(x, off, src, deg, w1, b1, w2, gate_w, gate_b, alpha_idx=3):
    """Mirror of web/js/ca/step-cpu.js with update_rate=1."""
    n, c = x.shape
    h = b1.shape[0]
    in_w = 3 * c + 1

    alive = np.zeros(n, dtype=bool)
    for i in range(n):
        ok = x[i, alpha_idx] > 0.1
        if not ok:
            for e in range(off[i], off[i + 1]):
                if x[src[e], alpha_idx] > 0.1:
                    ok = True
                    break
        alive[i] = ok

    # a = x @ [Wg1; Wg2]
    a = np.zeros((n, 2 * c), dtype=np.float64)
    for i in range(n):
        for r in range(c):
            g1 = gate_w[r, :c]
            g2 = gate_w[r, c : 2 * c]
            a[i, r] = x[i] @ g1
            a[i, c + r] = x[i] @ g2

    xn = np.zeros_like(x)
    for i in range(n):
        d = deg[i] if deg[i] > 0 else 1.0
        inv = 1.0 / d
        z = np.zeros(in_w, dtype=np.float64)
        for ch in range(c):
            s = 0.0
            sd = 0.0
            for e in range(off[i], off[i + 1]):
                j = src[e]
                diff = x[j, ch] - x[i, ch]
                logit = a[j, ch] + a[i, c + ch] + gate_b[ch]
                g3 = gate_w[ch, 2 * c :]
                logit += g3 @ np.abs(x[j] - x[i])
                g = 2.0 / (1.0 + np.exp(-logit))
                s += x[j, ch]
                sd += g * diff
            z[ch] = x[i, ch]
            z[c + ch] = s * inv
            z[2 * c + ch] = sd * inv
        z[3 * c] = np.log1p(deg[i])

        hid = np.maximum(0.0, w1 @ z + b1)
        dx = w2 @ hid
        xn[i] = (x[i] + dx) if alive[i] else 0.0
    return xn


def csr_from_edges(edges, n):
    src_e, dst_e = edges[0], edges[1]
    order = np.argsort(dst_e)
    src = src_e[order].astype(np.int64)
    deg = np.bincount(dst_e, minlength=n).astype(np.float64)
    off = np.concatenate([[0], np.cumsum(deg.astype(np.int64))]).astype(np.int64)
    return off, src, deg


def main():
    device = "cpu"
    # synthetic tiny graph so the test does not need a checkpoint file on CI
    torch.manual_seed(0)
    n, c, k = 32, 16, 4
    pos = torch.rand(n, 2)
    # build a simple ring+chords edge_index
    edges = []
    for i in range(n):
        for d in range(1, k + 1):
            j = (i + d) % n
            edges.append((i, j))
            edges.append((j, i))
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()

    model = GraphNCA(channels=c, hidden=32).to(device)
    # non-trivial gate + degree column
    torch.nn.init.normal_(model.gate.weight, std=0.05)
    torch.nn.init.normal_(model.gate.bias, std=0.05)
    torch.nn.init.normal_(model.net[0].weight, std=0.05)
    torch.nn.init.normal_(model.net[0].bias, std=0.05)
    torch.nn.init.normal_(model.net[2].weight, std=0.02)

    x0 = torch.zeros(n, c)
    x0[0, 3:] = 1.0
    x0[1, :4] = torch.tensor([0.2, 0.1, 0.3, 0.8])
    x0[2, 4:] = torch.randn(c - 4) * 0.1
    x0[2, 3] = 0.5

    with torch.no_grad():
        # training applies alive_mask *after* the residual update on the
        # pre-update mask — model.forward does not apply it; train loop does.
        m = alive_mask(x0, edges)
        y_t = model(x0, edges, update_rate=1.0) * m

    off, src, deg = csr_from_edges(edges.numpy(), n)
    w1 = model.net[0].weight.detach().numpy().astype(np.float64)
    b1 = model.net[0].bias.detach().numpy().astype(np.float64)
    w2 = model.net[2].weight.detach().numpy().astype(np.float64)
    gw = model.gate.weight.detach().numpy().astype(np.float64)
    gb = model.gate.bias.detach().numpy().astype(np.float64)

    y_np = step_numpy(
        x0.numpy().astype(np.float64), off, src, deg, w1, b1, w2, gw, gb
    )

    err = np.max(np.abs(y_np - y_t.numpy()))
    print(f"max|numpy - torch| = {err:.3e}")
    if err > 1e-5:
        print("FAIL", file=sys.stderr)
        # also check identity-gate / zero-degree-column path
        return 1

    # identity gate + zero degree column must match a stripped 3C rule
    model2 = GraphNCA(channels=c, hidden=32)
    with torch.no_grad():
        model2.net[0].weight.zero_()
        model2.net[0].weight[:, : 3 * c] = model.net[0].weight[:, : 3 * c]
        model2.net[0].bias.copy_(model.net[0].bias)
        model2.net[2].weight.copy_(model.net[2].weight)
        # gate stays zero (identity)
    with torch.no_grad():
        m = alive_mask(x0, edges)
        y2 = model2(x0, edges, update_rate=1.0) * m
    y2_np = step_numpy(
        x0.numpy().astype(np.float64),
        off, src, deg,
        model2.net[0].weight.detach().numpy().astype(np.float64),
        model2.net[0].bias.detach().numpy().astype(np.float64),
        model2.net[2].weight.detach().numpy().astype(np.float64),
        model2.gate.weight.detach().numpy().astype(np.float64),
        model2.gate.bias.detach().numpy().astype(np.float64),
    )
    err2 = np.max(np.abs(y2_np - y2.numpy()))
    print(f"max|numpy - torch| (identity gate) = {err2:.3e}")
    if err2 > 1e-5:
        print("FAIL identity-gate path", file=sys.stderr)
        return 1

    print("ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
