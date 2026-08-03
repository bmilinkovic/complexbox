"""SSDI-1 full pipeline — publication-quality reproduction of the MATLAB demo.

Faithful port of MATLAB's ``sim_model.m`` → ``preoptimise_dd.m`` →
``optimise_dd.m`` for a known modular ground truth (`tnet243`: a 9-node
network with a 2-module → 4-module directed connection and an isolated
3-module).

Run parameters can be selected via the ``MODE`` environment variable:

- ``MODE=fast``  — small numbers, finishes in ~30-40 s on a laptop.
- ``MODE=pub``   — publication-sized restarts with trimmed iteration limits.
- ``MODE=full``  — MATLAB defaults (100 restarts × 10000 iterations);
  takes several minutes but produces fully-converged, publication-quality
  results.

Use ``STAGE=model``, ``preopt``, ``opt``, ``figs-preopt``, ``figs-opt``,
``figs``, or ``stats`` to run or resume a subset of the pipeline. The default
``STAGE=all`` runs every stage. Cached optimisation stages are validated against
the numerical configuration before reuse.

Figures are saved as both PNG (300 dpi screen) and PDF (vector) into the
ignored ``results/ssdi_pipeline/`` directory. Set ``COMPLEXBOX_OUTPUT`` to
choose another location. Set ``COMPLEXBOX_BACKEND=torch`` (and optionally
``COMPLEXBOX_DEVICE=cuda``) to batch the multi-restart optimisations.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from complexbox import mvgc, ssdi

OUT = Path(
    os.environ.get(
        "COMPLEXBOX_OUTPUT",
        Path(__file__).parents[1] / "results" / "ssdi_pipeline",
    )
)
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Publication-quality matplotlib style (Nature-inspired)
# ---------------------------------------------------------------------------

# Wong colorblind-safe palette
WONG = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Module colours for the 2-4-3 schematic
MOD_2 = "#0072B2"
MOD_4 = "#D55E00"
MOD_3 = "#009E73"

plt.rcParams.update(
    {
        # Typography — sans-serif, slightly larger than mpl defaults
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.titlesize": 11,
        # Lines and ticks
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        # Spines: hide top & right for a cleaner look
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Colors and grid
        "axes.prop_cycle": plt.cycler(color=WONG),
        "axes.grid": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        # Figure
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def _save(fig: plt.Figure, name: str) -> None:
    """Save figure as PNG (300 dpi) and PDF (vector)."""
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _add_panel_label(ax: plt.Axes, label: str, x: float = -0.18, y: float = 1.05) -> None:
    """Add a bold lowercase panel label (a, b, c, …) in the corner."""
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


# ---------------------------------------------------------------------------
# Pipeline parameters
# ---------------------------------------------------------------------------

MODE = os.environ.get("MODE", "fast").lower()

if MODE == "full":
    NRUNSP = 100
    NITERSP = 10_000
    NITERSO = 10_000
    HAXA_N = 10_000
    VAR_ORDER = 7
    CTOL = 1e-6
    print("=== Publication-quality run (MATLAB defaults) ===")
elif MODE == "pub":
    NRUNSP = 100
    NITERSP = 5_000
    NITERSO = 2_000
    HAXA_N = 10_000
    VAR_ORDER = 7
    CTOL = 1e-6
    print("=== Publication run (slightly trimmed maxiters for in-session execution) ===")
elif MODE == "fast":
    NRUNSP = 100
    NITERSP = 2_000
    NITERSO = 200
    HAXA_N = 2_000
    VAR_ORDER = 5
    CTOL = 1e-2
    print("=== Fast tutorial run ===")
else:
    raise ValueError(f"unknown MODE={MODE!r}; use 'fast', 'pub', or 'full'")

STAGE = os.environ.get("STAGE", "all").lower()
VALID_STAGES = ("all", "model", "preopt", "opt", "figs-preopt", "figs-opt", "figs", "stats")
if STAGE not in VALID_STAGES:
    choices = ", ".join(VALID_STAGES)
    raise ValueError(f"unknown STAGE={STAGE!r}; use one of: {choices}")

PREOPT_CACHE = OUT / f"_preopt_{MODE}.npz"
REFINED_CACHE = OUT / f"_refined_{MODE}.npz"
CACHE_VERSION = 1

# Common (MATLAB defaults)
RHO = 0.9
RMII = 1.0
W_DECAY = 1.0
GDSIG0P = 1.0
GDLSP = 2.0
GDTOLP = 1e-8
GDSIG0O = 0.1
GDLSO = 2.0
GDTOLO = 1e-10
SEED = 20260516
STATS_SEED = SEED + 1
BACKEND = os.environ.get("COMPLEXBOX_BACKEND", "numpy").lower()
DEVICE = os.environ.get("COMPLEXBOX_DEVICE", "cpu")
if BACKEND not in {"numpy", "torch"}:
    raise ValueError("COMPLEXBOX_BACKEND must be 'numpy' or 'torch'")


# ---------------------------------------------------------------------------
# 1. sim_model
# ---------------------------------------------------------------------------


def sim_model(rng: np.random.Generator):
    CON = ssdi.tnet243()
    n = CON.shape[0]
    V0 = mvgc.corr_rand(n, g=RMII, rng=rng)
    ARA0 = mvgc.var_rand(CON, VAR_ORDER, rho=RHO, w=W_DECAY, rng=rng)
    A0, C0, K0, _ = mvgc.var_to_ss(ARA0, V0)
    gc = mvgc.var_to_pwcgc(ARA0, V0)
    fres = mvgc.var2fres(ARA0, V0)
    mdescript = f"{n}-variable VAR({VAR_ORDER})"
    print(f"\n{mdescript}: rho(A) = {mvgc.specnorm(ARA0):.4f}, fres = {fres}")
    return mdescript, CON, n, A0, C0, K0, V0, ARA0, gc, fres


def fig1_model(CON, gc, mdescript):
    """Figure 1: Ground-truth model — schematic, connectivity, GC."""
    fig = plt.figure(figsize=(7.5, 3.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1], wspace=0.45, top=0.85, bottom=0.18)
    n = CON.shape[0]

    # Panel (a): topology schematic — three modules laid out as separate clusters
    axA = fig.add_subplot(gs[0, 0])
    axA.set_xlim(-1.3, 1.3)
    axA.set_ylim(-1.4, 1.3)
    axA.set_aspect("equal")
    axA.set_axis_off()
    # Module colours per node
    mod_idx = np.array([0] * 2 + [1] * 4 + [2] * 3)
    colours = [MOD_2, MOD_4, MOD_3]
    # Hand-laid-out positions: 2-mod top-left, 4-mod top-right, 3-mod bottom-centre
    pos = np.array(
        [
            [-1.05, 0.55],
            [-1.05, 0.95],  # 2-module (0, 1)
            [0.45, 1.05],
            [1.05, 0.75],
            [1.05, 0.25],
            [0.45, -0.05],  # 4-module (2-5)
            [-0.55, -0.85],
            [0.0, -1.1],
            [0.55, -0.85],  # 3-module (6-8)
        ]
    )
    # Within-module convex hull "halos"
    from matplotlib.patches import FancyBboxPatch

    for m in range(3):
        idxs = np.where(mod_idx == m)[0]
        xs = pos[idxs, 0]
        ys = pos[idxs, 1]
        cx, cy = xs.mean(), ys.mean()
        w = np.ptp(xs) + 0.55
        h = np.ptp(ys) + 0.55
        rect = FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.18",
            facecolor=colours[m],
            alpha=0.12,
            edgecolor="none",
            zorder=1,
        )
        axA.add_patch(rect)
    # Inter-module directed edges (the 2 → 4 connection)
    for i in range(n):
        for j in range(n):
            if i == j or CON[i, j] == 0:
                continue
            if mod_idx[i] == mod_idx[j]:
                continue
            x0, y0 = pos[j]
            x1, y1 = pos[i]
            # Pull-back to the node edge
            dx, dy = x1 - x0, y1 - y0
            d = np.hypot(dx, dy)
            shrink = 0.18 / d
            axA.annotate(
                "",
                xy=(x1 - dx * shrink, y1 - dy * shrink),
                xytext=(x0 + dx * shrink, y0 + dy * shrink),
                arrowprops=dict(arrowstyle="->", color="0.30", lw=0.7, shrinkA=0, shrinkB=0),
                zorder=5,
            )
    # Node markers + labels
    for i in range(n):
        axA.scatter(
            pos[i, 0],
            pos[i, 1],
            s=240,
            c=colours[mod_idx[i]],
            edgecolors="black",
            linewidths=0.8,
            zorder=10,
        )
        axA.text(
            pos[i, 0],
            pos[i, 1],
            str(i),
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="white",
            zorder=11,
        )
    # Module legend
    handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MOD_2,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="2-module",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MOD_4,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="4-module",
        ),
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=MOD_3,
            markersize=8,
            markeredgecolor="black",
            markeredgewidth=0.7,
            label="3-module",
        ),
    ]
    # Legend goes outside the subplot, anchored to the figure
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=3,
        frameon=False,
        fontsize=8,
        columnspacing=2.0,
        handletextpad=0.5,
    )
    axA.set_title("Network topology", pad=8)
    _add_panel_label(axA, "a", x=-0.05, y=0.97)

    # Panel (b): connectivity matrix
    axB = fig.add_subplot(gs[0, 1])
    axB.imshow(CON, cmap="Greys", vmin=0, vmax=1, aspect="equal")
    axB.set_xticks(range(n))
    axB.set_yticks(range(n))
    axB.set_xticklabels(range(n), fontsize=7)
    axB.set_yticklabels(range(n), fontsize=7)
    axB.set_xlabel("source j")
    axB.set_ylabel("target i")
    axB.set_title("Connectivity CON", pad=8)
    # Module separators
    for v in [1.5, 5.5]:
        axB.axhline(v, color="white", lw=0.8)
        axB.axvline(v, color="white", lw=0.8)
    _add_panel_label(axB, "b")

    # Panel (c): pairwise-conditional GC
    axC = fig.add_subplot(gs[0, 2])
    gc_plot = np.where(np.isnan(gc), 0, gc)
    im = axC.imshow(gc_plot, cmap="viridis", aspect="equal")
    axC.set_xticks(range(n))
    axC.set_yticks(range(n))
    axC.set_xticklabels(range(n), fontsize=7)
    axC.set_yticklabels(range(n), fontsize=7)
    axC.set_xlabel("source j")
    axC.set_ylabel("target i")
    axC.set_title("Pairwise-conditional GC", pad=8)
    for v in [1.5, 5.5]:
        axC.axhline(v, color="white", lw=0.8)
        axC.axvline(v, color="white", lw=0.8)
    cbar = fig.colorbar(im, ax=axC, fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=7)
    cbar.outline.set_linewidth(0.5)
    _add_panel_label(axC, "c")

    fig.suptitle(f"Ground-truth model ({mdescript})", y=0.97, fontsize=11)
    _save(fig, "fig01_model")


# ---------------------------------------------------------------------------
# 2. preoptimise_dd
# ---------------------------------------------------------------------------


def preoptimise_dd(ARA0, V0, m_dims, rng):
    n = V0.shape[0]
    ARA, V = ssdi.transform_var(ARA0, V0)
    CAK = ARA
    results = {}
    for m in m_dims:
        L0p = ssdi.rand_orthonormal(n, m, runs=NRUNSP, rng=rng)
        t0 = time.time()
        dds, Lp, conv, hist = ssdi.opt_gd_ddx_mruns(
            CAK,
            L0p,
            maxiters=NITERSP,
            variant=2,
            gdsig0=GDSIG0P,
            gdls=GDLSP,
            tol=GDTOLP,
            history=True,
            backend=BACKEND,
            device=DEVICE,
            run_chunk_size=32,
            lag_chunk_size=16,
        )
        Loptp = ssdi.itransform_subspace(Lp, V0)
        goptp = ssdi.gmetrics(Loptp)
        results[m] = dict(
            dds=dds, Lp=Lp, Loptp=Loptp, goptp=goptp, conv=conv, hist=hist, cpu=time.time() - t0
        )
        n_conv = sum(c > 0 for c in conv)
        print(
            f"  preopt m={m}: best dd={dds[0]:.4e}  ({n_conv}/{NRUNSP} conv, {time.time() - t0:.1f}s)"
        )
    return results, ARA, V


def _multi_panel_dd_per_run(results, title, fname, ylabel):
    """3 panels: (a) DD per run, (b) histories grid, (c) Grass-distance grid."""
    m_dims = sorted(results)
    n_m = len(m_dims)

    # ---- (a) DD per run ----
    fig = plt.figure(figsize=(7.0, 3.6))
    ax = fig.add_subplot(1, 1, 1)
    cmap = plt.cm.viridis(np.linspace(0, 0.9, n_m))
    for c, m in zip(cmap, m_dims):
        d = results[m]["dds"]
        ax.semilogy(np.arange(1, d.size + 1), d, "o-", color=c, ms=3, lw=1.0, label=f"m = {m}")
    ax.set_xlabel("Run (sorted by DD)")
    ax.set_ylabel(ylabel)
    ax.set_title(title + ": DD per restart, all macro sizes")
    # Put the legend outside the plot frame
    ax.legend(
        bbox_to_anchor=(1.02, 1.0),
        loc="upper left",
        fontsize=8,
        frameon=False,
        borderaxespad=0.0,
    )
    fig.tight_layout()
    _save(fig, fname + "_a")

    # ---- (b) histories grid ----
    ncols = 4
    nrows = (n_m + ncols - 1) // ncols
    fig = plt.figure(figsize=(7.0, 2.0 * nrows))
    for idx, m in enumerate(m_dims):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        hist_color = WONG[5] if "Pre" in title else WONG[6]
        for h in results[m]["hist"]:
            if h is None or h.size == 0:
                continue
            ax.semilogy(h[:, 0], color=hist_color, alpha=0.25, lw=0.5)
        ax.set_title(f"m = {m}", fontsize=9)
        ax.set_xlabel("Iteration", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)
    fig.suptitle(title + ": convergence trajectories", y=1.02)
    fig.tight_layout()
    _save(fig, fname + "_b")

    # ---- (c) Grass-distance grid ----
    fig = plt.figure(figsize=(7.0, 2.0 * nrows))
    ims = []
    for idx, m in enumerate(m_dims):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        D = results[m]["goptp"] if "goptp" in results[m] else results[m]["gopto"]
        # restrict to reasonable size (cluster grids may be small)
        im = ax.imshow(
            D, cmap="magma", vmin=0, vmax=1, aspect="auto" if D.shape[0] != D.shape[1] else "equal"
        )
        ims.append(im)
        ax.set_title(f"m = {m}", fontsize=9)
        ax.set_xlabel("Run", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("Run", fontsize=8)
    if ims:
        cbar = fig.colorbar(
            ims[-1],
            ax=fig.get_axes(),
            shrink=0.5,
            label="Grassmannian distance",
            fraction=0.025,
            pad=0.02,
        )
        cbar.ax.tick_params(labelsize=7)
        cbar.outline.set_linewidth(0.5)
    fig.suptitle(title + ": inter-optima Grassmannian distances", y=1.02)
    _save(fig, fname + "_c")


def fig2_preopt(preopt_results):
    _multi_panel_dd_per_run(
        preopt_results,
        "Pre-optimisation",
        "fig02_preopt",
        "Proxy DD",
    )


# ---------------------------------------------------------------------------
# 3. optimise_dd
# ---------------------------------------------------------------------------


def optimise_dd(preopt_results, ARA, V0, fres_use):
    H = mvgc.var2trfun(ARA, fres_use)
    refined = {}
    for m, pr in preopt_results.items():
        clust = ssdi.Lcluster(pr["goptp"], tol=CTOL)
        L0o = pr["Lp"][:, :, clust.uidx]
        t0 = time.time()
        dds, Lo, conv, hist = ssdi.opt_gd_dds_mruns(
            H,
            L0o,
            maxiters=NITERSO,
            variant=2,
            gdsig0=GDSIG0O,
            gdls=GDLSO,
            tol=GDTOLO,
            history=True,
            backend=BACKEND,
            device=DEVICE,
            run_chunk_size=32,
            frequency_chunk_size=64,
        )
        Lopto = ssdi.itransform_subspace(Lo, V0)
        gopto = ssdi.gmetrics(Lopto)
        refined[m] = dict(
            dds=dds,
            Lo=Lo,
            Lopto=Lopto,
            gopto=gopto,
            conv=conv,
            hist=hist,
            clust=clust,
            cpu=time.time() - t0,
        )
        print(
            f"  opt    m={m}: {clust.nruns} clusters, best dd={dds[0]:.4e} ({time.time() - t0:.1f}s)"
        )
    return refined


def fig3_opt(refined_results):
    # Re-key 'goptp' for the shared helper
    rr = {
        m: {"dds": r["dds"], "hist": r["hist"], "goptp": r["gopto"]}
        for m, r in refined_results.items()
    }
    _multi_panel_dd_per_run(
        rr,
        "Spectral refinement",
        "fig03_opt",
        "Spectral DD",
    )


# ---------------------------------------------------------------------------
# 4. β-statistic analysis
# ---------------------------------------------------------------------------


def fig4_beta_stats(refined, n, rng):
    """Figure 4: β-statistic per node for each macro size, plus expected modules."""
    print("\nBuilding Monte-Carlo haxa null distribution …")
    haxa_stats = ssdi.make_haxa_stats(nmax=n, N=HAXA_N, rng=rng)

    m_dims = sorted(refined)
    n_m = len(m_dims)
    expected = {
        2: ("2-module", [0, 1], MOD_2),
        3: ("isolated 3-module", [6, 7, 8], MOD_3),
        5: ("2+3-module union", [0, 1, 6, 7, 8], "#7B68A8"),
        6: ("2+4-module union", [0, 1, 2, 3, 4, 5], "#A65628"),
    }

    fig, axes = plt.subplots(
        n_m,
        1,
        figsize=(7.0, 1.1 * n_m + 0.8),
        sharex=True,
    )
    axes = np.atleast_1d(axes)
    for idx, m in enumerate(m_dims):
        L_min = refined[m]["Lopto"][:, :, 0]
        beta = ssdi.habeta(L_min)
        res = ssdi.habeta_statinf(beta, n=n, m=m, slevel=0.05, tails="right", mhtc=True)
        sig = res.sig if res.sig.ndim == 1 else res.sig[:, 1]
        # beta = cos(theta)^2, so the upper 95% beta threshold is the
        # lower 5% principal-angle critical value.
        cval_mc = ssdi.get_haxa_cvals(n=n, stats=haxa_stats, mdim=[m], slev=[0.05])[0, 0]
        beta_thresh_mc = float(np.cos(cval_mc) ** 2)
        cv = res.cval if np.isscalar(res.cval) else float(np.atleast_1d(res.cval)[-1])

        ax = axes[idx]
        # Module colouring: for "expected" nodes use module colour, otherwise grey
        if m in expected:
            label, exp_nodes, exp_col = expected[m]
            bar_colours = [exp_col if i in exp_nodes else "0.78" for i in range(n)]
            # also highlight where significance test agrees
            for i in range(n):
                if sig[i] and i not in exp_nodes:
                    bar_colours[i] = "0.55"
        else:
            bar_colours = [WONG[1] if s else "0.78" for s in sig]

        ax.bar(np.arange(n), beta, color=bar_colours, edgecolor="black", lw=0.4)
        ax.axhline(cv, color="black", ls="--", lw=0.7)
        ax.axhline(beta_thresh_mc, color=WONG[6], ls=":", lw=0.7)
        ax.set_ylabel(f"m = {m}", rotation=0, ha="right", va="center", labelpad=15)
        ax.set_ylim(0, 1.08)
        ax.set_yticks([0, 0.5, 1.0])

        if m in expected:
            ax.text(
                0.985,
                0.78,
                expected[m][0],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color=expected[m][2],
                fontweight="bold",
                bbox=dict(
                    boxstyle="round,pad=0.15", facecolor="white", edgecolor="none", alpha=0.9
                ),
            )
    axes[-1].set_xlabel("Node index")
    axes[-1].set_xticks(np.arange(n))
    # leave horizontal headroom so the annotation never clips
    for ax in axes:
        ax.set_xlim(-0.7, n - 0.3)

    # Single legend at the bottom
    handles = [
        plt.Line2D([0], [0], color="black", ls="--", lw=0.7, label="Beta-marginal Bonf 5%"),
        plt.Line2D([0], [0], color=WONG[6], ls=":", lw=0.7, label="Monte-Carlo haxa 95%"),
        plt.Rectangle((0, 0), 1, 1, color="0.78", label="non-significant"),
        plt.Rectangle((0, 0), 1, 1, color=WONG[1], label="significant"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.03),
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle(
        "β-statistic per node — minimum-DD projection for each macro size",
        y=1.06,
    )
    fig.tight_layout()
    _save(fig, "fig04_beta_stats")


# ---------------------------------------------------------------------------
# 5. Headline summary figure
# ---------------------------------------------------------------------------


def fig5_summary(refined):
    """A single Nature-style summary: best DD vs m, with theoretical predictions."""
    m_dims = sorted(refined)
    best_dds = np.array([refined[m]["dds"][0] for m in m_dims])

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    # Highlight the predicted closed sub-systems
    closed_subsystems = {2: "2-mod", 3: "3-mod", 5: "2+3", 6: "2+4"}
    ax.semilogy(
        m_dims,
        np.maximum(best_dds, 1e-12),
        "o-",
        color=WONG[5],
        ms=8,
        lw=1.5,
        mew=0.8,
        mec="black",
        zorder=3,
    )
    # Annotate closed sub-systems (place labels above the points, in white space)
    for m, label in closed_subsystems.items():
        if m in m_dims:
            d = max(best_dds[m_dims.index(m)], 1e-12)
            ax.annotate(
                label,
                xy=(m, d),
                xytext=(m, d * 30.0),
                ha="center",
                fontsize=8,
                color=WONG[3],
                fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=WONG[3], lw=0.6),
            )
    ax.set_xlabel("Macro dimension m")
    ax.set_ylabel("Best spectral DD")
    ax.set_title("Emergent macros recovered by SSDI")
    ax.set_xticks(m_dims)
    ax.set_ylim(1e-13, 5.0)
    ax.grid(True, which="major", axis="y", linestyle=":", linewidth=0.4, alpha=0.4)
    _save(fig, "fig05_summary")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _preopt_signature() -> str:
    """Return the numerical configuration represented by a preopt cache."""
    return repr(
        (
            MODE,
            SEED,
            VAR_ORDER,
            RHO,
            RMII,
            W_DECAY,
            NRUNSP,
            NITERSP,
            GDSIG0P,
            GDLSP,
            GDTOLP,
        )
    )


def _refined_signature(fres_use: int) -> str:
    """Return the numerical configuration represented by a refinement cache."""
    return repr(
        (
            _preopt_signature(),
            NITERSO,
            GDSIG0O,
            GDLSO,
            GDTOLO,
            CTOL,
            int(fres_use),
        )
    )


def _cache_error(path: Path, stage: str, detail: str) -> RuntimeError:
    return RuntimeError(f"Cache {path} is stale or incompatible ({detail}); rerun STAGE={stage}")


def _validate_cache(z, path: Path, kind: str, signature: str, stage: str) -> None:
    expected = {
        "cache_version": CACHE_VERSION,
        "cache_kind": kind,
        "config_signature": signature,
    }
    for key, value in expected.items():
        if key not in z.files:
            raise _cache_error(path, stage, f"missing {key}")
        if z[key].item() != value:
            raise _cache_error(path, stage, f"{key} does not match")


def _pack_histories(histories) -> np.ndarray:
    """Pad variable-length three-column optimiser histories for NPZ storage."""
    hist_max = max((h.shape[0] for h in histories if h is not None), default=0)
    packed = np.full((len(histories), hist_max, 3), np.nan)
    for run, history in enumerate(histories):
        if history is not None:
            if history.ndim != 2 or history.shape[1] != 3:
                raise ValueError("optimiser histories must have shape (iterations, 3)")
            packed[run, : history.shape[0]] = history
    return packed


def _unpack_histories(packed: np.ndarray) -> list[np.ndarray | None]:
    if packed.ndim != 3 or packed.shape[2] != 3:
        raise ValueError("cached optimiser histories must have shape (runs, iterations, 3)")
    histories = []
    for run in range(packed.shape[0]):
        valid = ~np.isnan(packed[run, :, 0])
        histories.append(packed[run, valid].copy() if valid.any() else None)
    return histories


def _validate_m_dims(m_dims: list[int], n: int, path: Path, stage: str) -> None:
    if m_dims != list(range(1, n)):
        raise _cache_error(path, stage, "macro dimensions do not match the model")


def _save_preopt(preopt, ARA, V0):
    """Serialise preopt results so a later stage can resume from disk."""
    data = {
        "cache_version": np.array(CACHE_VERSION),
        "cache_kind": np.array("preopt"),
        "config_signature": np.array(_preopt_signature()),
        "ARA": ARA,
        "V0": V0,
    }
    for m, result in preopt.items():
        data[f"dds_{m}"] = result["dds"]
        data[f"Lp_{m}"] = result["Lp"]
        data[f"Loptp_{m}"] = result["Loptp"]
        data[f"goptp_{m}"] = result["goptp"]
        data[f"conv_{m}"] = np.asarray(result["conv"], dtype=int)
        data[f"hist_{m}"] = _pack_histories(result["hist"])
        data[f"cpu_{m}"] = np.array([result["cpu"]])
    np.savez_compressed(PREOPT_CACHE, m_dims=np.array(sorted(preopt)), **data)


def _load_preopt():
    """Recover preopt results from an earlier stage. Returns (preopt, ARA, V0)."""
    with np.load(PREOPT_CACHE, allow_pickle=False) as z:
        _validate_cache(z, PREOPT_CACHE, "preopt", _preopt_signature(), "preopt")
        ARA = z["ARA"].copy()
        V0 = z["V0"].copy()
        if V0.ndim != 2 or V0.shape[0] != V0.shape[1] or ARA.ndim != 3 or ARA.shape[:2] != V0.shape:
            raise _cache_error(PREOPT_CACHE, "preopt", "invalid model arrays")
        m_dims = [int(m) for m in z["m_dims"].tolist()]
        _validate_m_dims(m_dims, V0.shape[0], PREOPT_CACHE, "preopt")
        preopt = {}
        for m in m_dims:
            dds = z[f"dds_{m}"].copy()
            Lp = z[f"Lp_{m}"].copy()
            Loptp = z[f"Loptp_{m}"].copy()
            goptp = z[f"goptp_{m}"].copy()
            conv = z[f"conv_{m}"].astype(int).tolist()
            hist = _unpack_histories(z[f"hist_{m}"])
            runs = Lp.shape[2] if Lp.ndim == 3 else 0
            valid = (
                Lp.shape[:2] == (V0.shape[0], m)
                and Loptp.shape == Lp.shape
                and dds.shape == (runs,)
                and goptp.shape == (runs, runs)
                and len(conv) == runs
                and len(hist) == runs
                and runs == NRUNSP
            )
            if not valid:
                raise _cache_error(PREOPT_CACHE, "preopt", f"invalid arrays for m={m}")
            preopt[m] = dict(
                dds=dds,
                Lp=Lp,
                Loptp=Loptp,
                goptp=goptp,
                conv=conv,
                hist=hist,
                cpu=float(z[f"cpu_{m}"][0]),
            )
    return preopt, ARA, V0


def _save_refined(refined, fres_use: int):
    data = {
        "cache_version": np.array(CACHE_VERSION),
        "cache_kind": np.array("refined"),
        "config_signature": np.array(_refined_signature(fres_use)),
    }
    for m, result in refined.items():
        data[f"dds_{m}"] = result["dds"]
        data[f"Lo_{m}"] = result["Lo"]
        data[f"Lopto_{m}"] = result["Lopto"]
        data[f"gopto_{m}"] = result["gopto"]
        data[f"conv_{m}"] = np.asarray(result["conv"], dtype=int)
        data[f"clust_uidx_{m}"] = result["clust"].uidx
        data[f"clust_usiz_{m}"] = result["clust"].usiz
        data[f"clust_n_{m}"] = np.array([result["clust"].nruns])
        data[f"hist_{m}"] = _pack_histories(result["hist"])
        data[f"cpu_{m}"] = np.array([result["cpu"]])
    np.savez_compressed(REFINED_CACHE, m_dims=np.array(sorted(refined)), **data)


def _load_refined(n: int, fres_use: int):
    from complexbox.ssdi.cluster import ClusterResult

    with np.load(REFINED_CACHE, allow_pickle=False) as z:
        _validate_cache(
            z,
            REFINED_CACHE,
            "refined",
            _refined_signature(fres_use),
            "opt",
        )
        m_dims = [int(m) for m in z["m_dims"].tolist()]
        _validate_m_dims(m_dims, n, REFINED_CACHE, "opt")
        refined = {}
        for m in m_dims:
            dds = z[f"dds_{m}"].copy()
            Lo = z[f"Lo_{m}"].copy()
            Lopto = z[f"Lopto_{m}"].copy()
            gopto = z[f"gopto_{m}"].copy()
            conv = z[f"conv_{m}"].astype(int).tolist()
            hist = _unpack_histories(z[f"hist_{m}"])
            uidx = z[f"clust_uidx_{m}"].astype(int)
            usiz = z[f"clust_usiz_{m}"].astype(int)
            nruns = int(z[f"clust_n_{m}"][0])
            runs = Lo.shape[2] if Lo.ndim == 3 else 0
            valid = (
                Lo.shape[:2] == (n, m)
                and Lopto.shape == Lo.shape
                and dds.shape == (runs,)
                and gopto.shape == (runs, runs)
                and len(conv) == runs
                and len(hist) == runs
                and uidx.shape == (runs,)
                and usiz.shape == (runs,)
                and nruns == runs
            )
            if not valid:
                raise _cache_error(REFINED_CACHE, "opt", f"invalid arrays for m={m}")
            refined[m] = dict(
                dds=dds,
                Lo=Lo,
                Lopto=Lopto,
                gopto=gopto,
                conv=conv,
                hist=hist,
                clust=ClusterResult(uidx=uidx, usiz=usiz, nruns=nruns),
                cpu=float(z[f"cpu_{m}"][0]),
            )
    return refined


def main():
    rng = np.random.default_rng(SEED)
    stats_rng = np.random.default_rng(STATS_SEED)
    print(
        f"Mode={MODE}  STAGE={STAGE}  NRUNSP={NRUNSP}  NITERSP={NITERSP}  "
        f"NITERSO={NITERSO}  HAXA_N={HAXA_N}"
    )

    # 1. sim_model — always cheap, re-run every stage
    mdescript, CON, n, A0, C0, K0, V0, ARA0, gc, fres = sim_model(rng)
    if STAGE in ("all", "model", "preopt", "figs-preopt"):
        fig1_model(CON, gc, mdescript)
    if STAGE == "model":
        print(f"\nStage 'model' done. Figures saved to {OUT}/.")
        return

    m_dims = list(range(1, n))

    # Figure-only spectral stage depends only on its validated refinement cache.
    if STAGE == "figs-opt":
        if not REFINED_CACHE.exists():
            raise RuntimeError(f"Need to run STAGE=opt first ({REFINED_CACHE} missing)")
        refined = _load_refined(n, fres)
        fig3_opt(refined)
        return

    # 2. preoptimise_dd
    if STAGE in ("all", "preopt"):
        print("\nPre-optimisation …")
        preopt, ARA, V_d = preoptimise_dd(ARA0, V0, m_dims, rng)
        _save_preopt(preopt, ARA, V0)
        fig2_preopt(preopt)
        if STAGE == "preopt":
            print(f"\nStage 'preopt' done. Cached to {PREOPT_CACHE.name}.")
            return
    else:
        if not PREOPT_CACHE.exists():
            raise RuntimeError(f"Need to run STAGE=preopt first ({PREOPT_CACHE} missing)")
        preopt, ARA, V0 = _load_preopt()
        if STAGE == "figs-preopt":
            fig2_preopt(preopt)
            return

    # 3. optimise_dd
    if STAGE in ("all", "opt"):
        print("\nSpectral refinement …")
        refined = optimise_dd(preopt, ARA, V0, fres)
        _save_refined(refined, fres)
        fig3_opt(refined)
        if STAGE == "opt":
            print(f"\nStage 'opt' done. Cached to {REFINED_CACHE.name}.")
            return
    else:
        if not REFINED_CACHE.exists():
            raise RuntimeError(f"Need to run STAGE=opt first ({REFINED_CACHE} missing)")
        refined = _load_refined(n, fres)

    # 4. β-stat analysis + 5. summary (always cheap given cached refined results)
    if STAGE in ("all", "figs", "stats"):
        fig4_beta_stats(refined, n, stats_rng)
        fig5_summary(refined)

    # Numeric results summary
    np.savez(
        OUT / f"results_{MODE}.npz",
        m_dims=m_dims,
        best_dd_preopt=np.array([preopt[m]["dds"][0] for m in m_dims]),
        best_dd_refined=np.array([refined[m]["dds"][0] for m in m_dims]),
        n_clusters=np.array([refined[m]["clust"].nruns for m in m_dims]),
        cpu_preopt=np.array([preopt[m]["cpu"] for m in m_dims]),
        cpu_refined=np.array([refined[m]["cpu"] for m in m_dims]),
    )
    print(f"\nAll figures + results saved to {OUT}/")
    return preopt, refined


if __name__ == "__main__":
    main()
