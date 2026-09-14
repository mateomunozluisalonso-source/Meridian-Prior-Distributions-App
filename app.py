"""Visualizador interactivo de distribuciones a priori de Meridian (2.0.0).

Aplicación Streamlit que muestra cómo cambian las distribuciones a priori
al variar sus parámetros. El orden de los parámetros sigue exactamente el
orden de los argumentos de TensorFlow Probability tal como los usa Meridian.
"""

from io import BytesIO
import zipfile

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import (
    beta as beta_dist,
    gamma as gamma_dist,
    halfnorm,
    lognorm,
    norm,
    truncnorm,
    uniform,
)

st.set_page_config(page_title="Priors de Meridian", layout="wide")

# ---------------------------------------------------------------------------
# Colores por distribución
# ---------------------------------------------------------------------------
COLORS = {
    "normal": "#1f77b4",      # Azul
    "lognormal": "#ff7f0e",   # Naranja
    "halfnormal": "#d62728",  # Rojo
    "truncnormal": "#17becf", # Cyan
    "beta": "#2ca02c",        # Verde
    "gamma": "#d4a017",       # Dorado/Amarillo
    "uniform": "#9467bd",     # Morado
}

# ---------------------------------------------------------------------------
# Definición de distribuciones: orden de parámetros idéntico al de Meridian
# ---------------------------------------------------------------------------
DISTRIBUTIONS = [
    {
        "key": "lognormal",
        "name": "Distribución Log-Normal",
        "meridian": "LogNormal",
        "color": COLORS["lognormal"],
        "params": [
            {"key": "loc", "label": "loc — media de log(X)", "min": -3.0, "max": 3.0, "step": 0.1, "default": 0.2},
            {"key": "scale", "label": "scale — desv. estándar de log(X)", "min": 0.05, "max": 2.0, "step": 0.05, "default": 0.9},
        ],
    },
    {
        "key": "normal",
        "name": "Distribución Normal",
        "meridian": "Normal",
        "color": COLORS["normal"],
        "params": [
            {"key": "loc", "label": "loc — media", "min": -10.0, "max": 10.0, "step": 0.1, "default": 0.0},
            {"key": "scale", "label": "scale — desv. estándar", "min": 0.1, "max": 10.0, "step": 0.1, "default": 1.0},
        ],
    },
    {
        "key": "halfnormal",
        "name": "HalfNormal",
        "meridian": "HalfNormal",
        "color": COLORS["halfnormal"],
        "params": [
            {"key": "scale", "label": "scale", "min": 0.1, "max": 20.0, "step": 0.1, "default": 5.0},
        ],
    },
    {
        "key": "truncnormal",
        "name": "Truncated Normal",
        "meridian": "TruncatedNormal",
        "color": COLORS["truncnormal"],
        "params": [
            {"key": "loc", "label": "loc — media", "min": -10.0, "max": 10.0, "step": 0.1, "default": 0.8},
            {"key": "scale", "label": "scale — desv. estándar", "min": 0.1, "max": 10.0, "step": 0.1, "default": 0.8},
            {"key": "low", "label": "low — límite inferior", "min": -10.0, "max": 10.0, "step": 0.1, "default": 0.1},
            {"key": "high", "label": "high — límite superior", "min": -10.0, "max": 10.0, "step": 0.1, "default": 10.0},
        ],
    },
    {
        "key": "beta",
        "name": "Distribución Beta",
        "meridian": "Beta",
        "color": COLORS["beta"],
        "params": [
            {"key": "concentration1", "label": "concentration1 — α", "min": 0.1, "max": 100.0, "step": 0.1, "default": 1.0},
            {"key": "concentration0", "label": "concentration0 — β", "min": 0.1, "max": 100.0, "step": 0.1, "default": 99.0},
        ],
    },
    {
        "key": "gamma",
        "name": "Distribución Gamma",
        "meridian": "Gamma",
        "color": COLORS["gamma"],
        "params": [
            {"key": "concentration", "label": "concentration — forma (α)", "min": 0.1, "max": 20.0, "step": 0.1, "default": 2.0},
            {"key": "rate", "label": "rate", "min": 0.1, "max": 10.0, "step": 0.1, "default": 1.0},
        ],
    },
    {
        "key": "uniform",
        "name": "Distribución Uniforme",
        "meridian": "Uniform",
        "color": COLORS["uniform"],
        "params": [
            {"key": "low", "label": "low — límite inferior", "min": -5.0, "max": 5.0, "step": 0.1, "default": 0.0},
            {"key": "high", "label": "high — límite superior", "min": -5.0, "max": 5.0, "step": 0.1, "default": 1.0},
        ],
    },
]


def compute_density(key, p):
    """Devuelve (x, y) con la densidad de la distribución según sus parámetros.

    La conversión a scipy respeta la parametrización de Meridian / TFP.
    """
    if key == "lognormal":
        loc, scale = p["loc"], p["scale"]
        x = np.linspace(1e-6, lognorm.ppf(0.999, s=scale, scale=np.exp(loc)), 500)
        y = lognorm.pdf(x, s=scale, scale=np.exp(loc))
    elif key == "normal":
        loc, scale = p["loc"], p["scale"]
        x = np.linspace(loc - 4 * scale, loc + 4 * scale, 500)
        y = norm.pdf(x, loc=loc, scale=scale)
    elif key == "halfnormal":
        scale = p["scale"]
        x = np.linspace(0, halfnorm.ppf(0.999, scale=scale), 500)
        y = halfnorm.pdf(x, scale=scale)
    elif key == "truncnormal":
        loc, scale, low, high = p["loc"], p["scale"], p["low"], p["high"]
        if low >= high:
            raise ValueError("low debe ser menor que high")
        a, b = (low - loc) / scale, (high - loc) / scale
        x = np.linspace(low, high, 500)
        y = truncnorm.pdf(x, a, b, loc=loc, scale=scale)
    elif key == "beta":
        a, b = p["concentration1"], p["concentration0"]
        x = np.linspace(0, 1, 500)
        y = beta_dist.pdf(x, a, b)
    elif key == "gamma":
        concentration, rate = p["concentration"], p["rate"]
        x = np.linspace(0, gamma_dist.ppf(0.999, a=concentration, scale=1.0 / rate), 500)
        y = gamma_dist.pdf(x, a=concentration, scale=1.0 / rate)
    elif key == "uniform":
        low, high = p["low"], p["high"]
        if low >= high:
            raise ValueError("low debe ser menor que high")
        x = np.linspace(low, high, 500)
        y = uniform.pdf(x, loc=low, scale=high - low)
    else:
        raise ValueError(f"Distribución desconocida: {key}")
    return x, y


def distribution_statistics(key, p):
    """Calcula estadísticos analíticos y el intervalo central del 95 %."""
    if key == "lognormal":
        distribution = lognorm(s=p["scale"], scale=np.exp(p["loc"]))
        mode = np.exp(p["loc"] - p["scale"] ** 2)
    elif key == "normal":
        distribution = norm(loc=p["loc"], scale=p["scale"])
        mode = p["loc"]
    elif key == "halfnormal":
        distribution = halfnorm(scale=p["scale"])
        mode = 0.0
    elif key == "truncnormal":
        a = (p["low"] - p["loc"]) / p["scale"]
        b = (p["high"] - p["loc"]) / p["scale"]
        distribution = truncnorm(a, b, loc=p["loc"], scale=p["scale"])
        mode = min(max(p["loc"], p["low"]), p["high"])
    elif key == "beta":
        a, b = p["concentration1"], p["concentration0"]
        distribution = beta_dist(a, b)
        if a > 1 and b > 1:
            mode = (a - 1) / (a + b - 2)
        elif a == 1 and b == 1:
            mode = 0.5
        elif a <= 1 and b <= 1:
            mode = "0 y 1"
        elif a <= 1:
            mode = 0.0
        else:
            mode = 1.0
    elif key == "gamma":
        distribution = gamma_dist(a=p["concentration"], scale=1.0 / p["rate"])
        mode = max((p["concentration"] - 1) / p["rate"], 0.0)
    elif key == "uniform":
        distribution = uniform(loc=p["low"], scale=p["high"] - p["low"])
        mode = (p["low"] + p["high"]) / 2
    else:
        raise ValueError(f"Distribución desconocida: {key}")

    return {
        "Moda": mode,
        "Mediana": distribution.ppf(0.5),
        "Media": distribution.mean(),
        "Desv. Estándar": distribution.std(),
        "IC 95%": [distribution.ppf(0.025), distribution.ppf(0.975)],
    }


def format_statistic(value):
    if isinstance(value, str):
        return value
    return f"{value:.2f}"


def format_interval(interval):
    return f"[{interval[0]:.2f}, {interval[1]:.2f}]"


def build_statistics_table(active_data):
    rows = []
    for dist, x, y, p in active_data:
        stats = distribution_statistics(dist["key"], p)
        rows.append(
            {
                "Distribución": dist["name"],
                "Moda": format_statistic(stats["Moda"]),
                "Mediana": format_statistic(stats["Mediana"]),
                "Media": format_statistic(stats["Media"]),
                "Desv. Estándar": format_statistic(stats["Desv. Estándar"]),
                "IC 95%": format_interval(stats["IC 95%"]),
            }
        )
    return pd.DataFrame(rows)


def plot_single(ax, dist, x, y, show_stats=False, params=None):
    """Dibuja una distribución con borde definido e interior semitransparente."""
    color = dist["color"]
    ax.plot(x, y, color=color, lw=2.5)
    ax.fill_between(x, y, color=color, alpha=0.2)
    ax.set_title(dist["name"], fontsize=11, color=color)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    ax.grid(True, linestyle="--", alpha=0.5)
    if show_stats:
        stats = distribution_statistics(dist["key"], params)
        stats_text = (
            f"Moda: {format_statistic(stats['Moda'])}\n"
            f"Mediana: {format_statistic(stats['Mediana'])}\n"
            f"Media: {format_statistic(stats['Media'])}\n"
            f"DE: {format_statistic(stats['Desv. Estándar'])}\n"
            f"IC 95%: {format_interval(stats['IC 95%'])}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "alpha": 0.82},
        )


def plot_grid(active_data):
    n = len(active_data)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.4 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis("off")
    for (dist, x, y, p), ax in zip(active_data, axes[:n]):
        plot_single(ax, dist, x, y, show_stats=True, params=p)
    fig.tight_layout()
    return fig


def plot_overlay(active_data):
    fig, ax = plt.subplots(figsize=(11, 6))
    for dist, x, y, p in active_data:
        ax.plot(x, y, color=dist["color"], lw=2.5, label=dist["name"])
        ax.fill_between(x, y, color=dist["color"], alpha=0.2)
    ax.set_xlabel("Valor")
    ax.set_ylabel("Densidad")
    ax.set_title("Distribuciones a priori superpuestas")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    return fig


def build_snippet(active_data):
    """Genera el código Meridian/TFP equivalente con los valores actuales."""
    lines = [
        "import tensorflow_probability as tfp",
        "tfd = tfp.distributions",
        "",
        "# Los valores siguientes reproducen las distribuciones graficadas.",
        "priors = {",
    ]
    for dist, x, y, p in active_data:
        args = ", ".join(f"{k}={v:.3g}" for k, v in p.items())
        lines.append(f'    "{dist["name"]}": tfd.{dist["meridian"]}({args}),')
    lines.append("}")
    return "\n".join(lines)


def build_summary(active_data):
    """Construye una fila por distribución con sus parámetros ordenados."""
    rows = []
    for dist, x, y, p in active_data:
        ordered_params = ", ".join(f"{key}={value:g}" for key, value in p.items())
        rows.append(
            {
                "Distribución": dist["name"],
                "Constructor Meridian": dist["meridian"],
                "Parámetros (orden Meridian)": ordered_params,
            }
        )
    return pd.DataFrame(rows)


def figure_as_png(fig):
    """Serializa una figura matplotlib como PNG y libera sus recursos."""
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return buffer.getvalue()


def build_pdf_report(all_data, summary_df, statistics_df):
    """Crea un PDF con estadísticas en la primera página y configuración en la segunda."""
    pdf_buffer = BytesIO()
    with PdfPages(pdf_buffer) as pdf:
        report_fig = plt.figure(figsize=(11, 8.5))
        grid = report_fig.add_gridspec(
            2,
            1,
            height_ratios=[1.8, 1.2],
            hspace=0.35,
        )

        plot_ax = report_fig.add_subplot(grid[0])
        for dist, x, y, p in all_data:
            plot_ax.plot(x, y, color=dist["color"], lw=2.2, label=dist["name"])
            plot_ax.fill_between(x, y, color=dist["color"], alpha=0.16)
        plot_ax.set_title("Distribuciones a priori superpuestas", fontweight="bold")
        plot_ax.set_xlabel("Valor")
        plot_ax.set_ylabel("Densidad")
        plot_ax.grid(True, linestyle="--", alpha=0.5)
        plot_ax.legend(fontsize=8, ncol=2)

        table_ax = report_fig.add_subplot(grid[1])
        table_ax.axis("off")
        table_ax.set_title("Estadísticos de las distribuciones", fontsize=13, pad=12)
        table = table_ax.table(
            cellText=statistics_df.astype(str).values,
            colLabels=statistics_df.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.55)
        report_fig.tight_layout(rect=[0, 0, 1, 0.97])
        pdf.savefig(report_fig)
        plt.close(report_fig)

        config_fig, config_ax = plt.subplots(figsize=(11, 8.5))
        config_ax.axis("off")
        config_ax.set_title("Resumen de configuraciones elegidas", fontsize=14, pad=20)
        config_table = config_ax.table(
            cellText=summary_df.astype(str).values,
            colLabels=summary_df.columns,
            cellLoc="left",
            colLoc="left",
            loc="center",
        )
        config_table.auto_set_font_size(False)
        config_table.set_fontsize(9)
        config_table.scale(1, 2)
        config_fig.tight_layout()
        pdf.savefig(config_fig)
        plt.close(config_fig)

    return pdf_buffer.getvalue()


def build_download_zip(all_data, summary_df):
    """Empaqueta las imágenes actuales y el resumen en un ZIP descargable."""
    statistics_df = build_statistics_table(all_data)
    archive_buffer = BytesIO()
    with zipfile.ZipFile(archive_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("tabla_resumen_priors.csv", summary_df.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("tabla_estadisticos_priors.csv", statistics_df.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr(
            "reporte_priors_meridian.pdf",
            build_pdf_report(all_data, summary_df, statistics_df),
        )

        gallery_fig = plot_grid(all_data)
        archive.writestr("galeria_distribuciones.png", figure_as_png(gallery_fig))

        for index, (dist, x, y, p) in enumerate(all_data, start=1):
            individual_fig, individual_ax = plt.subplots(figsize=(7, 4.5))
            plot_single(individual_ax, dist, x, y, show_stats=True, params=p)
            filename = f"{index:02d}_{dist['key']}.png"
            archive.writestr(filename, figure_as_png(individual_fig))

    return archive_buffer.getvalue()


# ---------------------------------------------------------------------------
# Sidebar: widgets en el orden exacto de la lista de distribuciones
# ---------------------------------------------------------------------------
st.sidebar.title("Parámetros de los priors")

if st.sidebar.button("Restaurar defaults de Meridian"):
    for dist in DISTRIBUTIONS:
        st.session_state[f"show_{dist['key']}"] = True
        for spec in dist["params"]:
            st.session_state[f"{dist['key']}_{spec['key']}"] = spec["default"]
    st.rerun()

params = {}
show_flags = {}
for dist in DISTRIBUTIONS:
    with st.sidebar.expander(dist["name"]):
        show_flags[dist["key"]] = st.checkbox("Mostrar", key=f"show_{dist['key']}")
        p = {}
        for spec in dist["params"]:
            p[spec["key"]] = st.slider(
                spec["label"],
                min_value=spec["min"],
                max_value=spec["max"],
                value=spec["default"],
                step=spec["step"],
                key=f"{dist['key']}_{spec['key']}",
            )
        params[dist["key"]] = p

# ---------------------------------------------------------------------------
# Área principal
# ---------------------------------------------------------------------------
st.title("Priors de Meridian 2.0.0")
st.caption(
    "Orden de parámetros idéntico al de TensorFlow Probability / Meridian. "
    "Estilo basado en «Comparison of Common Prior Distributions»."
)

view = st.radio("Vista", ["Rejilla", "Superpuesto"], horizontal=True)

active_data = []
errors = []
for dist in DISTRIBUTIONS:
    try:
        x, y = compute_density(dist["key"], params[dist["key"]])
        data = (dist, x, y, params[dist["key"]])
        if show_flags[dist["key"]]:
            active_data.append(data)
    except ValueError as exc:
        errors.append(f"{dist['name']}: {exc}")

for msg in errors:
    st.warning(msg)

if not active_data:
    st.info("Activa al menos una distribución en la barra lateral para verla.")
else:
    with st.expander("Código Meridian equivalente"):
        st.code(build_snippet(active_data), language="python")

    if view == "Rejilla":
        fig = plot_grid(active_data)
    else:
        fig = plot_overlay(active_data)
    st.pyplot(fig)

    statistics_df = build_statistics_table(active_data)
    with st.expander("Ver estadísticos de las distribuciones", expanded=False):
        st.dataframe(statistics_df, use_container_width=True, hide_index=True)

    summary_df = build_summary(active_data)
    with st.expander("Ver resumen de configuraciones", expanded=False):
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    download_data = build_download_zip(active_data, summary_df)
    st.download_button(
        label="Descargar imágenes y tabla resumen",
        data=download_data,
        file_name="priors_meridian_resultados.zip",
        mime="application/zip",
        help="Descarga las imágenes PNG de las distribuciones y el resumen en CSV.",
    )
