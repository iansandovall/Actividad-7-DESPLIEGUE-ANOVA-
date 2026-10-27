from pathlib import Path
import io

import numpy as np
import pandas as pd
import plotly.express as px
import scipy.stats as stats
import streamlit as st
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(page_title="ANOVA | CLV México", page_icon="📊", layout="wide")

DATA_FILE = Path(__file__).resolve().parent / "Ventas_CLV_2022_2026_procesadas.xlsx"
REQUIRED_COLUMNS = {
    "fecha_documento",
    "folio",
    "folio_abreviado",
    "total_unidades",
    "total_neto",
    "estado",
    "ciudad",
    "vendedor",
    "nombre_cliente",
}
FACTORS = {
    "estado": "Estado",
    "anio": "Año",
    "trimestre": "Trimestre",
    "mes": "Mes",
    "tipo_folio": "Tipo de folio",
    "vendedor": "Vendedor",
}
MEASURES = {
    "total_facturado": "Venta total por ticket ($)",
    "unidades": "Unidades por ticket",
}

st.markdown(
    """
    <style>
        .stApp { background: linear-gradient(135deg, #f7f9fc, #edf4f7); }
        [data-testid="stMetric"] {
            background: white; border: 1px solid #e3eaf0;
            padding: 14px; border-radius: 14px;
        }
        h1, h2, h3 { color: #12304a; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner="Leyendo la base de ventas...")
def load_excel(file_bytes):
    if file_bytes is None:
        if not DATA_FILE.exists():
            raise FileNotFoundError(f"No se encontró {DATA_FILE.name} junto a app.py.")
        return pd.read_excel(DATA_FILE)
    return pd.read_excel(io.BytesIO(file_bytes))


def prepare_data(source):
    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError("Faltan columnas requeridas: " + ", ".join(sorted(missing)))

    lines = source.copy()
    lines["fecha_documento"] = pd.to_datetime(
        lines["fecha_documento"], errors="coerce"
    )
    for column in ("total_neto", "total_unidades"):
        lines[column] = pd.to_numeric(lines[column], errors="coerce")
    lines = lines.dropna(
        subset=["folio", "fecha_documento", "total_neto", "total_unidades"]
    )
    for column in ("estado", "ciudad", "vendedor", "folio_abreviado"):
        lines[column] = (
            lines[column]
            .fillna("Sin dato")
            .astype(str)
            .str.strip()
            .replace("", "Sin dato")
        )

    tickets = lines.groupby("folio", as_index=False, sort=False).agg(
        total_facturado=("total_neto", "sum"),
        unidades=("total_unidades", "sum"),
        fecha=("fecha_documento", "first"),
        estado=("estado", "first"),
        ciudad=("ciudad", "first"),
        vendedor=("vendedor", "first"),
        tipo_folio=("folio_abreviado", "first"),
        nombre_cliente=("nombre_cliente", "first"),
    )
    tickets["anio"] = tickets["fecha"].dt.year.astype(str)
    tickets["trimestre"] = "T" + tickets["fecha"].dt.quarter.astype(str)
    tickets["mes"] = tickets["fecha"].dt.month.map(
        {
            1: "Enero",
            2: "Febrero",
            3: "Marzo",
            4: "Abril",
            5: "Mayo",
            6: "Junio",
            7: "Julio",
            8: "Agosto",
            9: "Septiembre",
            10: "Octubre",
            11: "Noviembre",
            12: "Diciembre",
        }
    )
    return lines, tickets


def make_response(data, measure, use_log):
    result = data.copy()
    values = pd.to_numeric(result[measure], errors="coerce")
    if use_log:
        if values.min() < 0:
            raise ValueError("log(1+x) requiere valores no negativos.")
        result["_y"] = np.log1p(values)
        label = f"log(1 + {MEASURES[measure]})"
    else:
        result["_y"] = values
        label = MEASURES[measure]
    return result.dropna(subset=["_y"]), label


def top_levels(data, factor, limit, minimum):
    counts = data[factor].value_counts()
    return counts[counts >= minimum].head(limit).index.tolist()


def effect_sizes(table):
    residual = table.loc["Residual", "sum_sq"] if "Residual" in table.index else np.nan
    return {
        term: (
            row["sum_sq"] / (row["sum_sq"] + residual)
            if pd.notna(residual) and row["sum_sq"] + residual > 0
            else np.nan
        )
        for term, row in table.iterrows()
        if term not in ("Residual", "Intercept")
    }


def format_p(value):
    if value == 0:
        return "< 1e-300"
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def show_anova_table(table):
    result = table.copy()
    result["Eta² parcial"] = pd.Series(effect_sizes(table))
    result["PR(>F)"] = result["PR(>F)"].map(format_p)
    result = result.rename(
        columns={
            "sum_sq": "Suma de cuadrados",
            "df": "gl",
            "PR(>F)": "p-valor",
        }
    )
    st.dataframe(result.round(5), width="stretch")


st.sidebar.title("🦷 CLV México")
st.sidebar.caption("Despliegue interactivo · ANOVA y ANOVA factorial")
upload = st.sidebar.file_uploader(
    "Cargar otra base Excel (opcional)", type=["xlsx"]
)

try:
    lines, tickets = prepare_data(
        load_excel(upload.getvalue() if upload else None)
    )
except (FileNotFoundError, ValueError, ImportError) as error:
    st.error(f"No fue posible cargar la base: {error}")
    st.stop()

st.title("📊 Análisis de ventas · CLV México")
st.markdown(
    "Pruebas ANOVA sobre **tickets completos**. El archivo contiene renglones "
    "de artículos; se suman por folio para que cada observación represente "
    "una venta."
)
st.caption(
    f"Base: **{upload.name if upload else DATA_FILE.name}** · "
    "Unidad de análisis: folio/ticket"
)

page = st.sidebar.radio(
    "Sección",
    ["Resumen de la base", "ANOVA de una vía", "ANOVA factorial"],
)
minimum = st.sidebar.number_input(
    "Mínimo de tickets por grupo/combinación",
    min_value=5,
    max_value=500,
    value=30,
    step=5,
)
level_limit = st.sidebar.slider(
    "Máximo de niveles por factor", min_value=4, max_value=20, value=10
)
alpha = st.sidebar.select_slider(
    "Significancia α", options=[0.01, 0.05, 0.10], value=0.05
)

if page == "Resumen de la base":
    metrics = st.columns(4)
    metrics[0].metric("Tickets", f"{len(tickets):,}")
    metrics[1].metric("Renglones", f"{len(lines):,}")
    metrics[2].metric(
        "Venta acumulada", f"${tickets['total_facturado'].sum():,.0f}"
    )
    metrics[3].metric("Estados", f"{tickets['estado'].nunique()}")

    monthly = (
        tickets.set_index("fecha")
        .resample("MS")["total_facturado"]
        .sum()
        .rename_axis("Mes")
        .reset_index()
    )
    states = (
        tickets.groupby("estado", as_index=False)["total_facturado"]
        .sum()
        .nlargest(10, "total_facturado")
    )
    left, right = st.columns([1.3, 1])
    monthly_fig = px.area(
        monthly,
        x="Mes",
        y="total_facturado",
        title="Ventas por mes",
        labels={"total_facturado": "Venta ($)"},
        color_discrete_sequence=["#168c8c"],
    )
    left.plotly_chart(monthly_fig, width="stretch")
    states_fig = px.bar(
        states.sort_values("total_facturado"),
        x="total_facturado",
        y="estado",
        orientation="h",
        title="10 estados con mayor venta",
        labels={"total_facturado": "Venta ($)", "estado": "Estado"},
        color="total_facturado",
        color_continuous_scale="Teal",
    )
    right.plotly_chart(states_fig, width="stretch")
    with st.expander("Vista previa de tickets"):
        st.dataframe(tickets.head(100), width="stretch", hide_index=True)
    st.caption(
        "Los atributos de estado, ciudad, vendedor y fecha son consistentes "
        "dentro de cada folio en la base revisada."
    )

elif page == "ANOVA de una vía":
    st.subheader("ANOVA de una vía")
    st.write(
        "Compara la media de una variable de venta entre los niveles de "
        "un factor categórico."
    )
    first, second, third = st.columns(3)
    measure = first.selectbox(
        "Variable respuesta",
        list(MEASURES),
        format_func=lambda value: MEASURES[value],
        key="one_measure",
    )
    factor = second.selectbox(
        "Factor",
        list(FACTORS),
        format_func=lambda value: FACTORS[value],
        key="one_factor",
    )
    use_log = third.checkbox("Usar log(1+x)", value=True, key="one_log")

    data, response_label = make_response(tickets, measure, use_log)
    levels = top_levels(data, factor, int(level_limit), int(minimum))
    data = data[data[factor].isin(levels)].copy()
    if len(levels) < 2:
        st.warning(
            "Se necesitan al menos dos grupos con el mínimo de tickets indicado. "
            "Ajusta el mínimo o cambia el factor."
        )
    else:
        data["_f"] = data[factor].astype(str)
        model = ols("_y ~ C(_f)", data=data).fit()
        table = anova_lm(model, typ=2)
        term = next(name for name in table.index if name != "Residual")
        f_value = table.loc[term, "F"]
        p_value = table.loc[term, "PR(>F)"]
        groups = [
            group["_y"].to_numpy()
            for _, group in data.groupby("_f", observed=True)
        ]
        levene_stat, levene_p = stats.levene(*groups, center="median")

        metrics = st.columns(4)
        metrics[0].metric("F", f"{f_value:.3f}")
        metrics[1].metric("p-valor", format_p(p_value))
        metrics[2].metric(
            "Eta² parcial", f"{effect_sizes(table).get(term, np.nan):.4f}"
        )
        metrics[3].metric("Tickets", f"{len(data):,}")
        if p_value < alpha:
            st.success(
                f"Diferencia significativa; se rechaza H₀. "
                f"p = {format_p(p_value)}, α = {alpha:.2f}."
            )
        else:
            st.info(
                f"No hay evidencia suficiente para rechazar H₀. "
                f"p = {format_p(p_value)}, α = {alpha:.2f}."
            )
        st.caption(
            f"Levene (mediana): estadístico = {levene_stat:.3f}, "
            f"p = {format_p(levene_p)}. Respuesta: {response_label}."
        )
        if levene_p < alpha:
            st.warning(
                "Levene sugiere varianzas desiguales; interpreta con cautela "
                "el ANOVA clásico."
            )
        st.markdown("**Tabla ANOVA (Tipo II)**")
        show_anova_table(table)

        left, right = st.columns(2)
        box_fig = px.box(
            data,
            x="_f",
            y="_y",
            color="_f",
            points="outliers",
            title="Distribución por grupo",
            labels={"_f": FACTORS[factor], "_y": response_label},
        )
        box_fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        left.plotly_chart(box_fig, width="stretch")

        summary = (
            data.groupby("_f", observed=True)["_y"]
            .agg(media="mean", sem="sem")
            .reset_index()
        )
        summary["error"] = 1.96 * summary["sem"].fillna(0)
        mean_fig = px.bar(
            summary,
            x="_f",
            y="media",
            error_y="error",
            color="media",
            color_continuous_scale="Teal",
            title="Media por grupo (IC aprox. 95%)",
            labels={"_f": FACTORS[factor], "media": response_label},
        )
        mean_fig.update_layout(showlegend=False, xaxis_tickangle=-30)
        right.plotly_chart(mean_fig, width="stretch")
        counts = data.groupby("_f", observed=True).size().rename("Tickets")
        st.dataframe(counts.to_frame(), width="stretch")

        if p_value < alpha and 2 < len(levels) <= 20:
            st.markdown("**Tukey HSD · comparaciones post hoc**")
            tukey = pairwise_tukeyhsd(
                endog=data["_y"],
                groups=data["_f"],
                alpha=float(alpha),
            )
            tukey_table = pd.DataFrame(
                tukey._results_table.data[1:],
                columns=tukey._results_table.data[0],
            )
            st.dataframe(tukey_table, width="stretch", hide_index=True)
        elif p_value < alpha and len(levels) > 20:
            st.info(
                "Tukey se omite por haber más de 20 niveles; selecciona un "
                "factor con menos categorías para comparar pares."
            )

else:
    st.subheader("ANOVA factorial de dos vías")
    st.write(
        "Evalúa los efectos principales de dos factores y si existe "
        "interacción entre ambos."
    )
    first, second, third, fourth = st.columns(4)
    measure = first.selectbox(
        "Variable respuesta",
        list(MEASURES),
        format_func=lambda value: MEASURES[value],
        key="multi_measure",
    )
    factor1 = second.selectbox(
        "Factor A",
        list(FACTORS),
        format_func=lambda value: FACTORS[value],
        index=0,
        key="factor_a",
    )
    remaining_factors = [name for name in FACTORS if name != factor1]
    default_factor = (
        remaining_factors.index("anio") if "anio" in remaining_factors else 0
    )
    factor2 = third.selectbox(
        "Factor B",
        remaining_factors,
        format_func=lambda value: FACTORS[value],
        index=default_factor,
        key="factor_b",
    )
    use_log = fourth.checkbox("Usar log(1+x)", value=True, key="multi_log")

    data, response_label = make_response(tickets, measure, use_log)
    levels_a = top_levels(data, factor1, int(level_limit), int(minimum))
    levels_b = top_levels(data, factor2, int(level_limit), int(minimum))
    data = data[
        data[factor1].isin(levels_a) & data[factor2].isin(levels_b)
    ].copy()
    data["_a"] = data[factor1].astype(str)
    data["_b"] = data[factor2].astype(str)
    cell_counts = data.groupby(["_a", "_b"], observed=True).size()
    valid_cells = cell_counts[cell_counts >= int(minimum)].index
    valid_pairs = set(valid_cells.tolist())
    data = data[
        [
            pair in valid_pairs
            for pair in zip(data["_a"], data["_b"])
        ]
    ].copy()

    complete_design = (
        not data.empty
        and data["_a"].nunique() >= 2
        and data["_b"].nunique() >= 2
        and len(valid_pairs) == data["_a"].nunique() * data["_b"].nunique()
    )
    if not complete_design:
        st.warning(
            "No hay un diseño factorial completo con el mínimo de tickets "
            "indicado. Reduce el mínimo o selecciona factores que tengan "
            "combinaciones suficientes (por ejemplo, Estado × Año)."
        )
    else:
        model = ols("_y ~ C(_a) * C(_b)", data=data).fit()
        table = anova_lm(model, typ=2)
        interaction = next(
            (name for name in table.index if ":" in name),
            None,
        )
        if interaction is None:
            st.warning(
                "No se pudo estimar la interacción; revisa las combinaciones "
                "de factores."
            )
        else:
            interaction_p = table.loc[interaction, "PR(>F)"]
            groups = [
                group["_y"].to_numpy()
                for _, group in data.groupby(["_a", "_b"], observed=True)
            ]
            levene_stat, levene_p = stats.levene(*groups, center="median")

            metrics = st.columns(4)
            metrics[0].metric(
                "F de interacción", f"{table.loc[interaction, 'F']:.3f}"
            )
            metrics[1].metric("p de interacción", format_p(interaction_p))
            metrics[2].metric("Combinaciones", f"{len(valid_pairs):,}")
            metrics[3].metric("Tickets", f"{len(data):,}")
            if interaction_p < alpha:
                st.success(
                    f"La interacción es significativa; se rechaza H₀. "
                    f"p = {format_p(interaction_p)}, α = {alpha:.2f}."
                )
            else:
                st.info(
                    f"No hay evidencia suficiente de interacción. "
                    f"p = {format_p(interaction_p)}, α = {alpha:.2f}."
                )
            st.caption(
                f"Respuesta: {response_label}. Levene entre combinaciones: "
                f"estadístico = {levene_stat:.3f}, p = {format_p(levene_p)}."
            )
            if levene_p < alpha:
                st.warning(
                    "Levene sugiere varianzas desiguales entre combinaciones; "
                    "interpreta con cautela."
                )
            st.markdown("**Tabla ANOVA factorial (Tipo II)**")
            show_anova_table(table)

            means = (
                data.groupby(["_a", "_b"], observed=True)["_y"]
                .agg(media="mean", sem="sem")
                .reset_index()
            )
            means["error"] = 1.96 * means["sem"].fillna(0)
            left, right = st.columns(2)
            interaction_fig = px.line(
                means,
                x="_a",
                y="media",
                color="_b",
                markers=True,
                error_y="error",
                title="Gráfica de interacción",
                labels={
                    "_a": FACTORS[factor1],
                    "_b": FACTORS[factor2],
                    "media": response_label,
                },
            )
            interaction_fig.update_layout(
                xaxis_tickangle=-25,
                legend_title=FACTORS[factor2],
            )
            left.plotly_chart(interaction_fig, width="stretch")

            heatmap_data = means.pivot(
                index="_a", columns="_b", values="media"
            )
            heatmap_fig = px.imshow(
                heatmap_data,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="Tealgrn",
                title="Heatmap de medias",
                labels={
                    "x": FACTORS[factor2],
                    "y": FACTORS[factor1],
                    "color": response_label,
                },
            )
            right.plotly_chart(heatmap_fig, width="stretch")
            with st.expander("Tickets por combinación"):
                st.dataframe(
                    cell_counts.rename("Tickets").unstack(fill_value=0),
                    width="stretch",
                )

st.divider()
st.caption(
    "p < α indica evidencia de diferencias entre medias. La interacción "
    "refleja que el efecto de un factor varía según el nivel del otro; "
    "no implica causalidad."
)
