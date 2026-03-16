import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Empleabilidad IT Argentina 2023",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3, .stMetric label {
    font-family: 'Syne', sans-serif !important;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #94a3b8 !important;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Main bg */
.main .block-container {
    padding: 2rem 3rem;
    max-width: 1400px;
}

/* Metric cards */
[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    color: #0f172a !important;
}
[data-testid="metric-container"] [data-testid="stMetricLabel"] {
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b !important;
}
[data-testid="metric-container"] [data-testid="stMetricDelta"] {
    font-size: 0.8rem !important;
}

/* Section headers */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: #0f172a;
    border-left: 4px solid #6366f1;
    padding-left: 0.75rem;
    margin: 2rem 0 1.5rem 0;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #e2e8f0;
    margin: 2rem 0;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero-banner h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    color: white !important;
}
.hero-banner p {
    color: #94a3b8;
    margin: 0.5rem 0 0 0;
    font-size: 1rem;
}
.hero-tag {
    display: inline-block;
    background: rgba(99,102,241,0.3);
    border: 1px solid rgba(99,102,241,0.5);
    color: #a5b4fc;
    border-radius: 99px;
    padding: 0.2rem 0.75rem;
    font-size: 0.75rem;
    margin-bottom: 1rem;
    font-family: 'DM Sans', sans-serif;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}
</style>
""", unsafe_allow_html=True)

# ─── LOAD & CLEAN DATA ──────────────────────────────────────────────────────────
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Normalizar pagos_en_dolares
    df["pagos_en_dolares"] = df["pagos_en_dolares"].fillna("No")
    df["pagos_en_dolares"].replace("Cobro parte del salario en dólares", "Sí", inplace=True)
    df["pagos_en_dolares"].replace("Cobro todo el salario en dólares", "Sí", inplace=True)
    df["pagos_en_dolares"].replace("Mi sueldo está dolarizado (pero cobro en moneda local)", "No", inplace=True)

    # Seleccionar columnas de interés
    cols = {
        "donde_estas_trabajando": "Lugar de trabajo",
        "ultimo_salario_mensual_o_retiro_neto_en_tu_moneda_local": "Ultimo salario neto",
        "pagos_en_dolares": "recibe_pagos_en_dolares",
        "tipo_de_contrato": "tipo_de_contrato_laboral",
        "sueldo_dolarizado": "sueldo_dolarizado",
        "trabajo_de": "Profesion",
        "seniority": "Senority",
        "me_id_extra": "Genero",
        "tengo_edad": "Edad",
    }
    df = df[list(cols.keys())].rename(columns=cols)

    # Normalizar género
    df["Genero"].replace({"Hombre Cis": "Hombre", "Mujer Cis": "Mujer",
                          "Varón Cis": "Hombre"}, inplace=True)

    # Salario numérico y sin outliers extremos
    df["Ultimo salario neto"] = pd.to_numeric(df["Ultimo salario neto"], errors="coerce")
    df = df.dropna(subset=["Ultimo salario neto"])
    q_low = df["Ultimo salario neto"].quantile(0.01)
    q_high = df["Ultimo salario neto"].quantile(0.99)
    df = df[(df["Ultimo salario neto"] >= q_low) & (df["Ultimo salario neto"] <= q_high)]

    return df

# ─── PLOTLY THEME ───────────────────────────────────────────────────────────────
PALETTE = ["#6366f1", "#f59e0b", "#10b981", "#ef4444", "#3b82f6",
           "#8b5cf6", "#ec4899", "#14b8a6", "#f97316", "#84cc16"]

def base_layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(family="Syne", size=16, color="#0f172a"), x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color="#475569"),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", zerolinecolor="#e2e8f0"),
    )
    return fig

# ─── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎛️ Filtros")
    st.markdown("---")

    csv_path = st.text_input(
        "Ruta del CSV",
        value="2023.1_Sysarmy_Encuesta de remuneracin salarial Argentina.csv",
        help="Nombre del archivo CSV en la misma carpeta que app.py",
    )

    try:
        df_full = load_data(csv_path)
        data_ok = True
    except FileNotFoundError:
        st.error("❌ Archivo no encontrado. Verificá la ruta.")
        data_ok = False
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar: {e}")
        data_ok = False
        st.stop()

    st.success(f"✅ {len(df_full):,} registros cargados")
    st.markdown("---")

    # Filtros
    seniority_opts = sorted(df_full["Senority"].dropna().unique())
    sel_seniority = st.multiselect("Seniority", seniority_opts, default=seniority_opts)

    genero_opts = sorted(df_full["Genero"].dropna().unique())
    sel_genero = st.multiselect("Género", genero_opts, default=genero_opts)

    contrato_opts = sorted(df_full["tipo_de_contrato_laboral"].dropna().unique())
    sel_contrato = st.multiselect("Tipo de contrato", contrato_opts, default=contrato_opts)

    salario_range = st.slider(
        "Salario neto (ARS)",
        int(df_full["Ultimo salario neto"].min()),
        int(df_full["Ultimo salario neto"].max()),
        (int(df_full["Ultimo salario neto"].min()), int(df_full["Ultimo salario neto"].max())),
        step=10_000,
        format="%d",
    )

    st.markdown("---")
    st.caption("Fuente: Sysarmy · Encuesta Salarial 2023")

# ─── APPLY FILTERS ──────────────────────────────────────────────────────────────
df = df_full.copy()
if sel_seniority:
    df = df[df["Senority"].isin(sel_seniority)]
if sel_genero:
    df = df[df["Genero"].isin(sel_genero)]
if sel_contrato:
    df = df[df["tipo_de_contrato_laboral"].isin(sel_contrato)]
df = df[(df["Ultimo salario neto"] >= salario_range[0]) & (df["Ultimo salario neto"] <= salario_range[1])]

# ─── HERO ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <div class="hero-tag">Sysarmy · Argentina · 2023</div>
    <h1>💼 Empleabilidad IT Argentina</h1>
    <p>Análisis de remuneraciones y condiciones laborales del sector tecnológico</p>
</div>
""", unsafe_allow_html=True)

# ─── KPIs ────────────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total encuestados", f"{len(df):,}")
c2.metric("Salario neto promedio", f"${df['Ultimo salario neto'].mean():,.0f}")
c3.metric("Salario mediano", f"${df['Ultimo salario neto'].median():,.0f}")
pct_dolar = (df["recibe_pagos_en_dolares"] == "Sí").mean() * 100
c4.metric("Cobran en USD", f"{pct_dolar:.1f}%")
profesiones = df["Profesion"].nunique()
c5.metric("Profesiones distintas", str(profesiones))

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# SECCIÓN 1 — SALARIOS POR PROFESIÓN Y SENIORITY
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Salarios por Profesión y Seniority</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([3, 2])

with col_a:
    top_n = st.slider("Top N profesiones", 5, 20, 10, key="topn_prof")
    top_prof = (
        df.groupby("Profesion")["Ultimo salario neto"]
        .median()
        .sort_values(ascending=False)
        .head(top_n)
        .sort_values()
    )
    fig1 = go.Figure(go.Bar(
        x=top_prof.values,
        y=top_prof.index,
        orientation="h",
        marker=dict(
            color=top_prof.values,
            colorscale=[[0, "#c7d2fe"], [1, "#4338ca"]],
            showscale=False,
        ),
        text=[f"${v:,.0f}" for v in top_prof.values],
        textposition="outside",
        textfont=dict(size=11),
    ))
    base_layout(fig1, f"Top {top_n} profesiones — salario neto mediano (ARS)", height=420)
    fig1.update_layout(xaxis_title="Salario neto mediano (ARS)", yaxis_title="")
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    sen_order = ["Junior", "Semi-Senior", "Senior"]
    sen_data = (
        df[df["Senority"].isin(sen_order)]
        .groupby("Senority")["Ultimo salario neto"]
        .median()
        .reindex(sen_order)
    )
    fig2 = go.Figure()
    for i, (sen, val) in enumerate(sen_data.items()):
        fig2.add_trace(go.Bar(
            x=[sen],
            y=[val],
            name=sen,
            marker_color=PALETTE[i],
            text=f"${val:,.0f}",
            textposition="outside",
        ))
    base_layout(fig2, "Salario mediano por Seniority", height=420)
    fig2.update_layout(showlegend=False, yaxis_title="Salario neto mediano (ARS)")
    st.plotly_chart(fig2, use_container_width=True)

# Box plot profesión × seniority
sen_prof = df[df["Senority"].isin(sen_order)].copy()
fig3 = px.box(
    sen_prof,
    x="Profesion",
    y="Ultimo salario neto",
    color="Senority",
    color_discrete_sequence=PALETTE[:3],
    category_orders={"Senority": sen_order},
    labels={"Ultimo salario neto": "Salario neto (ARS)", "Profesion": ""},
)
base_layout(fig3, "Distribución salarial por profesión y seniority", height=440)
fig3.update_xaxes(tickangle=-35)
st.plotly_chart(fig3, use_container_width=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# SECCIÓN 2 — DISTRIBUCIÓN POR GÉNERO
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🧑‍🤝‍🧑 Distribución por Género</div>', unsafe_allow_html=True)

col_g1, col_g2, col_g3 = st.columns(3)

with col_g1:
    gen_counts = df["Genero"].value_counts().reset_index()
    gen_counts.columns = ["Genero", "Cantidad"]
    fig_g1 = px.pie(
        gen_counts,
        values="Cantidad",
        names="Genero",
        color_discrete_sequence=PALETTE,
        hole=0.5,
    )
    base_layout(fig_g1, "Distribución por género", height=340)
    fig_g1.update_traces(textinfo="percent+label", textfont_size=12)
    st.plotly_chart(fig_g1, use_container_width=True)

with col_g2:
    gen_sal = (
        df.groupby("Genero")["Ultimo salario neto"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    fig_g2 = px.bar(
        gen_sal,
        x="Genero",
        y="Ultimo salario neto",
        color="Genero",
        color_discrete_sequence=PALETTE,
        text=gen_sal["Ultimo salario neto"].apply(lambda v: f"${v:,.0f}"),
        labels={"Ultimo salario neto": "Salario neto mediano (ARS)"},
    )
    base_layout(fig_g2, "Salario mediano por género", height=340)
    fig_g2.update_layout(showlegend=False)
    fig_g2.update_traces(textposition="outside")
    st.plotly_chart(fig_g2, use_container_width=True)

with col_g3:
    gen_sen = (
        df[df["Senority"].isin(sen_order)]
        .groupby(["Genero", "Senority"])
        .size()
        .reset_index(name="Cantidad")
    )
    fig_g3 = px.bar(
        gen_sen,
        x="Senority",
        y="Cantidad",
        color="Genero",
        barmode="group",
        color_discrete_sequence=PALETTE,
        category_orders={"Senority": sen_order},
        labels={"Cantidad": "Cantidad de personas"},
    )
    base_layout(fig_g3, "Seniority por género", height=340)
    st.plotly_chart(fig_g3, use_container_width=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# SECCIÓN 3 — SALARIOS POR REGIÓN / PROVINCIA
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🗺️ Salarios por Región / Provincia</div>', unsafe_allow_html=True)

col_r1, col_r2 = st.columns([2, 3])

with col_r1:
    top_prov_n = st.slider("Top N provincias", 5, 20, 10, key="topn_prov")
    prov_sal = (
        df.groupby("Lugar de trabajo")["Ultimo salario neto"]
        .agg(["mean", "median", "count"])
        .reset_index()
        .rename(columns={"mean": "Promedio", "median": "Mediana", "count": "Registros"})
        .sort_values("Mediana", ascending=False)
        .head(top_prov_n)
        .sort_values("Mediana")
    )
    fig_r1 = go.Figure()
    fig_r1.add_trace(go.Bar(
        y=prov_sal["Lugar de trabajo"],
        x=prov_sal["Mediana"],
        name="Mediana",
        orientation="h",
        marker_color="#6366f1",
        text=[f"${v:,.0f}" for v in prov_sal["Mediana"]],
        textposition="outside",
    ))
    fig_r1.add_trace(go.Scatter(
        y=prov_sal["Lugar de trabajo"],
        x=prov_sal["Promedio"],
        mode="markers",
        name="Promedio",
        marker=dict(color="#f59e0b", size=10, symbol="diamond"),
    ))
    base_layout(fig_r1, f"Top {top_prov_n} provincias — salario neto mediano", height=480)
    fig_r1.update_layout(xaxis_title="ARS", legend=dict(orientation="h", y=1.05))
    st.plotly_chart(fig_r1, use_container_width=True)

with col_r2:
    prov_all = (
        df.groupby("Lugar de trabajo")["Ultimo salario neto"]
        .agg(["median", "count"])
        .reset_index()
        .rename(columns={"median": "Salario mediano", "count": "N"})
        .query("N >= 5")
        .sort_values("Salario mediano", ascending=False)
    )
    fig_r2 = px.scatter(
        prov_all,
        x="N",
        y="Salario mediano",
        size="N",
        color="Salario mediano",
        color_continuous_scale="Viridis",
        hover_name="Lugar de trabajo",
        labels={"N": "Cantidad de encuestados", "Salario mediano": "Salario mediano (ARS)"},
        size_max=40,
    )
    base_layout(fig_r2, "Salario mediano vs. representatividad por provincia (mín. 5 casos)", height=480)
    fig_r2.update_coloraxes(showscale=False)
    st.plotly_chart(fig_r2, use_container_width=True)

    st.dataframe(
        prov_all.assign(
            **{"Salario mediano": prov_all["Salario mediano"].apply(lambda x: f"${x:,.0f}")}
        ).rename(columns={"N": "Encuestados"}),
        use_container_width=True,
        height=200,
    )

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# SECCIÓN 4 — DOLARIZACIÓN Y TIPO DE CONTRATO
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-header">💵 Dolarización y Tipo de Contrato</div>', unsafe_allow_html=True)

col_d1, col_d2, col_d3 = st.columns(3)

with col_d1:
    dol_counts = df["recibe_pagos_en_dolares"].value_counts().reset_index()
    dol_counts.columns = ["Dolarizado", "Cantidad"]
    fig_d1 = px.pie(
        dol_counts,
        values="Cantidad",
        names="Dolarizado",
        color_discrete_map={"Sí": "#10b981", "No": "#e2e8f0"},
        hole=0.55,
    )
    base_layout(fig_d1, "¿Cobra en USD?", height=300)
    fig_d1.update_traces(textinfo="percent+label")
    st.plotly_chart(fig_d1, use_container_width=True)

with col_d2:
    cont_counts = df["tipo_de_contrato_laboral"].value_counts().reset_index()
    cont_counts.columns = ["Contrato", "Cantidad"]
    fig_d2 = px.bar(
        cont_counts.sort_values("Cantidad"),
        x="Cantidad",
        y="Contrato",
        orientation="h",
        color="Cantidad",
        color_continuous_scale=[[0, "#c7d2fe"], [1, "#4338ca"]],
        text="Cantidad",
    )
    base_layout(fig_d2, "Tipos de contrato", height=300)
    fig_d2.update_coloraxes(showscale=False)
    fig_d2.update_traces(textposition="outside")
    st.plotly_chart(fig_d2, use_container_width=True)

with col_d3:
    dol_sal = (
        df.groupby("recibe_pagos_en_dolares")["Ultimo salario neto"]
        .median()
        .reset_index()
    )
    dol_sal.columns = ["Dolarizado", "Salario mediano"]
    fig_d3 = px.bar(
        dol_sal,
        x="Dolarizado",
        y="Salario mediano",
        color="Dolarizado",
        color_discrete_map={"Sí": "#10b981", "No": "#6366f1"},
        text=dol_sal["Salario mediano"].apply(lambda v: f"${v:,.0f}"),
        labels={"Salario mediano": "Salario mediano (ARS)"},
    )
    base_layout(fig_d3, "Salario mediano: cobro en USD vs ARS", height=300)
    fig_d3.update_layout(showlegend=False)
    fig_d3.update_traces(textposition="outside")
    st.plotly_chart(fig_d3, use_container_width=True)

# Freelance top professions
st.markdown("#### Top profesiones — Freelance")
col_f1, col_f2 = st.columns(2)

with col_f1:
    freelance_df = df[df["tipo_de_contrato_laboral"] == "Freelance"]
    if not freelance_df.empty:
        fl_prof = (
            freelance_df["Profesion"]
            .value_counts()
            .head(10)
            .sort_values()
        )
        total_fl = fl_prof.sum()
        fig_fl = go.Figure(go.Bar(
            x=fl_prof.values,
            y=fl_prof.index,
            orientation="h",
            marker=dict(color=fl_prof.values, colorscale=[[0, "#d1fae5"], [1, "#059669"]], showscale=False),
            text=[f"{v:,} ({v/total_fl*100:.1f}%)" for v in fl_prof.values],
            textposition="outside",
        ))
        base_layout(fig_fl, "Top 10 profesiones en modalidad Freelance", height=380)
        st.plotly_chart(fig_fl, use_container_width=True)
    else:
        st.info("No hay registros Freelance con los filtros actuales.")

with col_f2:
    cont_sal = (
        df.groupby("tipo_de_contrato_laboral")["Ultimo salario neto"]
        .median()
        .sort_values(ascending=False)
        .reset_index()
    )
    cont_sal.columns = ["Contrato", "Salario mediano"]
    fig_cs = px.bar(
        cont_sal.sort_values("Salario mediano"),
        x="Salario mediano",
        y="Contrato",
        orientation="h",
        color="Salario mediano",
        color_continuous_scale=[[0, "#fef3c7"], [1, "#d97706"]],
        text=cont_sal.sort_values("Salario mediano")["Salario mediano"].apply(lambda v: f"${v:,.0f}"),
        labels={"Salario mediano": "Salario neto mediano (ARS)"},
    )
    base_layout(fig_cs, "Salario mediano por tipo de contrato", height=380)
    fig_cs.update_coloraxes(showscale=False)
    fig_cs.update_traces(textposition="outside")
    st.plotly_chart(fig_cs, use_container_width=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# SECCIÓN 5 — PROYECCIÓN SALARIAL 2028
# ════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔮 Proyección Salarial 2028</div>', unsafe_allow_html=True)

salario_2023 = df["Ultimo salario neto"].mean()

col_p1, col_p2 = st.columns([2, 3])

with col_p1:
    st.markdown("#### Parámetros del modelo")
    crec_anual = st.slider("Crecimiento anual estimado (%)", 10, 100, 30, step=5) / 100
    anio_target = st.slider("Año de proyección", 2024, 2035, 2028)

    # Proyección simple
    sal_simple = salario_2023 * ((1 + crec_anual) ** (anio_target - 2023))

    # Proyección polinómica
    anios = np.array([2018, 2019, 2020, 2021, 2022, 2023]).reshape(-1, 1)
    salarios_hist = salario_2023 / ((1 + crec_anual) ** (2023 - anios.flatten()))
    poly = PolynomialFeatures(degree=2)
    anios_poly = poly.fit_transform(anios)
    modelo_poly = LinearRegression().fit(anios_poly, salarios_hist)
    sal_poly = modelo_poly.predict(poly.transform(np.array([[anio_target]])))[0]

    st.markdown("---")
    st.metric(f"Proyección simple ({anio_target})", f"${sal_simple:,.2f}",
              delta=f"+{(sal_simple - salario_2023):,.0f} ARS vs 2023")
    st.metric(f"Proyección polinómica ({anio_target})", f"${max(sal_poly, 0):,.2f}",
              delta=f"+{(max(sal_poly,0) - salario_2023):,.0f} ARS vs 2023")
    st.caption(f"Base 2023: ${salario_2023:,.0f} | Crecimiento: {crec_anual*100:.0f}%/año")

with col_p2:
    years_range = list(range(2018, anio_target + 1))
    sal_simple_series = [salario_2023 * ((1 + crec_anual) ** (y - 2023)) for y in years_range]
    sal_poly_series = [
        max(modelo_poly.predict(poly.transform(np.array([[y]])))[0], 0)
        for y in years_range
    ]

    fig_proj = go.Figure()
    fig_proj.add_trace(go.Scatter(
        x=years_range[:6], y=sal_simple_series[:6],
        mode="markers+lines",
        name="Datos históricos (aprox.)",
        line=dict(color="#94a3b8", dash="dot", width=2),
        marker=dict(size=7),
    ))
    fig_proj.add_trace(go.Scatter(
        x=years_range[5:], y=sal_simple_series[5:],
        mode="lines",
        name=f"Proyección simple ({int(crec_anual*100)}%/año)",
        line=dict(color="#6366f1", width=2.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.07)",
    ))
    fig_proj.add_trace(go.Scatter(
        x=years_range[5:], y=sal_poly_series[5:],
        mode="lines",
        name="Proyección polinómica",
        line=dict(color="#f59e0b", width=2.5, dash="dash"),
    ))
    # Anotación año target
    fig_proj.add_vline(x=2023, line=dict(color="#cbd5e1", dash="dot", width=1))
    fig_proj.add_annotation(x=2023, y=max(sal_simple_series) * 0.95,
                             text="2023", showarrow=False,
                             font=dict(color="#94a3b8", size=11))
    base_layout(fig_proj, f"Proyección de salario neto promedio hacia {anio_target}", height=420)
    fig_proj.update_layout(
        xaxis_title="Año",
        yaxis_title="Salario neto promedio (ARS)",
        legend=dict(orientation="h", y=-0.15),
    )
    st.plotly_chart(fig_proj, use_container_width=True)

st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#94a3b8; font-size:0.8rem; padding: 1rem 0 2rem 0;">
    Dashboard construido con <strong>Streamlit</strong> · Datos: <em>Sysarmy Encuesta Salarial Argentina 2023</em>
</div>
""", unsafe_allow_html=True)
