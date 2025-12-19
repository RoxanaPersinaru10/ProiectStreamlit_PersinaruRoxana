import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(
    page_title="EDA - Cerinta 1, 2, 3, 4 & 5",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("EDA cu Streamlit ")

@st.cache_data(ttl=3600)
def load_data(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file, engine="openpyxl")
    elif name.endswith(".xls"):
        df = pd.read_excel(uploaded_file, engine="xlrd")
    else:
        raise ValueError("Format neacceptat. Incarca un fisier CSV sau Excel (.xlsx/.xls).")

    df.columns = [str(c).strip() for c in df.columns]
    return df


def get_numeric_slider_bounds(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    vmin = float(s.min())
    vmax = float(s.max())
    return vmin, vmax


def iqr_bounds(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return None
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return float(lower), float(upper)


uploaded = st.sidebar.file_uploader("Incarca fisier CSV sau Excel", type=["csv", "xlsx", "xls"])

df = None
err = None

if uploaded is not None:
    try:
        df = load_data(uploaded)
    except Exception as e:
        err = str(e)

if err:
    st.error(f"Fisierul NU a putut fi citit: {err}")
    st.stop()

if df is None:
    st.info("Incarca un fisier din sidebar ca sa incepi.")
    st.stop()


file_signature = f"{uploaded.name}_{uploaded.size}"
if st.session_state.get("file_signature") != file_signature:
    st.session_state["file_signature"] = file_signature
    st.session_state.pop("filtered_df", None)

if df.shape[0] == 0 or df.shape[1] == 0:
    st.error("Fisierul a fost citit, dar dataset-ul pare gol (0 randuri sau 0 coloane).")
    st.stop()
else:
    st.success(f"Fisier citit cu succes |  Randuri: {df.shape[0]}  |  Coloane: {df.shape[1]}")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Cerinta 1 — Filtrare",
    "Cerinta 2 — Valori lipsa + statistici",
    "Cerinta 3 — Histograma + Boxplot",
    "Cerinta 4 — Variabile Categorice",
    "Cerinta 5 — Corelatii + Outlieri"
])

# =========================
#  CERINTA 1
# =========================
with tab1:
    with st.expander("Primele 10 randuri (preview)", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in numeric_cols]

    st.sidebar.markdown("---")
    st.sidebar.subheader("Filtrare (Cerinta 1)")

    # Filtre numerice (slidere)
    numeric_filters = {}
    if numeric_cols:
        st.sidebar.markdown("### Coloane numerice")
        for col in numeric_cols:
            bounds = get_numeric_slider_bounds(df[col])
            if bounds is None:
                continue
            vmin, vmax = bounds

            is_int_like = pd.api.types.is_integer_dtype(df[col].dropna()) or (
                float(vmin).is_integer() and float(vmax).is_integer()
            )

            if is_int_like:
                numeric_filters[col] = st.sidebar.slider(
                    f"{col}",
                    min_value=int(np.floor(vmin)),
                    max_value=int(np.ceil(vmax)),
                    value=(int(np.floor(vmin)), int(np.ceil(vmax))),
                    step=1
                )
            else:
                numeric_filters[col] = st.sidebar.slider(
                    f"{col}",
                    min_value=float(vmin),
                    max_value=float(vmax),
                    value=(float(vmin), float(vmax))
                )
    else:
        st.sidebar.info("Nu exista coloane numerice in dataset.")

    # Filtre categorice (multiselect)
    cat_filters = {}
    if cat_cols:
        st.sidebar.markdown("### Coloane categorice")
        selected_cat_cols = st.sidebar.multiselect(
            "Alege coloanele categorice pentru filtrare",
            options=cat_cols,
            default=[]
        )

        for col in selected_cat_cols:
            values = df[col].dropna().astype(str).unique().tolist()
            values.sort()
            chosen = st.sidebar.multiselect(
                f"Valori pentru {col}",
                options=values,
                default=values
            )
            cat_filters[col] = chosen
    else:
        st.sidebar.info("Nu exista coloane categorice în dataset.")

    before_rows = df.shape[0]
    filtered = df.copy()


    for col, (lo, hi) in numeric_filters.items():
        s = pd.to_numeric(filtered[col], errors="coerce")
        filtered = filtered[s.isna() | ((s >= lo) & (s <= hi))]

    for col, allowed in cat_filters.items():
        if allowed is None or len(allowed) == 0:
            filtered = filtered.iloc[0:0]
            break
        filtered = filtered[filtered[col].astype(str).isin(set(allowed))]

    after_rows = filtered.shape[0]

    # pastram filtratul pentru cerintele 2, 3, 4, 5
    st.session_state["filtered_df"] = filtered

    c1, c2, c3 = st.columns(3)
    c1.metric("Randuri inainte", before_rows)
    c2.metric("Randuri dupa", after_rows)
    c3.metric("Randuri eliminate", before_rows - after_rows)

    st.markdown("---")
    st.subheader("Dataframe filtrat")
    st.dataframe(filtered, use_container_width=True)

# =========================
#  CERINTA 2
# =========================
with tab2:
    df_use = st.session_state.get("filtered_df", df)

    st.subheader("Structura dataset")

    c1, c2 = st.columns(2)
    c1.metric("Numar randuri", int(df_use.shape[0]))
    c2.metric("Numar coloane", int(df_use.shape[1]))

    st.markdown("---")
    st.subheader("Tipuri de date pe coloane")
    dtypes_df = pd.DataFrame({
        "Coloana": df_use.columns,
        "Tip date": [str(df_use[c].dtype) for c in df_use.columns]
    })
    st.dataframe(dtypes_df, use_container_width=True)

    st.markdown("---")
    st.subheader("Valori lipsa (missing values)")

    missing_count = df_use.isna().sum()
    missing_pct = (missing_count / len(df_use) * 100).round(2) if len(df_use) > 0 else missing_count * 0

    missing_table = pd.DataFrame({
        "Coloana": df_use.columns,
        "Missing (count)": missing_count.values,
        "Missing (%)": missing_pct.values
    }).sort_values("Missing (%)", ascending=False)

    cols_with_missing = missing_table[missing_table["Missing (count)"] > 0]

    if cols_with_missing.empty:
        st.success("Nu exista valori lipsa in dataset ")
    else:
        st.warning(f"Exista valori lipsa in {cols_with_missing.shape[0]} coloane.")
        st.dataframe(cols_with_missing, use_container_width=True)

        fig, ax = plt.subplots()
        plot_df = cols_with_missing.sort_values("Missing (%)", ascending=True)
        ax.barh(plot_df["Coloana"], plot_df["Missing (%)"])
        ax.set_xlabel("Missing (%)")
        ax.set_ylabel("Coloana")
        ax.set_title("Procent valori lipsa per coloana")
        st.pyplot(fig, clear_figure=True)

    st.markdown("---")
    st.subheader("Statistici descriptive (coloane numerice)")

    num_cols = df_use.select_dtypes(include=["number"]).columns.tolist()
    if not num_cols:
        st.info("Nu exista coloane numerice pentru statistici descriptive.")
    else:
        desc = df_use[num_cols].describe().T
        desc = desc.rename(columns={"50%": "median"})
        desired = ["mean", "median", "std", "min", "25%", "75%", "max"]
        keep = [c for c in desired if c in desc.columns]
        st.dataframe(desc[keep].round(4), use_container_width=True)

# =========================
#  CERINTA 3
# =========================
with tab3:
    st.subheader("Cerinta 3 — Histograma + Boxplot + Statistici")

    df_use = st.session_state.get("filtered_df", df)

    numeric_cols3 = df_use.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols3:
        st.error("Nu exista coloane numerice in dataset (dupa filtrare).")
        st.stop()

    col = st.selectbox("Selecteaza o coloana numerica:", options=numeric_cols3)

    bins = st.slider("Numar bins (10-100):", min_value=10, max_value=100, value=30, step=1)

    series = pd.to_numeric(df_use[col], errors="coerce").dropna()

    if series.empty:
        st.warning("Coloana selectata nu are valori numerice valide (dupa filtrare).")
        st.stop()

    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std(ddof=1)) if len(series) > 1 else 0.0

    m1, m2, m3 = st.columns(3)
    m1.metric("Medie", f"{mean_val:.4f}")
    m2.metric("Mediana", f"{median_val:.4f}")
    m3.metric("Deviatie standard", f"{std_val:.4f}")

    c1, c2 = st.columns(2)

    with c1:
        fig_hist = px.histogram(
            df_use,
            x=col,
            nbins=bins,
            title=f"Histograma: {col} (bins={bins})"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with c2:
        fig_box = px.box(
            df_use,
            y=col,
            points="outliers",
            title=f"Box Plot: {col}"
        )
        st.plotly_chart(fig_box, use_container_width=True)

# =========================
# CERINTA 4
# =========================
with tab4:
    st.subheader("Cerinta 4 — Variabile Categorice (Count Plot + Frecvente)")

    df_use = st.session_state.get("filtered_df", df)

    cat_cols4 = df_use.select_dtypes(include=["object"]).columns.tolist()
    if not cat_cols4:
        st.error("Nu exista coloane categorice in dataset (dupa filtrare).")
        st.stop()

    cat_col = st.selectbox("Selecteaza o coloana categorica:", options=cat_cols4)

    freq_abs = df_use[cat_col].value_counts(dropna=False)
    freq_pct = (freq_abs / freq_abs.sum() * 100).round(2)

    freq_df = pd.DataFrame({
        "Categorie": freq_abs.index.astype(str),
        "Frecventa": freq_abs.values,
        "Procent (%)": freq_pct.values
    }).sort_values("Frecventa", ascending=False)

    fig = px.bar(
        freq_df,
        x="Categorie",
        y="Frecventa",
        text="Frecventa",
        title=f"Count Plot: {cat_col}"
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title=cat_col, yaxis_title="Frecventa")

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Tabel frecvente (absolute si procente)")
    st.dataframe(freq_df, use_container_width=True)

# =========================
# CERINTA 5
# =========================
with tab5:
    st.subheader("Cerinta 5 — Corelatii, Scatter & Outlieri (IQR)")

    df_use = st.session_state.get("filtered_df", df)

    numeric_cols5 = df_use.select_dtypes(include=["number"]).columns.tolist()
    if len(numeric_cols5) < 2:
        st.error("Ai nevoie de cel putin 2 coloane numerice pentru corelatii.")
        st.stop()

    st.markdown("### Matrice de corelatie (Pearson) + Heatmap")
    corr_matrix = df_use[numeric_cols5].corr(method="pearson")
    st.markdown("#### Matricea de corelatie (tabel)")
    st.dataframe(corr_matrix.round(3), use_container_width=True)

    fig_corr = px.imshow(
        corr_matrix,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="Heatmap Corelatii Pearson"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("### Scatter plot pentru 2 variabile + coeficient Pearson")

    c1, c2 = st.columns(2)
    with c1:
        x_col = st.selectbox("Variabila X:", options=numeric_cols5, key="corr_x")
    with c2:
        default_y_index = 1 if len(numeric_cols5) > 1 else 0
        y_col = st.selectbox("Variabila Y:", options=numeric_cols5, index=default_y_index, key="corr_y")

    pair_df = df_use[[x_col, y_col]].dropna()

    if pair_df.empty:
        st.warning("Nu exista valori suficiente (dupa eliminarea NaN) pentru variabilele selectate.")
        st.stop()

    pearson_r = float(pair_df[x_col].corr(pair_df[y_col], method="pearson"))
    st.metric("Coeficient Pearson (r)", f"{pearson_r:.4f}")

    fig_sc = px.scatter(
        pair_df,
        x=x_col,
        y=y_col,
        title=f"Scatter: {x_col} vs {y_col}"
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("### Outlieri (Metoda IQR) — numar si procent per coloana numerica")

    outlier_rows = []
    bounds_map = {}

    for col in numeric_cols5:
        bounds = iqr_bounds(df_use[col])
        if bounds is None:
            continue
        lower, upper = bounds
        bounds_map[col] = (lower, upper)

        s = pd.to_numeric(df_use[col], errors="coerce").dropna()
        mask = (s < lower) | (s > upper)
        count_out = int(mask.sum())
        pct_out = round(count_out / len(s) * 100, 2) if len(s) > 0 else 0.0

        outlier_rows.append({
            "Coloana": col,
            "Outlieri (count)": count_out,
            "Outlieri (%)": pct_out
        })

    outlier_df = pd.DataFrame(outlier_rows).sort_values("Outlieri (%)", ascending=False)
    st.dataframe(outlier_df, use_container_width=True)

    st.markdown("### Vizualizare outlieri pe grafic (scatter colorat)")

    c3, c4 = st.columns(2)
    with c3:
        out_x = st.selectbox("X pentru outlieri:", options=numeric_cols5, key="out_x")
    with c4:
        default_out_y = 1 if len(numeric_cols5) > 1 else 0
        out_y = st.selectbox("Y pentru outlieri:", options=numeric_cols5, index=default_out_y, key="out_y")

    plot_df = df_use[[out_x, out_y]].dropna()

    if plot_df.empty:
        st.warning("Nu exista valori suficiente pentru scatter-ul de outlieri.")
        st.stop()

    lower_x, upper_x = bounds_map.get(out_x, iqr_bounds(plot_df[out_x]))
    lower_y, upper_y = bounds_map.get(out_y, iqr_bounds(plot_df[out_y]))

    mask_out = (
        (plot_df[out_x] < lower_x) | (plot_df[out_x] > upper_x) |
        (plot_df[out_y] < lower_y) | (plot_df[out_y] > upper_y)
    )

    plot_df = plot_df.copy()
    plot_df["Outlier"] = np.where(mask_out, "Outlier", "Normal")

    fig_out = px.scatter(
        plot_df,
        x=out_x,
        y=out_y,
        color="Outlier",
        title=f"Outlieri (IQR) — {out_x} vs {out_y}"
    )
    st.plotly_chart(fig_out, use_container_width=True)
