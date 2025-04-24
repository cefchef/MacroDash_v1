
import streamlit as st
import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime, timedelta
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración inicial
st.set_page_config(page_title="Dashboard Curva del Tesoro FRED", layout="wide")

st.title("📊 Dashboard Curva del Tesoro - FRED API")
st.markdown("Datos reales de tasas del Tesoro de EE.UU. directamente desde la FRED. Incluye tasas spot, reales, break-even, forward, análisis PCA y visualizaciones interactivas.")

# API KEY
fred_api_key = st.text_input("🔐 Ingresá tu FRED API Key:", type="password")
if not fred_api_key:
    st.stop()
fred = Fred(api_key=fred_api_key)

# Rango temporal
start_date = datetime.today() - timedelta(days=4*365)
end_date = datetime.today()

# Descargar datos
spot_codes = {'1M':'GS1M','3M':'GS3M','6M':'GS6M','1Y':'GS1','2Y':'GS2','3Y':'GS3','5Y':'GS5','7Y':'GS7','10Y':'GS10','20Y':'GS20','30Y':'GS30'}
real_codes = {'5Y': 'DFII5', '7Y': 'DFII7', '10Y': 'DFII10', '20Y': 'DFII20', '30Y': 'DFII30'}

@st.cache_data
def get_data():
    df_spot = pd.DataFrame({k: fred.get_series(v, start_date) for k, v in spot_codes.items()})
    df_real = pd.DataFrame({k: fred.get_series(v, start_date) for k, v in real_codes.items()})
    return df_spot.dropna(), df_real.dropna()

df_spot, df_real = get_data()
df_breakeven = df_spot[df_real.columns] - df_real

# PCA
scaler = StandardScaler()
scaled = scaler.fit_transform(df_spot.dropna())
pca = PCA(n_components=3)
pca_result = pca.fit_transform(scaled)
pca_df = pd.DataFrame(pca_result, columns=["Nivel", "Pendiente", "Curvatura"], index=df_spot.dropna().index)

# FORWARD
maturity_years = {'1M': 1/12, '3M': 0.25, '6M': 0.5, '1Y': 1, '2Y': 2, '3Y': 3, '5Y': 5, '7Y': 7, '10Y': 10, '20Y': 20, '30Y': 30}
sorted_terms = sorted(df_spot.columns, key=lambda x: maturity_years[x])
df_forward_all = pd.DataFrame(index=df_spot.index)
def forward_rate(t1, t2, R1, R2): return ((t2 * R2) - (t1 * R1)) / (t2 - t1)
for i, t1 in enumerate(sorted_terms[:-1]):
    for t2 in sorted_terms[i+1:]:
        df_forward_all[f"{t1}→{t2}"] = forward_rate(maturity_years[t1], maturity_years[t2], df_spot[t1], df_spot[t2])

# Sidebar para seleccionar fecha
valid_dates = df_spot.index.intersection(df_breakeven.index).intersection(pca_df.index)
selected_date = st.sidebar.selectbox("📅 Seleccionar fecha", valid_dates.sort_values(ascending=False).strftime("%Y-%m-%d"))
selected_date = pd.to_datetime(selected_date)

# Sección 1: Curva Spot
st.subheader("1️⃣ Curva Spot del Tesoro")
spot_curve = df_spot.loc[selected_date].dropna()
fig1, ax1 = plt.subplots()
x = [maturity_years[k] for k in spot_curve.index]
ax1.plot(x, spot_curve.values, marker='o')
ax1.set_title("Curva Spot")
ax1.set_xlabel("Vencimiento (años)")
ax1.set_ylabel("Tasa (%)")
ax1.grid(True)
st.pyplot(fig1)

# Sección 2: Curva Break-even
st.subheader("2️⃣ Curva Break-even (Inflación Implícita)")
if selected_date in df_breakeven.index:
    be_curve = df_breakeven.loc[selected_date].dropna()
    x = [int(k.strip("Y")) for k in be_curve.index]
    fig2, ax2 = plt.subplots()
    ax2.plot(x, be_curve.values, marker='o', label="Break-even")
    ax2.axhline(2.0, color='gray', linestyle='--', label='Objetivo Fed')
    ax2.set_title("Curva Break-even")
    ax2.set_xlabel("Vencimiento (años)")
    ax2.set_ylabel("Inflación Esperada (%)")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)

# Sección 3: Componentes PCA
st.subheader("3️⃣ Componentes Principales (PCA)")
pca_vals = pca_df.loc[selected_date]
fig3, ax3 = plt.subplots()
bars = ax3.bar(pca_vals.index, pca_vals.values, color=["#1f77b4", "red" if pca_vals['Pendiente'] < -1 else "#1f77b4", "#2ca02c"])
ax3.axhline(0, color='black', linestyle='--')
ax3.set_title("PCA: Nivel, Pendiente, Curvatura")
st.pyplot(fig3)

# Sección 4: Heatmap Forward
st.subheader("4️⃣ Heatmap de Forward Rates")
from matplotlib.colors import TwoSlopeNorm
fwd_matrix = {}
for i, t1 in enumerate(sorted_terms[:-1]):
    row = {}
    for t2 in sorted_terms[i+1:]:
        label = f"{t1}→{t2}"
        if label in df_forward_all.columns:
            row[t2] = df_forward_all.loc[selected_date, label]
    fwd_matrix[t1] = row
df_fwd = pd.DataFrame(fwd_matrix).T
fig4, ax4 = plt.subplots(figsize=(10, 6))
sns.heatmap(df_fwd, annot=True, fmt=".2f", cmap="coolwarm", ax=ax4, cbar_kws={'label': 'Forward Rate (%)'})
ax4.set_title("Forward Rates Matrix")
st.pyplot(fig4)

# Sección 5: Tablero con Cambios Porcentuales
st.subheader("5️⃣ Tablero de Tasas y Cambios (%)")

# Función para calcular variaciones
def calcular_cambios(df, label, columnas):
    tabla = pd.DataFrame(index=columnas)
    tabla["Último"] = df.loc[selected_date, columnas]
    for dias, nombre in zip([1, 5, 21, 252], ['D', 'W', 'M', 'A']):
        try:
            pasada = df.shift(dias).loc[selected_date, columnas]
            cambio = ((tabla["Último"] - pasada) / pasada) * 100
            tabla[f"Cambio {nombre} (%)"] = cambio
        except:
            tabla[f"Cambio {nombre} (%)"] = np.nan
    tabla.index = [f"{label} {col}" for col in tabla.index]
    return tabla.round(2)

# Compilar tablas de tasas y PCA
tablas = []

# Spot
tablas.append(calcular_cambios(df_spot, "Spot", df_spot.columns))

# Real
tablas.append(calcular_cambios(df_real, "Real", df_real.columns))

# Break-even
tablas.append(calcular_cambios(df_breakeven, "BEI", df_breakeven.columns))

# Forward
tablas.append(calcular_cambios(df_forward_all, "FWD", df_forward_all.columns))

# PCA
tablas.append(calcular_cambios(pca_df, "PCA", pca_df.columns))

# Mostrar tabla final
df_tablero = pd.concat(tablas)
st.dataframe(df_tablero.style.highlight_null(null_color='lightgray'))
