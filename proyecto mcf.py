import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, norm, t, shapiro

# Configuración inicial de la página
st.set_page_config(page_title="Proyecto Riesgos TSLA", layout="wide")

st.title("📊 Análisis de Riesgo: Tesla (TSLA)")

#DESCARGA DE DATOS
# Usamos tus nombres de variables originales
df_precios = yf.download("TSLA", start="2010-01-01")['Close']
df_rend = df_precios.pct_change().dropna()

# ---ESTADÍSTICOS ---
media = df_rend.mean()
if isinstance(media, pd.Series):
    media = media.item()

kurt = kurtosis(df_rend)
if isinstance(kurt, np.ndarray): # kurtosis de scipy suele devolver arrays
    kurt = kurt[0]

sesgo = skew(df_rend)
if isinstance(sesgo, np.ndarray):
    sesgo = sesgo[0]

col1, col2, col3 = st.columns(3)
col1.metric("Media de Rendimientos", f"{media:.4f}")
col2.metric("Kurtosis (Exceso)", f"{kurt:.2f}")
col3.metric("Sesgo (Skew)", f"{sesgo:.2f}")

# --- VISUALIZACIÓN DE RENDIMIENTOS ---
st.subheader("Gráfico de Rendimientos Diarios")
fig_rend, ax_rend = plt.subplots(figsize=(13, 5))
ax_rend.plot(df_rend.index, df_rend.values, color='#E69F42', linewidth=0.7)
ax_rend.axhline(y=0, color='red', linestyle='--', alpha=0.6)
ax_rend.set_title("Rendimientos Diarios de TSLA")
st.pyplot(fig_rend)
# Prueba 
# PRUEBA DE SHAPIRO-WILK
# La prueba devuelve: (estadístico W, p-valor)
stat_shapiro, p_val_shapiro = shapiro(df_rend)

# Mostrar resultado de Shapiro
st.subheader("Prueba de Normalidad (Shapiro-Wilk)")
col_s1, col_s2 = st.columns(2)

with col_s1:
    st.metric("Estadístico W", f"{stat_shapiro:.4f}")
    st.metric("P-V", f"{p_val_shapiro:.4f}")

with col_s2:
    if p_val_shapiro < 0.05:
        st.error("Los datos NO siguen una distribución normal (Rechazamos H0)")
        st.write("Esto sugiere que el VaR Normal podría subestimar el riesgo debido a colas pesadas.")
    else:
        st.success("Los datos siguen una distribución normal (No rechazamos H0)")
# --- TABLA DE VaR Y ES ESTÁTICO ---
st.subheader("Cálculo de VaR y ES")

mean = np.mean(df_rend)
stdev = np.std(df_rend)
nu, loc_t, scale_t = t.fit(df_rend)

alphas = [0.95, 0.975, 0.99]
n_sims = 100000
sim_normal = np.random.normal(mean, stdev, n_sims)
sim_student = t.rvs(nu, loc=loc_t, scale=scale_t, size=n_sims)

resultados_lista = []

for alpha in alphas:
    # --- TUS FÓRMULAS ORIGINALES ---
    var_h = df_rend.quantile(1 - alpha)
    es_h = df_rend[df_rend <= var_h].mean()

    var_n = norm.ppf(1 - alpha, mean, stdev)
    es_n = mean - stdev * (norm.pdf(norm.ppf(1 - alpha)) / (1 - alpha))

    z_t = t.ppf(1 - alpha, nu)
    es_t_analitico = ((t.pdf(z_t, nu) / (1 - alpha)) * ((nu + z_t**2) / (nu - 1))) * scale_t + loc_t
    var_t = t.ppf(1 - alpha, nu, loc_t, scale_t)

    var_mcn = np.percentile(sim_normal, (1 - alpha) * 100)
    es_mcn = sim_normal[sim_normal <= var_mcn].mean()

    var_mct = np.percentile(sim_student, (1 - alpha) * 100)
    es_mct = sim_student[sim_student <= var_mct].mean()

    # --- CORRECCIÓN: Convertir cada valor a un float escalar ---
    # Usamos float(x.item()) si es una Serie/Array de un solo elemento
    def to_scalar(x):
        try:
            return float(x.item()) if hasattr(x, 'item') else float(x)
        except:
            return float(x)

    resultados_lista.append({
        "Confianza": f"{alpha*100}%",
        "VaR Hist": to_scalar(var_h), 
        "ES Hist": to_scalar(es_h),
        "VaR Norm": to_scalar(var_n), 
        "ES Norm": to_scalar(es_n),
        "VaR t-Stud": to_scalar(var_t), 
        "ES t-Stud": to_scalar(es_t_analitico),
        "VaR MC-Norm": to_scalar(var_mcn), 
        "ES MC-Norm": to_scalar(es_mcn),
        "VaR MC-t": to_scalar(var_mct), 
        "ES MC-t": to_scalar(es_mct)
    })

# Crear DataFrame y aplicar formato
df_final = pd.DataFrame(resultados_lista).set_index("Confianza")
st.dataframe(df_final.style.format("{:.4%}"))

# ---ROLLING WINDOWS ---
st.subheader("Backtesting: VaR Rolling (95% vs 99%)")

window = 252
alphas_roll = [0.95, 0.99]
df_rend_sq = df_rend.squeeze()
resultados_rolling = {'Rendimientos': df_rend_sq.iloc[window:]}

for a in alphas_roll:
    resultados_rolling[f'VaR_Hist_{a}'] = []
    resultados_rolling[f'VaR_Norm_{a}'] = []

for i in range(window, len(df_rend_sq)):
    ventana = df_rend_sq.iloc[i-window : i]
    for a in alphas_roll:
        resultados_rolling[f'VaR_Hist_{a}'].append(ventana.quantile(1 - a))
        m_roll, s_roll = ventana.mean(), ventana.std()
        resultados_rolling[f'VaR_Norm_{a}'].append(norm.ppf(1 - a, m_roll, s_roll))

df_rolling = pd.DataFrame(resultados_rolling, index=df_rend_sq.index[window:])

fig_roll, ax_roll = plt.subplots(figsize=(14, 7))
ax_roll.plot(df_rolling.index, df_rolling['Rendimientos'], color='silver', alpha=0.4, label='P&L')
ax_roll.plot(df_rolling.index, df_rolling['VaR_Hist_0.95'], label='VaR Hist 95%', color='orange')
ax_roll.plot(df_rolling.index, df_rolling['VaR_Hist_0.99'], label='VaR Hist 99%', color='red')
ax_roll.legend()
st.pyplot(fig_roll)

# --- EFICIENCIA (VIOLACIONES) ---
st.subheader("Tabla de Eficiencia (Violaciones)")

resumen_eficiencia = []
for col in df_rolling.columns:
    if col == 'Rendimientos': continue
    n_violaciones = (df_rolling['Rendimientos'] < df_rolling[col]).sum()
    pct_violaciones = (n_violaciones / len(df_rolling)) * 100
    resumen_eficiencia.append({
        'Medida de Riesgo': col,
        'Violaciones': n_violaciones,
        '% Violaciones': f"{pct_violaciones:.2f}%",
        'Estatus': "✅ Buena" if pct_violaciones < 2.5 else "❌ Ajustar"
    })

st.table(pd.DataFrame(resumen_eficiencia).set_index('Medida de Riesgo'))

# ---VaR VOLATILIDAD MÓVIL ---
st.subheader("VaR con Volatilidad Móvil (Fórmula Especial)")

alphas_f = [0.05, 0.01]
resultados_f = {'Rendimientos': df_rend_sq.iloc[window:]}

for alpha_f in alphas_f:
    q_alpha = norm.ppf(alpha_f)
    vol_rolling = df_rend_sq.rolling(window=window).std().iloc[window:]
    resultados_f[f'VaR_Vol_Movil_{alpha_f}'] = q_alpha * vol_rolling

df_f = pd.DataFrame(resultados_f)

fig_f, ax_f = plt.subplots(figsize=(14, 7))
ax_f.plot(df_f.index, df_f['Rendimientos'], color='silver', alpha=0.4)
ax_f.plot(df_f.index, df_f['VaR_Vol_Movil_0.05'], label='VaR Vol 5%', color='blue')
ax_f.plot(df_f.index, df_f['VaR_Vol_Movil_0.01'], label='VaR Vol 1%', color='darkblue')
ax_f.legend()
st.pyplot(fig_f)
# TABLA PARA VOLATILIDAD MÓVIL
st.subheader("Eficiencia: VaR Volatilidad Móvil")

resumen_eficiencia_f = []
for col in df_f.columns:
    if col == 'Rendimientos': continue
    
    # Una violación ocurre si el rendimiento es menor al VaR estimado
    n_violaciones = (df_f['Rendimientos'] < df_f[col]).sum()
    pct_violaciones = (n_violaciones / len(df_f)) * 100
    
    # Una buena estimación tiene < 2.5% de violaciones
    resumen_eficiencia_f.append({
        'Medida de Riesgo': col,
        'Violaciones Totales': n_violaciones,
        '% Violaciones': f"{pct_violaciones:.2f}%",
        'Estatus': "✅ Buena" if pct_violaciones < 2.5 else "❌ Ajustar"
    })

# Mostrar la tabla en Streamlit
st.table(pd.DataFrame(resumen_eficiencia_f).set_index('Medida de Riesgo'))
