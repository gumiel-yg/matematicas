import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st

# Configuración general
st.set_page_config(page_title="Crecimiento de Ventas - Progresión Geométrica", layout="centered")

st.title("📈 Simulación de Crecimiento de Ventas (Progresión Geométrica)")

# Entradas del usuario
a = st.number_input("🔹 Valor inicial de ventas (en miles):", min_value=1.0, value=10.0, step=1.0)
r = st.number_input("🔹 Razón de crecimiento geométrica (por ejemplo: 1.10 = +10%):", min_value=0.01, value=1.10, step=0.01)
n = 10  # Número de períodos (puedes permitir que el usuario lo cambie si deseas)

# Cálculo de la progresión geométrica
ventas = [a * (r**i) for i in range(n)]
porcentaje_crecimiento = [(ventas[i] - ventas[i-1]) / ventas[i-1] * 100 if i > 0 else 0 for i in range(n)]

# Crear DataFrame
df = pd.DataFrame({
    'Periodo': range(1, n + 1),
    'Ventas (miles)': ventas,
    'Crecimiento %': porcentaje_crecimiento
})

st.subheader("📊 Tabla de Ventas y Crecimiento")
st.dataframe(df.style.format({"Ventas (miles)": "{:.2f}", "Crecimiento %": "{:.2f}"}))

# Visualización con Seaborn y Matplotlib
st.subheader("📈 Gráfico de crecimiento geométrico")
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='Periodo', y='Ventas (miles)', marker='o', ax=ax)
ax.set_title("Crecimiento de Ventas")
ax.set_ylabel("Ventas (en miles)")
ax.set_xlabel("Periodo")
st.pyplot(fig)

# Análisis con scikit-learn: regresión lineal sobre el logaritmo (porque es crecimiento exponencial)
log_ventas = np.log(df['Ventas (miles)']).values.reshape(-1, 1)
X = np.array(df['Periodo']).reshape(-1, 1)
model = LinearRegression()
model.fit(X, log_ventas)

st.subheader("🔍 Análisis con regresión lineal (sobre logaritmo)")
st.write(f"🔹 Coeficiente de crecimiento logarítmico: {model.coef_[0][0]:.4f}")
st.write(f"🔹 Intercepto: {model.intercept_[0]:.4f}")

# Predicción
ventas_pred = np.exp(model.predict(X))
df['Ventas (Regresión)'] = ventas_pred

# Mostrar comparación visual
fig2, ax2 = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df, x='Periodo', y='Ventas (miles)', marker='o', label="Ventas reales", ax=ax2)
sns.lineplot(data=df, x='Periodo', y='Ventas (Regresión)', marker='X', label="Regresión (estimada)", ax=ax2)
ax2.set_title("Comparación: Ventas reales vs estimadas por regresión")
ax2.set_ylabel("Ventas (en miles)")
ax2.set_xlabel("Periodo")
st.pyplot(fig2)
