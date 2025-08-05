import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Crecimiento de Ventas - Zapatos", layout="wide")
st.title("📊 Análisis de Crecimiento en Ventas de Zapatos (Progresión Geométrica)")

# Cargar archivo CSV
st.sidebar.header("📁 Cargar archivo CSV")
uploaded_file = st.sidebar.file_uploader("Sube tu archivo 'venta_decada.csv'", type=["csv"])

if uploaded_file is not None:
    # Leer archivo
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Datos cargados")
    st.dataframe(df)

    # Validación de columnas
    try:
        productos = df.iloc[:, 0]
        ventas = df.iloc[:, 1:].copy()
        ventas.columns = ventas.columns.astype(str)  # Asegurar nombres de columnas como strings (años)

        # Selección de producto
        producto_seleccionado = st.selectbox("🔍 Selecciona un producto para analizar", productos)
        idx = df[df.iloc[:, 0] == producto_seleccionado].index[0]
        ventas_producto = ventas.loc[idx].astype(float)

        # Cálculo del % de crecimiento geométrico (razón entre años)
        razones = ventas_producto.values[1:] / ventas_producto.values[:-1]
        crecimiento_pct = (razones - 1) * 100

        # Crear DataFrame resumen
        años = ventas.columns.astype(int)
        resumen_df = pd.DataFrame({
            "Año": años,
            "Ventas": ventas_producto.values
        })
        resumen_df["% Crecimiento"] = [0] + crecimiento_pct.tolist()

        # Mostrar tabla
        st.subheader("📈 Datos de crecimiento")
        st.dataframe(resumen_df.style.format({"Ventas": "{:.0f}", "% Crecimiento": "{:.2f}"}))

        # Gráfico de ventas y crecimiento
        st.subheader("📊 Visualización")
        fig, ax1 = plt.subplots(figsize=(12, 5))

        sns.lineplot(data=resumen_df, x="Año", y="Ventas", marker="o", ax=ax1, label="Ventas")
        ax1.set_ylabel("Ventas")
        ax1.set_title(f"Ventas y crecimiento - {producto_seleccionado}")

        ax2 = ax1.twinx()
        sns.barplot(data=resumen_df, x="Año", y="% Crecimiento", alpha=0.3, ax=ax2, color="orange", label="% Crecimiento")
        ax2.set_ylabel("% Crecimiento")

        st.pyplot(fig)

        # Análisis con regresión logarítmica
        X = resumen_df["Año"].values.reshape(-1, 1)
        y_log = np.log(resumen_df["Ventas"].values.reshape(-1, 1))  # Log para regresión exponencial

        modelo = LinearRegression()
        modelo.fit(X, y_log)
        y_pred = np.exp(modelo.predict(X))  # Inversa de log para volver a escala original

        resumen_df["Regresión (Estimado)"] = y_pred.flatten()

        st.subheader("📉 Regresión exponencial (estimación)")
        st.write(f"Coeficiente de crecimiento logarítmico: **{modelo.coef_[0][0]:.4f}**")
        st.write(f"Intercepto: **{modelo.intercept_[0]:.4f}**")

        # Gráfico comparación real vs estimado
        fig2, ax3 = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=resumen_df, x="Año", y="Ventas", marker="o", label="Ventas reales", ax=ax3)
        sns.lineplot(data=resumen_df, x="Año", y="Regresión (Estimado)", marker="X", label="Regresión estimada", ax=ax3)
        ax3.set_title(f"Comparación de ventas reales vs estimadas - {producto_seleccionado}")
        ax3.set_ylabel("Ventas")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error procesando el archivo: {e}")

else:
    st.warning("Por favor sube el archivo 'venta_decada.csv' para comenzar.")
