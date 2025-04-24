
# 📊 Dashboard Curva del Tesoro de EE.UU. - FRED API

Este proyecto permite visualizar en tiempo real la curva de tasas del Tesoro de EE.UU. utilizando datos de la [FRED (Federal Reserve Economic Data)](https://fred.stlouisfed.org/), mediante un dashboard web desarrollado con Streamlit.

---

## 🚀 Funcionalidades incluidas

- **Curva Spot** interactiva por vencimiento
- **Curva Break-even** con comparación al objetivo de inflación de la Fed
- **Componentes Principales (PCA):** Nivel, Pendiente, Curvatura
- **Heatmap de Tasas Forward**
- **Tablero de tasas** con cambios diarios, semanales, mensuales y anuales
- Integración directa con la **API oficial de la FRED**

---

## 🧰 Requisitos

- Python 3.7+
- Paquetes:

```bash
pip install streamlit pandas numpy matplotlib seaborn sklearn fredapi
```

---

## 🔑 API Key de FRED

Obtené tu API Key gratuita desde: [https://fred.stlouisfed.org/](https://fred.stlouisfed.org/)
Será requerida al iniciar el dashboard para acceder a los datos.

---

## ▶️ Cómo ejecutar el dashboard

```bash
streamlit run streamlit_app.py
```

Luego abrí el navegador en [http://localhost:8501](http://localhost:8501)

---
