# 💼 Empleabilidad IT Argentina 2023 — Dashboard

Dashboard interactivo construido con **Streamlit** basado en la Encuesta de Remuneración Salarial Argentina 2023 de Sysarmy.

## 📂 Estructura del proyecto

```
dashboard/
├── app.py                   ← App principal
├── requirements.txt         ← Dependencias Python
└── README.md                ← Este archivo
```

> ⚠️ El archivo CSV debe estar en la **misma carpeta** que `app.py`.

---

## 🚀 Correr localmente

```bash
# 1. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Levantar la app
streamlit run app.py
```

La app quedará disponible en `http://localhost:8501`.

---

## ☁️ Deploy en Streamlit Community Cloud (gratuito)

1. Subí este proyecto a un repositorio **GitHub** (público o privado).
2. Incluí el CSV en el repositorio o usá el campo de ruta en el sidebar.
3. Entrá a [share.streamlit.io](https://share.streamlit.io) y conectá tu repo.
4. Seleccioná `app.py` como archivo principal → **Deploy**.

### Configurar la ruta del CSV en producción

En la barra lateral del dashboard hay un campo **"Ruta del CSV"**.  
Escribí exactamente el nombre del archivo tal como está en tu repositorio:

```
2023.1_Sysarmy_Encuesta de remuneracin salarial Argentina.csv
```

---

## 🐳 Deploy con Docker (opcional)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t empleabilidad-dashboard .
docker run -p 8501:8501 empleabilidad-dashboard
```

---

## 📊 Secciones del dashboard

| Sección | Descripción |
|---|---|
| **KPIs** | Totales, salario promedio/mediano, % dolarizado |
| **Profesión y Seniority** | Top N profesiones, box plots, comparación Junior/SS/Senior |
| **Género** | Distribución, brecha salarial, seniority por género |
| **Región / Provincia** | Top provincias por salario, scatter representatividad |
| **Dolarización y Contrato** | Breakdown por tipo, salario según cobro en USD |
| **Proyección 2028** | Modelo simple + polinómico con parámetros ajustables |

---

## 🔧 Filtros disponibles

Todos los gráficos respetan los filtros del sidebar:
- Seniority (Junior / Semi-Senior / Senior)
- Género
- Tipo de contrato
- Rango de salario neto

---

*Fuente de datos: [Sysarmy — Encuesta de Remuneración Salarial Argentina 2023](https://sysarmy.com/blog/posts/resultados-de-la-encuesta-de-sueldos-2023-1/)*
