# app_fourier_es.py
# App en español con dos modos:
#  1) "Solo graficar f(t)": escribís la función y la app la grafica
#  2) "Serie de Fourier": calcula a0, an, bn y la reconstrucción S_N(t)
# Requisitos: streamlit, numpy, matplotlib, pandas

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Series de Fourier (ES)", page_icon="📈", layout="wide")

# ==========================
# Utilidades en español
# ==========================
PI = np.pi

def fraccion(u):
    """Parte fraccionaria de u (vectorizable)."""
    return u - np.floor(u)

# Entorno seguro: funciones y constantes permitidas (en español)
entorno_permitido = {
    # constantes
    "pi": np.pi,
    "e": np.e,
    # variable tiempo (se setea dinámicamente)
    "t": None,
    # funciones trigonométricas
    "sen": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsen": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    # varias
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "raiz": np.sqrt,
    "abs": np.abs,
    "signo": np.sign,
    "heaviside": np.heaviside,
    "mod": np.mod,
    "min": np.minimum,
    "max": np.maximum,
    "fraccion": fraccion,
}

# ==========================
# Señales base (útiles para Fourier)
# ==========================

def onda_cuadrada(t, T, A=1.0, offset=0.0, duty=0.5):
    fase = fraccion(t / T)
    y = np.where(fase < duty, A, -A)
    return y + offset

def onda_triangular(t, T, A=1.0, offset=0.0):
    fase = fraccion(t / T)
    y = 4 * A * np.abs(fase - 0.5) - A
    return y + offset

def diente_de_sierra(t, T, A=1.0, offset=0.0):
    fase = fraccion(t / T)
    y = 2 * A * fase - A
    return y + offset

def pulso(t, T, A_alto=1.0, A_bajo=0.0, duty=0.2):
    fase = fraccion(t / T)
    return np.where(fase < duty, A_alto, A_bajo)

def escalon(t, T, A=1.0):
    fase = fraccion(t / T)
    return np.where(fase < 0.5, A, 0.0)

# ==========================
# Evaluación segura de expresiones en t
# ==========================

def eval_personalizada(expr: str, t: np.ndarray, variables_extra: dict | None = None):
    """Evalúa expr en un entorno controlado. Usa 't' como variable.
    variables_extra permite pasar, por ejemplo, T=..., A=..., etc.
    """
    permitido = dict(entorno_permitido)
    permitido["t"] = t
    if variables_extra:
        permitido.update(variables_extra)
    try:
        return eval(expr, {"__builtins__": {}}, permitido)
    except Exception as e:
        raise ValueError(f"Error al evaluar la expresión: {e}")

# ==========================
# Cálculo de coeficientes de Fourier (período T, L=T/2)
# ==========================

def coeficientes_fourier(f_vals, t_grid, T, N):
    L = T / 2
    a0 = (1 / L) * np.trapz(f_vals, t_grid)
    an = np.zeros(N)
    bn = np.zeros(N)
    for n in range(1, N + 1):
        cosn = np.cos(n * np.pi * t_grid / L)
        sinn = np.sin(n * np.pi * t_grid / L)
        an[n - 1] = (1 / L) * np.trapz(f_vals * cosn, t_grid)
        bn[n - 1] = (1 / L) * np.trapz(f_vals * sinn, t_grid)
    return a0, an, bn

def serie_fourier(a0, an, bn, t_grid, T):
    L = T / 2
    S = np.full_like(t_grid, a0 / 2.0)
    N = len(an)
    for n in range(1, N + 1):
        S += an[n - 1] * np.cos(n * np.pi * t_grid / L) + bn[n - 1] * np.sin(n * np.pi * t_grid / L)
    return S

# ==========================
# UI
# ==========================

st.title("📈 Series de Fourier — App simple (ES)")
st.markdown(
    """
    Elegí un **modo**:
    - **Solo graficar f(t)**: escribís la función en términos de `t` y la app la dibuja.
    - **Serie de Fourier**: además calcula \(a_0, a_n, b_n\) y la aproximación \(S_N(t)\) en un período.
    """
)

modo = st.radio("Modo de trabajo", ["Solo graficar f(t)", "Serie de Fourier"], horizontal=True)

if modo == "Solo graficar f(t)":
    with st.form("form_grafica_simple"):
        colA, colB, colC = st.columns([1,1,1])
        with colA:
            t_min = st.number_input("t mínimo", value=-10.0)
        with colB:
            t_max = st.number_input("t máximo", value=10.0)
        with colC:
            M = st.slider("Puntos", min_value=200, max_value=20000, value=2000, step=200)
        st.info("Usá `t` como variable. Si usás `T`, definila abajo.")
        expr = st.text_input(
            "f(t) =", value="sen(t) + 0.5*cos(3*t)",
            help="Funciones válidas: sen, cos, tan, exp, log, sqrt, abs, signo, heaviside(x, h0), fraccion, etc. Constantes: pi, e."
        )
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            usar_T = st.checkbox("Definir T (opcional)", value=True)
        with col2:
            T_val = st.number_input("T", value=float(2*np.pi)) if usar_T else None
        with col3:
            pass
        submitted = st.form_submit_button("Graficar f(t)")

    if submitted:
        if t_max <= t_min:
            st.error("t máximo debe ser mayor que t mínimo.")
        else:
            t = np.linspace(t_min, t_max, int(M))
            try:
                vars_extra = {"T": T_val} if usar_T else None
                f = eval_personalizada(expr, t, vars_extra)
                fig = plt.figure(figsize=(9, 4.8))
                plt.plot(t, f, label="f(t)")
                plt.xlabel("t")
                plt.ylabel("f(t)")
                plt.title("Gráfica de la función ingresada")
                plt.grid(True, alpha=0.35)
                plt.legend()
                st.pyplot(fig, clear_figure=True)
            except Exception as e:
                st.error(str(e))

else:  # Serie de Fourier
    col_izq, col_der = st.columns([1, 2])

    with col_izq:
        st.subheader("Parámetros del problema")
        T = st.number_input("Período T", value=float(2 * np.pi), min_value=1e-6, step=0.1, format="%.6f")
        N = st.slider("N° de términos N", 1, 100, 20)
        M = st.slider("Puntos de muestreo por período", 2000, 20000, 6000, step=1000)

        tipo = st.selectbox(
            "Tipo de señal",
            (
                "Personalizada (expresión en t)",
                "Onda cuadrada",
                "Onda triangular",
                "Diente de sierra",
                "Pulso",
                "Escalón",
            ),
        )

        params = {}
        expr = None
        if tipo == "Onda cuadrada":
            params["A"] = st.number_input("Amplitud ±A", value=1.0)
            params["offset"] = st.number_input("Offset", value=0.0)
            params["duty"] = st.slider("Duty (fracción periodo)", 0.05, 0.95, 0.5, step=0.05)
        elif tipo == "Onda triangular":
            params["A"] = st.number_input("Amplitud A", value=1.0)
            params["offset"] = st.number_input("Offset", value=0.0)
        elif tipo == "Diente de sierra":
            params["A"] = st.number_input("Amplitud A", value=1.0)
            params["offset"] = st.number_input("Offset", value=0.0)
        elif tipo == "Pulso":
            params["A_alto"] = st.number_input("Nivel alto", value=1.0)
            params["A_bajo"] = st.number_input("Nivel bajo", value=0.0)
            params["duty"] = st.slider("Duty (fracción periodo)", 0.01, 0.99, 0.2, step=0.01)
        elif tipo == "Escalón":
            params["A"] = st.number_input("Nivel del escalón (mitad del período)", value=1.0)
        else:
            st.markdown(
                "Ejemplos: `sen(2*pi*t/T)`, `abs(sen(3*t))`, `heaviside(sen(t), 0)`, `exp(-abs(t))`"
            )
            expr = st.text_input("f(t) en un período (usá 't'; funciones en español)", value="sen(2*pi*t/T)")

        st.markdown("---")
        st.caption(
            "Coeficientes:  a0=1/L∫ f,  an=1/L∫ f cos(nπx/L),  bn=1/L∫ f sin(nπx/L), con L=T/2."
        )

    with col_der:
        # Dominio de integración y muestreo
        L = T / 2
        t = np.linspace(-L, L, int(M), endpoint=False)

        # Construcción de f(t)
        if tipo == "Onda cuadrada":
            f = onda_cuadrada(t, T, **params)
        elif tipo == "Onda triangular":
            f = onda_triangular(t, T, **p
