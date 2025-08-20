# app_fourier_es.py
# App simple en español para calcular y visualizar series de Fourier
# Autor: (tu nombre)
# Requisitos: streamlit, numpy, matplotlib

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from io import StringIO

st.set_page_config(page_title="Series de Fourier (ES)", page_icon="📈", layout="wide")

# ==========================
# Utilidades en español
# ==========================
PI = np.pi

def fraccion(u):
    """Parte fraccionaria de u (u puede ser vector)."""
    return u - np.floor(u)

# Diccionario de funciones permitidas en expresiones personalizadas (todo en español)
entorno_permitido = {
    "np": np,
    "t": None,  # se setea dinámicamente antes de evaluar
    # constantes
    "pi": np.pi,
    "e": np.e,
    # funciones matemáticas en español
    "sen": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "arcsen": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
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
# Generadores de señales base (período T)
# ==========================

def onda_cuadrada(t, T, A=1.0, offset=0.0, duty=0.5):
    """Onda cuadrada de amplitud A (±A), duty en [0,1]."""
    fase = fraccion(t / T)
    y = np.where(fase < duty, A, -A)
    return y + offset


def onda_triangular(t, T, A=1.0, offset=0.0):
    """Onda triangular simétrica en [-A, A]."""
    fase = fraccion(t / T)
    y = 4 * A * np.abs(fase - 0.5) - A
    return y + offset


def diente_de_sierra(t, T, A=1.0, offset=0.0):
    """Diente de sierra simétrico en [-A, A]."""
    fase = fraccion(t / T)
    y = 2 * A * fase - A
    return y + offset


def pulso(t, T, A_alto=1.0, A_bajo=0.0, duty=0.2):
    fase = fraccion(t / T)
    return np.where(fase < duty, A_alto, A_bajo)


def escalon(t, T, A=1.0):
    # escalón periódico en 0 (mitad del período positivo, mitad negativo)
    fase = fraccion(t / T)
    return np.where(fase < 0.5, A, 0.0)


def eval_personalizada(expr, t):
    # Seguridad básica: deshabilitar builtins y sólo permitir el entorno definido
    permitido = dict(entorno_permitido)
    permitido["t"] = t
    try:
        return eval(expr, {"__builtins__": {}}, permitido)
    except Exception as e:
        raise ValueError(f"Error al evaluar la expresión: {e}")

# ==========================
# Cálculo de coeficientes de Fourier (período T, L=T/2)
# ==========================

def coeficientes_fourier(f_vals, t_grid, T, N):
    """Calcula a0, an, bn mediante integración numérica (trapecio) en [-L, L]."""
    L = T / 2
    # t_grid debe cubrir un período completo [-L, L)
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
col_izq, col_der = st.columns([1, 2])

with col_izq:
    st.title("📈 Series de Fourier — App simple (ES)")
    st.markdown(
        """
        Esta aplicación calcula **coeficientes de Fourier** \(a_0, a_n, b_n\) y muestra la aproximación con \(N\) términos.
        Todo está en español, con señales **predeterminadas** o **personalizadas** (usando `t` como variable y funciones como `sen`, `cos`, `exp`, `abs`, `heaviside`, etc.).
        """
    )

    with st.expander("Parámetros del problema"):
        T = st.number_input("Período T", value=float(2 * np.pi), min_value=1e-6, step=0.1, format="%.6f")
        N = st.slider("N° de términos N", 1, 100, 20)
        M = st.slider("Puntos de muestreo por período", 2000, 20000, 6000, step=1000)

    tipo = st.selectbox(
        "Tipo de señal",
        (
            "Onda cuadrada",
            "Onda triangular",
            "Diente de sierra",
            "Pulso",
            "Escalón",
            "Personalizada (expresión en t)",
        ),
    )

    params = {}
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
            "Ejemplos de expresiones válidas: `sen(2*pi*t/T)`, `abs(sen(3*t))`, `heaviside(sen(t), 0)`, `exp(-abs(t))`"
        )
        expr = st.text_input("f(t) en un período (usá 't' como variable; funciones en español)", value="sen(2*pi*t/T)")

    st.markdown("---")
    st.subheader("¿Qué significan los parámetros?")
    st.markdown(
        """
        - **T**: Período de la función periódica.
        - **N**: Cantidad de términos en la serie de Fourier que se usarán en la aproximación.
        - **M**: Puntos de muestreo para integrar numéricamente (más puntos = mejor precisión, más tiempo).
        - **Coeficientes**: 
            - \(a_0 = \frac{1}{L}\int_{-L}^{L} f(x)\,dx\), con \(L = T/2\).
            - \(a_n = \frac{1}{L}\int_{-L}^{L} f(x)\cos\!\left(\frac{n\pi x}{L}\right) dx\), \(b_n = \frac{1}{L}\int_{-L}^{L} f(x)\sin\!\left(\frac{n\pi x}{L}\right) dx\).
        """
    )

with col_der:
    # Dominio de integración y muestreo
    L = T / 2
    t = np.linspace(-L, L, int(M), endpoint=False)

    # Construcción de f(t)
    if tipo == "Onda cuadrada":
        f = onda_cuadrada(t, T, **params)
    elif tipo == "Onda triangular":
        f = onda_triangular(t, T, **params)
    elif tipo == "Diente de sierra":
        f = diente_de_sierra(t, T, **params)
    elif tipo == "Pulso":
        f = pulso(t, T, **params)
    elif tipo == "Escalón":
        f = escalon(t, T, **params)
    else:
        expr_local = expr.replace("T", str(T))  # permitir usar T en la expresión
        f = eval_personalizada(expr_local, t)

    # Coeficientes y reconstrucción
    a0, an, bn = coeficientes_fourier(f, t, T, N)
    S = serie_fourier(a0, an, bn, t, T)

    # Gráfico principal
    fig = plt.figure(figsize=(8, 4.5))
    plt.plot(t, f, label="f(t)")
    plt.plot(t, S, label=f"S_N(t) con N={N}")
    plt.title("Función original y aproximación por serie de Fourier")
    plt.xlabel("t")
    plt.ylabel("valor")
    plt.grid(True, alpha=0.3)
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    # Tabla de coeficientes (primeros 10 mostrados)
    st.subheader("Coeficientes (primeros 10)")
    k = np.arange(1, N + 1)
    tabla = np.column_stack([k, an, bn])
    import pandas as pd
    df = pd.DataFrame(tabla, columns=["n", "a_n", "b_n"])
    st.dataframe(df.head(10), use_container_width=True)

    # Fórmula en LaTeX
    st.subheader("Serie de Fourier (forma trigonométrica)")
    st.latex(r"f(x) \approx \frac{a_0}{2} + \sum_{n=1}^{N} \left[a_n\cos\left(\frac{n\pi x}{L}\right) + b_n\sin\left(\frac{n\pi x}{L}\right)\right],\quad L=\tfrac{T}{2}")

    # Descarga CSV
    st.download_button(
        label="Descargar coeficientes (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="coeficientes_fourier.csv",
        mime="text/csv",
    )

# ==========================
# Notas y ayuda
# ==========================
st.markdown("---")
st.markdown(
    """
    #### Notas
    - Para **señales personalizadas**, use `t` como variable y funciones en español: `sen`, `cos`, `tan`, `exp`, `log`, `sqrt`, `abs`, `signo`, `heaviside(x, 0.5)`, `fraccion(t/T)`, etc.
    - La integración numérica usa el **método del trapecio** sobre un período completo.
    - Si su función tiene **saltos**, la aproximación mostrará el **fenómeno de Gibbs** cerca de las discontinuidades.
    - Puede aumentar **M** y **N** para mejorar la precisión (con mayor costo computacional).
    """
)

# --- Modo simple: graficar f(t) ---
st.markdown("---")
st.header("🖊️ Modo simple: graficar f(t)")
with st.form("grafica_simple"):
    colA, colB, colC = st.columns([1, 1, 1])
    with colA:
        t_min = st.number_input("t mínimo", value=-10.0)
    with colB:
        t_max = st.number_input("t máximo", value=10.0)
    with colC:
        M = st.slider("Puntos", min_value=200, max_value=20000, value=2000, step=200)
    st.info("Usá `t` como variable. Si usás `T`, definila abajo.")
    expr = st.text_input("f(t) =", value="sen(t) + 0.5*cos(3*t)")
    usar_T = st.checkbox("Definir T (opcional)", value=True)
    if usar_T:
        T_val = st.number_input("T", value=float(2*np.pi))
    submitted_simple = st.form_submit_button("Graficar f(t)")

if 'submitted_simple' in locals() and submitted_simple:
    if t_max <= t_min:
        st.error("t máximo debe ser mayor que t mínimo.")
    else:
        try:
            t_simple = np.linspace(t_min, t_max, int(M))
            vars_extra = {"T": T_val} if usar_T else None
            f_simple = eval_personalizada(expr, t_simple, vars_extra)
            fig_simple = plt.figure(figsize=(9, 4.8))
            plt.plot(t_simple, f_simple, label="f(t)")
            plt.xlabel("t")
            plt.ylabel("f(t)")
            plt.title("Gráfica de la función ingresada")
            plt.grid(True, alpha=0.35)
            plt.legend()
            st.pyplot(fig_simple, clear_figure=True)
        except Exception as e:
            st.error(str(e))


