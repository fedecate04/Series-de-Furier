# app_fourier_simple_es.py
# Calculadora simple de Serie de Fourier (estilo Symbolab) — Todo en español
# Autor: CARRASCO SERGIO FEDERICO
# Requisitos: streamlit, numpy, matplotlib, pandas (para CSV opcional)

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

st.set_page_config(page_title="Serie de Fourier (ES) — Simple", page_icon="📈", layout="wide")

# ==========================
# Utilidades y funciones en español
# ==========================
PI = np.pi

def fraccion(u):
    return u - np.floor(u)

def fase_01(x):
    """Fase normalizada en [0,1) a partir de x en radianes (período 2π)."""
    return (x / (2*PI)) - np.floor(x / (2*PI))

# Señales elementales parametrizadas por fase x (en radianes)

def cuadrada(x, duty=0.5, A=1.0):
    ph = fase_01(x)
    y = np.where(ph < duty, A, -A)
    return y

def triangular(x, A=1.0):
    ph = fase_01(x)
    y = 4*A*np.abs(ph - 0.5) - A  # en [-A, A]
    return y

def sierra(x, width=1.0, A=1.0):
    # sierra simétrica simple [-A, A]; ignoramos width para mantenerlo simple
    ph = fase_01(x)
    y = 2*A*ph - A
    return y

def pulso(x, duty=0.5, A=1.0):
    ph = fase_01(x)
    return np.where(ph < duty, A, 0.0)

def escalon(x, h0=0.5):
    return np.heaviside(x, h0)

def si(cond, v1, v2):
    return np.where(cond, v1, v2)

# Mapeo de nombres en español para expresiones
ENTORNO = {
    # constantes y variable (t se setea en runtime)
    "pi": np.pi,
    "e": np.e,
    "t": None,
    # trigonométricas en español
    "seno": np.sin,
    "coseno": np.cos,
    "tan": np.tan,
    "arcseno": np.arcsin,
    "arccos": np.arccos,
    "arctan": np.arctan,
    # utilitarias
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "raiz": np.sqrt,
    "abs": np.abs,
    "signo": np.sign,
    "heaviside": np.heaviside,
    "min": np.minimum,
    "max": np.maximum,
    "mod": np.mod,
    "fraccion": fraccion,
    # señales por fase
    "cuadrada": cuadrada,
    "triangular": triangular,
    "sierra": sierra,
    "pulso": pulso,
    "escalon": escalon,
    # condicional
    "si": si,
    "donde": si,
}

# ==========================
# Cálculo de coeficientes de Fourier (período T, L=T/2)
# ==========================

def coeficientes_fourier(f_vals, t_grid, T, N):
    L = T/2
    a0 = (1/L) * np.trapz(f_vals, t_grid)
    an = np.zeros(N)
    bn = np.zeros(N)
    for n in range(1, N+1):
        cosn = np.cos(n*np.pi*t_grid/L)
        sinn = np.sin(n*np.pi*t_grid/L)
        an[n-1] = (1/L) * np.trapz(f_vals * cosn, t_grid)
        bn[n-1] = (1/L) * np.trapz(f_vals * sinn, t_grid)
    return a0, an, bn

def serie_fourier(a0, an, bn, t_grid, T):
    L = T/2
    S = np.full_like(t_grid, a0/2.0)
    for n in range(1, len(an)+1):
        S += an[n-1]*np.cos(n*np.pi*t_grid/L) + bn[n-1]*np.sin(n*np.pi*t_grid/L)
    return S

# ==========================
# Evaluador seguro de expresiones f(t)
# ==========================

def eval_ft(expr: str, t: np.ndarray, extra: dict | None = None):
    ns = dict(ENTORNO)
    ns["t"] = t
    if extra:
        ns.update(extra)
    try:
        return eval(expr, {"__builtins__": {}}, ns)
    except Exception as e:
        raise ValueError(f"Error al evaluar f(t): {e}")

# ==========================
# UI — Simple y directo
# ==========================

st.title("📈 Serie de Fourier — Calculadora simple (ES)")
st.caption("Ingresá f(t), el período T y la cantidad de términos N. La app calcula a₀, aₙ, bₙ y grafica f(t) y S_N(t).")

# Parámetros básicos
colA, colB, colC = st.columns([1,1,1])
with colA:
    T = st.number_input("Período T", value=float(2*np.pi), min_value=1e-6, step=0.1, format="%.6f")
with colB:
    N = st.slider("Términos N", min_value=1, max_value=80, value=20)
with colC:
    M = st.select_slider("Muestreo (puntos)", options=[1000, 2000, 4000, 6000, 8000, 12000], value=6000)

# Plantillas rápidas
preset = st.selectbox(
    "Plantillas (opcional)",
    ["— Ninguna —", "Seno básico", "Coseno básico", "Cuadrada ±1 (50%)", "Triangular ±1", "Sierra ±1", "Pulso 0/1 (25%)"],
    index=0,
)

default_expr = "seno(2*pi*(1/T)*t)"
if preset == "Seno básico":
    default_expr = "seno(2*pi*(1/T)*t)"
elif preset == "Coseno básico":
    default_expr = "coseno(2*pi*(1/T)*t)"
elif preset == "Cuadrada ±1 (50%)":
    default_expr = "cuadrada(2*pi*(1/T)*t, duty=0.5)"
elif preset == "Triangular ±1":
    default_expr = "triangular(2*pi*(1/T)*t)"
elif preset == "Sierra ±1":
    default_expr = "sierra(2*pi*(1/T)*t)"
elif preset == "Pulso 0/1 (25%)":
    default_expr = "pulso(2*pi*(1/T)*t, duty=0.25)"

expr = st.text_input(
    "f(t) =", value=default_expr,
    help=(
        "Usá `t` y, si querés, `T`. Funciones: seno, coseno, tan, exp, log, sqrt/raiz, abs, signo, "
        "heaviside(x,h0), fraccion, si(cond,a,b) o donde(cond,a,b). Señales por fase: cuadrada(x,duty), triangular(x), sierra(x), pulso(x,duty), escalon(x)."
    )
)

if st.button("Calcular y graficar", type="primary"):
    # Mallado en un período [-L, L)
    L = T/2
    t_grid = np.linspace(-L, L, int(M), endpoint=False)
    try:
        f_vals = eval_ft(expr, t_grid, extra={"T": T})
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Coeficientes y reconstrucción
    a0, an, bn = coeficientes_fourier(f_vals, t_grid, T, N)
    S = serie_fourier(a0, an, bn, t_grid, T)

    # Gráfico
    fig = plt.figure(figsize=(9, 4.8))
    plt.plot(t_grid, f_vals, label="f(t)")
    plt.plot(t_grid, S, label=f"S_N(t), N={N}")
    plt.xlabel("t")
    plt.ylabel("valor")
    plt.title("Función y aproximación por Serie de Fourier")
    plt.grid(True, alpha=0.35)
    plt.legend()
    st.pyplot(fig, clear_figure=True)

    # Resultados
    st.markdown("### Coeficientes")
    k = np.arange(1, N+1)
    df = pd.DataFrame({"n": k, "a_n": an, "b_n": bn})
    col1, col2 = st.columns([1,2])
    with col1:
        st.metric("a₀", f"{a0:.6f}")
    with col2:
        st.dataframe(df.head(12), use_container_width=True)

    st.download_button(
        label="Descargar coeficientes (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="coeficientes_fourier.csv",
        mime="text/csv",
    )

    st.caption(
        "Nota: cerca de discontinuidades puede verse el fenómeno de Gibbs. Aumentar N y el muestreo ayuda, aunque el sobreimpulso no desaparece."
    )

# ==========================
# Ayuda mínima
# ==========================
with st.expander("Ayuda rápida (funciones y ejemplos)"):
    st.markdown(
        """
        **Ejemplos para pegar en `f(t)`**:
        - `seno(2*pi*(1/T)*t)`
        - `coseno(4*pi*(1/T)*t)`
        - `cuadrada(2*pi*(1/T)*t, duty=0.5)`  
        - `pulso(2*pi*(1/T)*t, duty=0.25)`
        - `triangular(2*pi*(1/T)*t)`
        - `sierra(2*pi*(1/T)*t)`
        - `si(t < 0, -1, 1)`  (por tramos en un período)
        """
    )
