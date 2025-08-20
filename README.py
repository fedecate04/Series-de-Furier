
# -*- coding: utf-8 -*-
# Fourier App – UTN | Ingeniería
# Autor: Carrasco Sergio Federico
# Email: fedeneu@gmail.com
# -----------------------------------------------------------
# Ejecutar con: streamlit run app.py
# Requiere: ver requirements.txt

import io
import math
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from scipy.signal import sawtooth, square
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from PIL import Image

# =============== ESTILO BÁSICO ===============
st.set_page_config(
    page_title="Fourier App - UTN",
    page_icon="🎛️",
    layout="wide"
)

# CSS suave
st.markdown(
    """
    <style>
      .small { font-size:0.9rem; color:#555; }
      .muted { color:#666; }
      .center { text-align:center; }
      .tight { margin-top:0.25rem; margin-bottom:0.25rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 8px; }
      .stTabs [data-baseweb="tab"] { padding-top: 8px; padding-bottom: 8px; }
    </style>
    """,
    unsafe_allow_html=True
)

# =============== UTILIDADES ===============
# =============== UTILIDADES ===============
SAFE_NS = {
    # constantes
    "pi": np.pi, "e": np.e,
    # funciones numpy (inglés)
    "sin": np.sin, "cos": np.cos, "tan": np.tan,
    "sinh": np.sinh, "cosh": np.cosh, "tanh": np.tanh,
    "arcsin": np.arcsin, "arccos": np.arccos, "arctan": np.arctan,
    "exp": np.exp, "log": np.log, "log10": np.log10,
    "sqrt": np.sqrt, "abs": np.abs, "sign": np.sign,
    "heaviside": np.heaviside, "where": np.where,
    "mod": np.mod, "floor": np.floor, "ceil": np.ceil,
    # señales scipy (inglés)
    "sawtooth": sawtooth,
    "square": square,
    # ===== alias en español =====
    "seno": np.sin,
    "coseno": np.cos,
    "tangente": np.tan,
    "senh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "arcseno": np.arcsin,
    "arccoseno": np.arccos,
    "arctangente": np.arctan,
    "exponencial": np.exp,
    "logaritmo": np.log,
    "logaritmo10": np.log10,
    "raiz": np.sqrt,
    "valor_absoluto": np.abs,
    "signo": np.sign,
    "escalon": np.heaviside,      # escalon(x, 0.5) para valor en 0
    "si": np.where,                # si(cond, v1, v2) == where(cond, v1, v2)
    "modulo": np.mod,
    "piso": np.floor,
    "techo": np.ceil,
    # ondas en español
    "sierra": sawtooth,            # diente de sierra
    "triangular": lambda x, width=0.5: sawtooth(x, width=width),
    "cuadrada": square,
    "pulso": lambda x, duty=0.5: (square(x, duty=duty) + 1)/2.0
}

def eval_func_expression(expr: str, t: np.ndarray, extra_ns: dict = None) -> np.ndarray:
    """Evalúa expresión f(t) de forma segura usando un namespace controlado."""
    ns = dict(SAFE_NS)
    if extra_ns:
        ns.update(extra_ns)
    ns["t"] = t
    try:
        y = eval(expr, {"__builtins__": {}}, ns)
    except Exception as ex:
        raise ValueError(f"Error evaluando la expresión: {ex}")
    if np.isscalar(y):
        y = np.full_like(t, float(y))
    return np.asarray(y, dtype=float)

def trapz_periodic_coef(f_vals, basis_vals, T, t_grid):
    """Coeficiente por integración numérica (trapecio) en un periodo [0,T]."""
    integrand = f_vals * basis_vals
    # integral_0^T f(t)*basis dt
    I = np.trapz(integrand, t_grid)
    # factor 2/T por convención trigonométrica
    return (2.0 / T) * I

def compute_fourier_coeffs(f_t, T, N, t_grid, symmetry_hint=None):
    """
    Calcula a0, an, bn en [0,T] con definición trigonométrica:
    a0 = (2/T)∫_0^T f(t) dt
    an = (2/T)∫_0^T f(t)cos(n*w0*t) dt
    bn = (2/T)∫_0^T f(t)sin(n*w0*t) dt
    """
    w0 = 2.0 * np.pi / T
    a0 = (2.0 / T) * np.trapz(f_t, t_grid)
    an = np.zeros(N+1)  # index 0 no se usa
    bn = np.zeros(N+1)

    # aplicar hint de simetría: "par" => bn=0; "impar" => an=0 (excepto a0)
    use_par = (symmetry_hint == "Par")
    use_impar = (symmetry_hint == "Impar")

    for n in range(1, N+1):
        if not use_impar:
            an[n] = trapz_periodic_coef(f_t, np.cos(n*w0*t_grid), T, t_grid)
        if not use_par:
            bn[n] = trapz_periodic_coef(f_t, np.sin(n*w0*t_grid), T, t_grid)

    return a0, an, bn, w0

def reconstruct_signal(a0, an, bn, w0, t_grid, Nsum):
    """S_N(t) = a0/2 + sum_{n=1..N} an cos(n w0 t) + bn sin(n w0 t)"""
    s = np.full_like(t_grid, a0/2.0)
    for n in range(1, Nsum+1):
        if n < len(an):
            s += an[n]*np.cos(n*w0*t_grid)
        if n < len(bn):
            s += bn[n]*np.sin(n*w0*t_grid)
    return s

def an_bn_to_amp_phase(an, bn):
    """Convierte (an,bn) a (A_n, phi_n) con convención: an cos + bn sin = A cos(nw0 t + phi)."""
    N = len(an) - 1
    A = np.zeros(N+1)
    phi = np.zeros(N+1)
    for n in range(1, N+1):
        A[n] = np.hypot(an[n], bn[n])
        # an = A cos(phi), bn = A sin(phi) con signo para consistencia:
        phi[n] = np.arctan2(-bn[n], an[n])  # de modo que an cos + bn sin = A cos(nwt + phi)
    return A, phi

def build_pdf_bytes(
        title, subtitle, params_text,
        fig_signal_bytes, fig_spectrum_bytes,
        table_df: pd.DataFrame,
        logo_path: Path | None = None
    ) -> bytes:
    """Genera un PDF con ReportLab y devuelve bytes."""
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    W, H = A4
    margin = 36

    # Logo (si existe)
    y_cursor = H - margin
    if logo_path and logo_path.exists():
        try:
            img = Image.open(logo_path)
            aspect = img.width / img.height
            logo_w = 120
            logo_h = logo_w / aspect
            c.drawImage(ImageReader(img), margin, y_cursor - logo_h, width=logo_w, height=logo_h, mask='auto')
        except Exception:
            pass

    # Título
    c.setFont("Helvetica-Bold", 14)
    c.drawString(margin + 140, H - margin - 10, title)
    c.setFont("Helvetica", 11)
    c.drawString(margin + 140, H - margin - 26, subtitle)

    # Parámetros
    c.setFont("Helvetica", 10)
    text_obj = c.beginText()
    text_obj.setTextOrigin(margin, H - margin - 70)
    for line in params_text.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    # Gráfico señal + suma
    if fig_signal_bytes:
        c.drawString(margin, H - 360, "Señal y suma parcial S_N(t):")
        c.drawImage(ImageReader(io.BytesIO(fig_signal_bytes)), margin, H - 680, width=W - 2*margin, height=300, preserveAspectRatio=True, mask='auto')

    # Nueva página para espectro y tabla
    c.showPage()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, H - margin - 10, "Espectro de armónicos")
    if fig_spectrum_bytes:
        c.drawImage(ImageReader(io.BytesIO(fig_spectrum_bytes)), margin, H - 420, width=W - 2*margin, height=320, preserveAspectRatio=True, mask='auto')

    # Tabla de coeficientes (primeros 15 por espacio)
    c.setFont("Helvetica-Bold", 11)
    c.drawString(margin, H - 450, "Coeficientes (primeros 15):")
    c.setFont("Helvetica", 9)

    tbl = table_df.head(15)
    col_names = list(tbl.columns)
    rows = [col_names] + [[str(v) for v in row] for row in tbl.values]
    x0, y0 = margin, H - 470
    row_h = 14
    col_w = (W - 2*margin) / len(col_names)

    for i, row in enumerate(rows):
        y = y0 - i*row_h
        for j, cell in enumerate(row):
            c.drawString(x0 + j*col_w + 2, y, cell[:22])  # recorte simple
    c.showPage()
    c.save()
    return buffer.getvalue()

def fig_to_png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()

# =============== SIDEBAR (ENTRADAS) ===============
with st.sidebar:
    st.markdown("### Parámetros de entrada")
    st.info("**CREADOR:** CARRASCO SERGIO FEDERICO\n\n**EMAIL:** fedeneu@gmail.com")

    input_mode = st.radio("Tipo de señal", ["Analítica (expresión)", "Datos (CSV)"], index=0)

    colT = st.columns(2)
    with colT[0]:
        T = st.number_input("Período T", min_value=1e-6, value=2*np.pi, step=0.1, format="%.6f")
    with colT[1]:
        N = st.number_input("Nº de armónicos (N)", min_value=1, value=20, step=1)

    M = st.slider("Puntos de muestreo por período (M)", 200, 5000, 2000, step=100)

    symmetry_hint = st.selectbox("Sugerir simetría (opcional)", ["Ninguna", "Par", "Impar"], index=0)

    st.markdown("---")
    st.markdown("#### Biblioteca de señales (atajo)")
    preset = st.selectbox(
        "Plantillas",
        [
            "— Ninguna —",
            "Cuadrada (±1, duty 50%)",
            "Triangular (±1)",
            "Diente de sierra (±1)",
            "Pulso (0/1) con duty",
            "Coseno base (amplitud 1)"
        ],
        index=0
    )
    duty = st.slider("Duty (para cuadrada/pulso/sierra)", 1, 99, 50) / 100.0

# =============== ENCABEZADO PRINCIPAL ===============
left, center, right = st.columns([1,2,1])

with center:
    # Cargar logo
    logo_candidates = ["logoutn.png", "logoutn.PNG", "logoutn"]
    logo_path = None
    for cand in logo_candidates:
        p = Path(cand)
        if p.exists():
            logo_path = p
            break
    if logo_path is not None:
        st.image(str(logo_path), width=160, use_container_width=False)

    st.markdown("<h2 class='center tight'>UNIVERSIDAD TECNOLÓGICA NACIONAL</h2>", unsafe_allow_html=True)
    st.markdown("<div class='center muted tight'>Herramienta profesional de Series de Fourier</div>", unsafe_allow_html=True)

# =============== EXPANDER DE AYUDA ===============
with st.expander("USO DE LA APLICACIÓN", expanded=False):
    st.markdown(
        """
**1) Elegí el tipo de señal**:  
- *Analítica (expresión)*: escribí `f(t)` (podés usar `sin`, `cos`, `sawtooth`, `square`, `heaviside`, `where`, etc.).  
- *Datos (CSV)*: subí un archivo con dos columnas `t` y `y` (un período).  

**2) Definí el período `T` y el número de armónicos `N`**.  
**3) Opcional**: seleccioná una *plantilla* (cuadrada, triangular, etc.).  
**4) Calculá**: la app integra en [0, T] y devuelve `a0, an, bn`, la reconstrucción `S_N(t)` y el **espectro de armónicos**.  
**5) Exportá**: generá un **PDF** con los resultados y gráficos.  

> **Tips:**  
> - Si la señal es **par**, los `b_n` tienden a 0; si es **impar**, los `a_n` tienden a 0.  
> - A mayor suavidad de `f(t)`, más rápido decaen los armónicos.
        """
    )

# =============== ENTRADA / SEÑAL ===============
# Malla temporal en [0, T] con M puntos
t = np.linspace(0.0, float(T), int(M), endpoint=False)

st.markdown("### Elegí el modo de entrada")
entrada_modo = st.radio(
    "Cómo querés definir f(t):",
    ["Constructor simple (recomendado)", "Expresión avanzada (en español)"],
    index=0,
    help="El constructor te deja combinar ondas predefinidas; la expresión te permite escribir f(t) con nombres en español."
)

def componente(t, tipo, A, f_hz, fase, duty, offset):
    """Genera una componente según 'tipo'."""
    w = 2*np.pi*f_hz
    x = w*t + fase
    if tipo == "Seno":
        y = A * np.sin(x)
    elif tipo == "Coseno":
        y = A * np.cos(x)
    elif tipo == "Cuadrada":
        y = A * square(x)
    elif tipo == "Triangular":
        y = A * sawtooth(x, width=0.5)
    elif tipo == "Diente de sierra":
        y = A * sawtooth(x, width=duty)    # duty controla el ancho de subida
    elif tipo == "Pulso":
        y = A * ((square(x, duty=duty)+1)/2.0)  # 0..A
    elif tipo == "Constante":
        y = A * np.ones_like(t)
    else:
        y = np.zeros_like(t)
    return y + offset

f_vals = None
expr_used = None
data_uploaded = False

if entrada_modo == "Constructor simple (recomendado)":
    st.markdown("#### Constructor de señal (sumador de componentes)")
    st.caption("Consejo: usá la frecuencia fundamental f₀ = 1/T si querés una serie en armónicos enteros.")

    f0 = 1.0/float(T)
    n_comp = st.slider("Número de componentes", 1, 6, 2)
    suma = np.zeros_like(t)

    # Crear controles por componente
    for i in range(n_comp):
        st.markdown(f"**Componente {i+1}**")
        c1, c2, c3, c4 = st.columns([1.2,1,1,1])
        with c1:
            tipo = st.selectbox(
                "Tipo",
                ["Seno", "Coseno", "Cuadrada", "Triangular", "Diente de sierra", "Pulso", "Constante"],
                key=f"tipo_{i}"
            )
        with c2:
            A = st.number_input("Amplitud A", value=1.0, step=0.1, format="%.6f", key=f"A_{i}")
            fase = st.number_input("Fase φ (rad)", value=0.0, step=0.1, format="%.6f", key=f"fase_{i}")
        with c3:
            f_hz = st.number_input("Frecuencia (Hz)", value=float(f0), step=float(f0), format="%.6f", key=f"f_{i}")
            offset = st.number_input("Offset", value=0.0, step=0.1, format="%.6f", key=f"off_{i}")
        with c4:
            duty = st.slider("Duty", 1, 99, 50, key=f"duty_{i}")/100.0 if tipo in ("Diente de sierra","Pulso") else 0.5
            st.write("")  # espacio

        suma += componente(t, tipo, A, f_hz, fase, duty, offset)

    f_vals = suma
    st.success("Señal generada con el constructor.")

    with st.expander("Ver fórmula equivalente (aproximada)"):
        # Texto pedagógico simple (no exacto si hay triángulo/sierra/pulso)
        st.markdown(
            "La señal se construyó sumando componentes. Si todas fueran senos/cosenos, "
            "una forma típica sería `Σ A_i * cos(2π f_i t + φ_i) + offset_i`."
        )

else:
    # ======= Modo EXPRESIÓN (en español) =======
    st.markdown("#### f(t) – Expresión (en español)")
    st.markdown(
        """
**Ejemplos:**
- `seno(2*pi*(1/T)*t)`  
- `si(t<T/2, 1, -1)`  ← (equivale a where)  
- `cuadrada(2*pi*(1/T)*t, duty=0.5)`  
- `pulso(2*pi*(1/T)*t, duty=0.25)`  
- `triangular(2*pi*(1/T)*t)` o `sierra(2*pi*(1/T)*t, width=0.3)`  
- `escalon(t-0.5*T, 0.5)`  (valor 0.5 en discontinuidad)
        """
    )

    # plantillas también disponibles aquí
    preset = st.selectbox(
        "Plantillas rápidas",
        ["— Ninguna —", "Cuadrada (±1, duty 50%)", "Triangular (±1)", "Diente de sierra (±1)", "Pulso (0/1) duty 25%", "Coseno base (A=1)"],
        index=0
    )
    duty_q = st.slider("Duty para plantillas (si aplica)", 1, 99, 50)/100.0
    default_expr = ""
    if preset == "Cuadrada (±1, duty 50%)":
        default_expr = f"cuadrada(2*pi*(1/T)*t, duty=0.5)"
    elif preset == "Triangular (±1)":
        default_expr = f"triangular(2*pi*(1/T)*t)"
    elif preset == "Diente de sierra (±1)":
        default_expr = f"sierra(2*pi*(1/T)*t, width={duty_q})"
    elif preset == "Pulso (0/1) duty 25%":
        default_expr = f"pulso(2*pi*(1/T)*t, duty=0.25)"
    elif preset == "Coseno base (A=1)":
        default_expr = f"coseno(2*pi*(1/T)*t)"

    expr = st.text_area(
        "Ingrese la expresión de f(t)",
        value=default_expr,
        height=90,
        help="Usá nombres en español: seno, coseno, si(cond, v1, v2), escalon(x,0.5), cuadrada(x,duty), pulso(x,duty), triangular(x), sierra(x,width)."
    )
    if expr.strip():
        try:
            f_vals = eval_func_expression(expr, t, extra_ns={"T": T})
            expr_used = expr
            st.success("Expresión evaluada correctamente.")
        except Exception as ex:
            st.error(str(ex))

# (Opcional) Entrada por CSV se mantiene igual que antes:
st.markdown("#### O cargar datos (CSV con columnas t,y) en lugar de definir f(t)")
up = st.file_uploader("Archivo CSV", type=["csv"])
if up is not None:
    try:
        df = pd.read_csv(up)
        cols = [c.lower() for c in df.columns]
        col_t = df.columns[cols.index("t")] if "t" in cols else df.columns[0]
        col_y = df.columns[cols.index("y")] if "y" in cols else df.columns[1]
        t_raw = df[col_t].to_numpy(dtype=float)
        y_raw = df[col_y].to_numpy(dtype=float)
        if t_raw.min() < 0:  # normalizar inicio en 0
            t_raw = t_raw - t_raw.min()
        span = t_raw.max() - t_raw.min()
        if abs(span - T) > 1e-9:
            t_norm = (t_raw - t_raw.min()) * (T / span)
            y_interp = np.interp(t, t_norm, y_raw)
        else:
            y_interp = np.interp(t, t_raw, y_raw)
        f_vals = y_interp
        expr_used = None
        data_uploaded = True
        st.success("Datos cargados y remuestreados correctamente.")
    except Exception as ex:
        st.error(f"Error leyendo CSV: {ex}")



# =============== CÁLCULO Y PESTAÑAS DE RESULTADOS ===============
tabs = st.tabs(["Resultados", "Coeficientes", "Espectro", "Armónicos", "Exportar PDF"])

if f_vals is None:
    with tabs[0]:
        st.warning("Ingresá una señal para calcular (expresión o CSV).")
else:
    # Calcular coeficientes
    a0, an, bn, w0 = compute_fourier_coeffs(
        f_vals, T=float(T), N=int(N), t_grid=t,
        symmetry_hint=(symmetry_hint if symmetry_hint != "Ninguna" else None)
    )
    A, phi = an_bn_to_amp_phase(an, bn)

    # ---------- TAB 1: Resultados ----------
    with tabs[0]:
        col_info = st.columns(3)
        with col_info[0]:
            st.metric("Período T", f"{T:.6g}")
        with col_info[1]:
            st.metric("Frecuencia f₀", f"{1.0/float(T):.6g} Hz")
        with col_info[2]:
            st.metric("ω₀", f"{w0:.6g} rad/s")

        Nsum = st.slider("Armonicos en la suma parcial S_N(t)", 1, int(N), min(10, int(N)))
        sN = reconstruct_signal(a0, an, bn, w0, t, Nsum)

        fig1, ax1 = plt.subplots(figsize=(8, 3.5))
        ax1.plot(t, f_vals, label="f(t)")
        ax1.plot(t, sN, linestyle="--", label=f"S_{Nsum}(t)")
        ax1.set_xlabel("t")
        ax1.set_ylabel("Amplitud")
        ax1.set_title("Señal vs suma parcial")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        st.pyplot(fig1, use_container_width=True)

        # Error L2 normalizado
        err = np.sqrt(np.trapz((f_vals - sN)**2, t) / np.trapz((f_vals)**2 + 1e-14, t))
        st.caption(f"Error L2 relativo (aprox.): {err:.3e}")

        if expr_used:
            st.code(f"f(t) = {expr_used}", language="python")
        elif data_uploaded:
            st.caption("Señal proveniente de CSV (remuestreada en [0, T]).")

    # ---------- TAB 2: Coeficientes ----------
    with tabs[1]:
        n_idx = np.arange(0, int(N)+1)
        df_coeffs = pd.DataFrame({
            "n": n_idx,
            "a_n": [a0] + list(an[1:]) if len(an) > 1 else [a0],
            "b_n": [0.0] + list(bn[1:]) if len(bn) > 1 else [0.0],
            "A_n": [0.0] + list(A[1:]) if len(A) > 1 else [0.0],
            "phi_n (rad)": [0.0] + list(phi[1:]) if len(phi) > 1 else [0.0],
            "freq (Hz)": [0.0] + list((np.arange(1, int(N)+1))/float(T))
        })
        st.dataframe(df_coeffs, use_container_width=True)

        csv = df_coeffs.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar coeficientes (CSV)", csv, file_name="coeficientes_fourier.csv", mime="text/csv")

    # ---------- TAB 3: Espectro ----------
    with tabs[2]:
        fig2, ax2 = plt.subplots(figsize=(8, 3.5))
        freqs = (np.arange(1, int(N)+1))/float(T)
        ax2.stem(freqs, A[1:], use_line_collection=True)
        ax2.set_xlabel("Frecuencia (Hz) = n / T")
        ax2.set_ylabel("Amplitud A_n = sqrt(a_n^2 + b_n^2)")
        ax2.set_title("Espectro de líneas")
        ax2.grid(True, alpha=0.3)
        st.pyplot(fig2, use_container_width=True)

    # ---------- TAB 4: Armónicos ----------
    with tabs[3]:
        st.markdown("Active/desactive armónicos para ver su contribución.")
        picks = st.multiselect(
            "Seleccionar armónicos a incluir",
            options=list(range(1, int(N)+1)),
            default=list(range(1, min(int(N), 10)+1))
        )
        s_sel = np.full_like(t, a0/2.0)
        for n in picks:
            s_sel += an[n]*np.cos(n*w0*t) + bn[n]*np.sin(n*w0*t)

        fig3, ax3 = plt.subplots(figsize=(8, 3.5))
        ax3.plot(t, f_vals, label="f(t)")
        ax3.plot(t, s_sel, linestyle="--", label=f"S_{{{picks}}}(t)")
        ax3.set_xlabel("t")
        ax3.set_ylabel("Amplitud")
        ax3.set_title("Reconstrucción con armónicos seleccionados")
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        st.pyplot(fig3, use_container_width=True)

    # ---------- TAB 5: Exportar PDF ----------
    with tabs[4]:
        st.markdown("### Exportar informe PDF")
        # Capturar figuras en bytes
        fig1_bytes = fig_to_png_bytes(fig1)
        fig2_bytes = fig_to_png_bytes(fig2)

        # Texto de parámetros
        params_text = (
            f"Período T: {T:.6g}\n"
            f"Frecuencia fundamental f0: {1.0/float(T):.6g} Hz\n"
            f"ω0: {w0:.6g} rad/s\n"
            f"N de armónicos calculados: {int(N)}\n"
            f"Sugerencia de simetría: {symmetry_hint}\n"
            f"Error L2 relativo aprox.: {err:.3e}\n"
        )
        title = "UNIVERSIDAD TECNOLÓGICA NACIONAL"
        subtitle = "Herramienta de Series de Fourier – Informe de resultados"

        pdf_bytes = build_pdf_bytes(
            title=title,
            subtitle=subtitle,
            params_text=params_text,
            fig_signal_bytes=fig1_bytes,
            fig_spectrum_bytes=fig2_bytes,
            table_df=df_coeffs,
            logo_path=logo_path
        )

        st.download_button(
            "Descargar PDF",
            data=pdf_bytes,
            file_name="fourier_resultados.pdf",
            mime="application/pdf"
        )

# Pie de página
st.markdown("---")
st.markdown(
    "<div class='center small'>© 2025 · Fourier App – UTN · Creador: Carrasco Sergio Federico · "
    "Contacto: <a href='mailto:fedeneu@gmail.com'>fedeneu@gmail.com</a></div>",
    unsafe_allow_html=True
)
