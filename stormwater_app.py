import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Справочник коэффициентов шероховатости ---
materials_n = {
    "Бетон": 0.013,
    "ПВХ": 0.009,
    "Чугун": 0.014,
    "Асбестоцемент": 0.011,
    "Сталь": 0.012
}

st.set_page_config(page_title="Калькулятор ливневых стоков", layout="wide")
st.title("💧 Калькулятор расхода ливневых стоков (Маннинг)")

# --- Функции ---
def hydraulic_params_circular(D, h):
    r = D / 2
    theta = 2 * np.arccos((r - h) / r)
    A = (r**2 / 2) * (theta - np.sin(theta))  # площадь
    P = r * theta  # смоченный периметр
    return A, P

def hydraulic_params_rectangular(b, h):
    A = b * h
    P = b + 2*h
    return A, P

def manning_flow(A, P, S, n):
    R = A / P
    V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
    Q = A * V
    return R, V, Q

# --- Ввод общих параметров ---
shape = st.selectbox("Форма трубы", ["Круглая", "Прямоугольная"])
material = st.selectbox("Материал трубы", list(materials_n.keys()))
n = materials_n[material]

num_segments = st.number_input("Количество сегментов трубы", min_value=1, max_value=10, value=1, step=1)

segments = []
for i in range(num_segments):
    st.markdown(f"### Сегмент {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        if shape == "Круглая":
            D = st.number_input(f"Диаметр сегмента {i+1}, м", value=0.1, step=0.01, format="%.3f", key=f"D{i}")
            h = st.number_input(f"Высота заполнения h сегмента {i+1}, м", value=0.07, step=0.01, format="%.3f", key=f"h{i}")
            geom = (D, h)
        else:
            b = st.number_input(f"Ширина сегмента {i+1}, м", value=0.2, step=0.01, format="%.3f", key=f"b{i}")
            h = st.number_input(f"Высота заполнения h сегмента {i+1}, м", value=0.15, step=0.01, format="%.3f", key=f"h{i}")
            geom = (b, h)
    with col2:
        top1 = st.number_input(f"Высотная отметка верха 1-го колодца (сегмент {i+1}), м", value=246.0, step=0.01, key=f"top1_{i}")
        depth1 = st.number_input(f"Глубина 1-го колодца (сегмент {i+1}), м", value=5.55, step=0.01, key=f"depth1_{i}")
        top2 = st.number_input(f"Высотная отметка верха 2-го колодца (сегмент {i+1}), м", value=245.0, step=0.01, key=f"top2_{i}")
        depth2 = st.number_input(f"Глубина 2-го колодца (сегмент {i+1}), м", value=5.57, step=0.01, key=f"depth2_{i}")
        length = st.number_input(f"Длина сегмента {i+1}, м", value=73.2, step=0.1, key=f"length{i}")
    segments.append((geom, h, top1, depth1, top2, depth2, length))

# --- Расчёт по сегментам ---
results = []
for idx, (geom, h, top1, depth1, top2, depth2, length) in enumerate(segments):
    invert1 = top1 - depth1
    invert2 = top2 - depth2
    S = (invert1 - invert2) / length if length > 0 else 0.0001

    if shape == "Круглая":
        D, h = geom
        A, P = hydraulic_params_circular(D, h)
    else:
        b, h = geom
        A, P = hydraulic_params_rectangular(b, h)

    R, V, Q = manning_flow(A, P, S, n)
    results.append((idx+1, A, P, R, V, Q, Q*1000))

    # --- Вывод графиков рядом ---
    col_left, col_right = st.columns([1,1.2])

    # Поперечное сечение
    with col_left:
        fig, ax = plt.subplots(figsize=(3, 3))
        if shape == "Круглая":
            D, _ = geom
            circle = plt.Circle((0, 0), D/2, color="lightgray")
            ax.add_patch(circle)
            ax.fill_between(
                np.linspace(-D/2, D/2, 200),
                -D/2, -D/2+h,
                color="blue", alpha=0.5
            )
            ax.set_xlim(-D/2-0.05, D/2+0.05)
            ax.set_ylim(-D/2-0.05, D/2+0.05)
        else:
            b, _ = geom
            ax.add_patch(plt.Rectangle((-b/2, -h), b, h, color="blue", alpha=0.5))
            ax.add_patch(plt.Rectangle((-b/2, -geom[1]), b, geom[1], fill=False, edgecolor="black"))
            ax.set_xlim(-b/2-0.05, b/2+0.05)
            ax.set_ylim(-geom[1]-0.05, 0.05)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Сегмент {idx+1}: поперечное сечение", fontsize=10)
        st.pyplot(fig)

    # Мини-график расхода
    with col_right:
        fig, ax = plt.subplots(figsize=(3.5, 3))
        ax.bar(["Q"], [Q], color="blue")
        ax.set_ylabel("Q, м³/с")
        ax.set_title(f"Расход сегмента {idx+1}", fontsize=10)
        st.pyplot(fig)

# --- Суммарные результаты ---
df = pd.DataFrame(results, columns=["Сегмент", "A (м²)", "P (м)", "R (м)", "V (м/с)", "Q (м³/с)", "Q (л/с)"])
total_Q = df["Q (м³/с)"].sum()

st.subheader("📊 Итоговые результаты")
st.dataframe(df.style.format({"A (м²)": "{:.6f}", "P (м)": "{:.4f}", "R (м)": "{:.4f}", "V (м/с)": "{:.3f}", "Q (м³/с)": "{:.5f}", "Q (л/с)": "{:.3f}"}))
st.write(f"**Суммарный расход линии: {total_Q:.5f} м³/с ({total_Q*1000:.3f} л/с)**")

# --- Общие графики (рядом) ---
col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(df["Сегмент"], df["Q (м³/с)"], marker="o")
    ax.set_xlabel("Сегмент")
    ax.set_ylabel("Q, м³/с")
    ax.set_title("Расход Q по сегментам")
    ax.grid(True)
    st.pyplot(fig)

with col2:
    ratios = np.linspace(0.05, 1.0, 100)
    Q_values = []
    for ratio in ratios:
        if shape == "Круглая":
            D, _ = segments[0][0]
            hh = ratio * D
            A, P = hydraulic_params_circular(D, hh)
        else:
            b, _ = segments[0][0]
            hh = ratio * h
            A, P = hydraulic_params_rectangular(b, hh)
        R, V, Q = manning_flow(A, P, S, n)
        Q_values.append(Q)

    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(ratios, Q_values, color="blue")
    ax.set_xlabel("h/D")
    ax.set_ylabel("Q, м³/с")
    ax.set_title("Зависимость Q от заполнения")
    ax.grid(True)
    st.pyplot(fig)
