import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Справочник коэффициентов шероховатости n
materials_n = {
    "Бетон": 0.013,
    "ПВХ": 0.009,
    "Чугун": 0.014,
    "Асбестоцемент": 0.011,
    "Сталь": 0.012
}

st.set_page_config(page_title="Калькулятор расхода ливневых стоков", layout="wide")
st.title("💧 Калькулятор расхода ливневых стоков (формула Маннинга)")

# Выбор формы трубы
pipe_shape = st.selectbox("Форма трубы", ["Круглая", "Прямоугольная"])

# Ввод параметров сегментов трубы
num_segments = st.number_input("Количество сегментов трубы", min_value=1, step=1, value=1)

segments = []
for i in range(num_segments):
    st.subheader(f"Сегмент {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        if pipe_shape == "Круглая":
            D = st.number_input(f"Диаметр трубы сегмента {i+1}, м", value=0.1, step=0.01, format="%.3f", key=f"D{i}")
        else:
            D = st.number_input(f"Ширина трубы сегмента {i+1}, м", value=0.1, step=0.01, format="%.3f", key=f"b{i}")
            H = st.number_input(f"Высота трубы сегмента {i+1}, м", value=0.1, step=0.01, format="%.3f", key=f"H{i}")
        h = st.number_input(f"Высота заполнения потоком сегмента {i+1}, м", value=0.07, step=0.01, format="%.3f", key=f"h{i}")
        material = st.selectbox(f"Материал трубы сегмента {i+1}", list(materials_n.keys()), key=f"mat{i}")
    with col2:
        top1 = st.number_input(f"Высотная отметка верха 1-го колодца сегмента {i+1}, м", value=246.0, step=0.01, key=f"top1_{i}")
        depth1 = st.number_input(f"Глубина 1-го колодца сегмента {i+1}, м", value=5.55, step=0.01, key=f"depth1_{i}")
        top2 = st.number_input(f"Высотная отметка верха 2-го колодца сегмента {i+1}, м", value=245.0, step=0.01, key=f"top2_{i}")
        depth2 = st.number_input(f"Глубина 2-го колодца сегмента {i+1}, м", value=5.57, step=0.01, key=f"depth2_{i}")
        length = st.number_input(f"Длина сегмента {i+1}, м", value=73.2, step=0.1, key=f"length{i}")
    
    segments.append({
        "D": D,
        "H": H if pipe_shape == "Прямоугольная" else None,
        "h": h,
        "material": material,
        "top1": top1,
        "depth1": depth1,
        "top2": top2,
        "depth2": depth2,
        "length": length
    })

total_Q_ls = 0
st.subheader("📊 Результаты расчёта для каждого сегмента")
for idx, seg in enumerate(segments):
    # Расчет уклона
    invert1 = seg["top1"] - seg["depth1"]
    invert2 = seg["top2"] - seg["depth2"]
    S = (invert1 - invert2) / seg["length"]

    n = materials_n[seg["material"]]

    if pipe_shape == "Круглая":
        r = seg["D"] / 2
        theta = 2 * np.arccos((r - seg["h"]) / r)
        A = (r**2 / 2) * (theta - np.sin(theta))
        P = r * theta
        fill_ratio = seg["h"] / seg["D"]
    else:
        b = seg["D"]
        H = seg["H"]
        y = seg["h"]
        A = b * y
        P = b + 2 * y
        fill_ratio = seg["h"] / seg["H"]

    R = A / P
    V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
    Q_m3s = A * V
    Q_ls = Q_m3s * 1000
    total_Q_ls += Q_ls

    st.write(f"**Сегмент {idx+1}**")
    st.write(f"Уклон S: `{S:.5f}` ({S*100:.3f} %)")
    st.write(f"Площадь A: `{A:.6f}` м²")
    st.write(f"Смоченный периметр P: `{P:.4f}` м")
    st.write(f"Гидравлический радиус R: `{R:.4f}` м")
    st.write(f"Скорость V: `{V:.3f}` м/с")
    st.write(f"Расход Q: `{Q_m3s:.5f}` м³/с  = `{Q_ls:.3f}` л/с")

st.subheader("💡 Суммарный расход всей линии")
st.write(f"**Qsum = {total_Q_ls:.3f} л/с**")

# ===== График расхода по сегментам =====
fig1, ax1 = plt.subplots(figsize=(6, 4))
Q_values = []
for seg in segments:
    invert1 = seg["top1"] - seg["depth1"]
    invert2 = seg["top2"] - seg["depth2"]
    S = (invert1 - invert2) / seg["length"]
    n = materials_n[seg["material"]]
    if pipe_shape == "Круглая":
        r = seg["D"] / 2
        theta = 2 * np.arccos((r - seg["h"]) / r)
        A = (r**2 / 2) * (theta - np.sin(theta))
        P = r * theta
    else:
        b = seg["D"]
        H = seg["H"]
        y = seg["h"]
        A = b * y
        P = b + 2 * y
    R = A / P
    V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
    Q_values.append(A * V)
ax1.bar([f"Сегмент {i+1}" for i in range(len(segments))], Q_values, color="skyblue")
ax1.set_ylabel("Q, м³/с")
ax1.set_title("Расход Q по сегментам")
st.pyplot(fig1)

# ===== График зависимости Q от степени заполнения =====
st.subheader("📈 Зависимость Q от степени заполнения трубы")
fig2, ax2 = plt.subplots(figsize=(6, 4))
for idx, seg in enumerate(segments):
    n = materials_n[seg["material"]]
    invert1 = seg["top1"] - seg["depth1"]
    invert2 = seg["top2"] - seg["depth2"]
    S = (invert1 - invert2) / seg["length"]

    fill_ratios = np.linspace(0.01, 1.0, 50)
    Q_curve = []
    if pipe_shape == "Круглая":
        r = seg["D"] / 2
        for fr in fill_ratios:
            h = fr * seg["D"]
            theta = 2 * np.arccos((r - h) / r)
            A = (r**2 / 2) * (theta - np.sin(theta))
            P = r * theta
            R = A / P
            V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
            Q_curve.append(A * V)
    else:
        b = seg["D"]
        H = seg["H"]
        for fr in fill_ratios:
            h = fr * H
            A = b * h
            P = b + 2 * h
            R = A / P
            V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
            Q_curve.append(A * V)
    ax2.plot(fill_ratios, Q_curve, label=f"Сегмент {idx+1}")
ax2.set_xlabel("Степень заполнения (h/D или h/H)")
ax2.set_ylabel("Q, м³/с")
ax2.set_title("Зависимость Q от заполнения трубы")
ax2.legend()
st.pyplot(fig2)
