import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =================== Справочник коэффициентов шероховатости n ===================
materials_n = {
    "Бетон": 0.013,
    "ПВХ": 0.009,
    "Чугун": 0.014,
    "Асбестоцемент": 0.011,
    "Сталь": 0.012
}

# =================== Интерфейс Streamlit ===================
st.set_page_config(page_title="Калькулятор расхода ливневых стоков", layout="wide")
st.title("💧 Калькулятор расхода ливневых стоков (несколько сегментов)")

# Выбор типа трубы
pipe_shape = st.selectbox("Форма трубы", ["Круглая", "Квадратная"])

# Ввод данных сегментов трубы
num_segments = st.number_input("Количество сегментов трубы", min_value=1, value=2, step=1)

segments = []
for i in range(num_segments):
    st.subheader(f"Сегмент {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        D_or_B = st.number_input(f"Диаметр трубы / ширина квадрата сегмента {i+1}, м", value=0.1, step=0.01, format="%.3f")
        h = st.number_input(f"Высота заполнения потоком сегмента {i+1}, м", value=0.07, step=0.01, format="%.3f")
        material = st.selectbox(f"Материал трубы сегмента {i+1}", list(materials_n.keys()))
    with col2:
        top = st.number_input(f"Высотная отметка верха колодца сегмента {i+1}, м", value=246.0, step=0.01)
        depth = st.number_input(f"Глубина колодца сегмента {i+1}, м", value=5.55, step=0.01)
        length = st.number_input(f"Длина сегмента трубы {i+1}, м", value=73.2, step=0.1)
    segments.append({
        "D_or_B": D_or_B,
        "h": h,
        "material": material,
        "top": top,
        "depth": depth,
        "length": length
    })

# =================== Расчеты ===================
total_Q_m3s = 0
Q_per_segment = []

st.subheader("📊 Результаты по сегментам")

fig_cross_sections, axes_cs = plt.subplots(1, num_segments, figsize=(4*num_segments, 4))

for idx, seg in enumerate(segments):
    D_or_B = seg["D_or_B"]
    h = seg["h"]
    material = seg["material"]
    top = seg["top"]
    depth = seg["depth"]
    length = seg["length"]
    
    invert = top - depth
    # для первого сегмента уклон пока просто 0.01 если один сегмент
    if idx < num_segments - 1:
        next_invert = segments[idx+1]["top"] - segments[idx+1]["depth"]
        S = (invert - next_invert) / length
    else:
        S = 0.01
    
    n = materials_n[material]
    
    if pipe_shape == "Круглая":
        r = D_or_B / 2
        theta = 2 * np.arccos((r - h) / r)
        A = (r**2 / 2) * (theta - np.sin(theta))
        P = r * theta
    else:  # Квадратная труба
        A = D_or_B * h
        P = D_or_B + 2*h
    
    R = A / P
    V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
    Q = A * V
    Q_per_segment.append(Q)
    total_Q_m3s += Q
    
    st.write(f"**Сегмент {idx+1}:** Q = {Q:.5f} м³/с = {Q*1000:.2f} л/с, уклон S = {S:.5f}")
    
    # =================== Поперечное сечение ===================
    ax = axes_cs[idx] if num_segments > 1 else axes_cs
    if pipe_shape == "Круглая":
        circle = plt.Circle((0, 0), D_or_B/2, color="lightgray", zorder=1)
        ax.add_patch(circle)
        x = np.linspace(-D_or_B/2, D_or_B/2, 200)
        mask = x**2 <= (D_or_B/2)**2
        y_min = -np.sqrt((D_or_B/2)**2 - x[mask]**2)
        y_max = np.sqrt((D_or_B/2)**2 - x[mask]**2)
        ax.fill_between(x[mask], y_min, y_min + h, color="blue", alpha=0.5, zorder=2)
    else:
        # квадратная труба
        ax.add_patch(plt.Rectangle((-D_or_B/2, -D_or_B/2), D_or_B, D_or_B, color="lightgray", zorder=1))
        ax.add_patch(plt.Rectangle((-D_or_B/2, -D_or_B/2), D_or_B, h, color="blue", alpha=0.5, zorder=2))
    
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlim(-D_or_B/2 - 0.05, D_or_B/2 + 0.05)
    ax.set_ylim(-D_or_B/2 - 0.05, D_or_B/2 + 0.05)
    ax.axis("off")
    ax.set_title(f"Сегмент {idx+1}", fontsize=12)

st.pyplot(fig_cross_sections)

st.subheader(f"💧 Суммарный расход по линии: {total_Q_m3s:.5f} м³/с = {total_Q_m3s*1000:.2f} л/с")

# =================== График Q по сегментам ===================
fig_seg, ax_seg = plt.subplots(figsize=(6,4))
ax_seg.bar(range(1, num_segments+1), [q*1000 for q in Q_per_segment], color='skyblue')
ax_seg.set_xlabel("Сегмент трубы")
ax_seg.set_ylabel("Q, л/с")
ax_seg.set_title("Расход Q по сегментам")
st.pyplot(fig_seg)

# =================== График зависимости Q от h/D ===================
st.subheader("📈 Зависимость Q от заполнения трубы")
fig_h_ratio, ax_hr = plt.subplots(figsize=(6,4))
for idx, seg in enumerate(segments):
    D_or_B = seg["D_or_B"]
    h_vals = np.linspace(0.05*D_or_B, D_or_B, 100)
    Q_vals = []
    n = materials_n[seg["material"]]
    invert = seg["top"] - seg["depth"]
    if idx < num_segments - 1:
        next_invert = segments[idx+1]["top"] - segments[idx+1]["depth"]
        S = (invert - next_invert) / seg["length"]
    else:
        S = 0.01
    for hh in h_vals:
        if pipe_shape == "Круглая":
            r = D_or_B/2
            theta = 2 * np.arccos((r - hh)/r)
            A = (r**2/2)*(theta - np.sin(theta))
            P = r*theta
        else:
            A = D_or_B*hh
            P = D_or_B + 2*hh
        R = A/P
        V = (1/n)*(R**(2/3))*(S**0.5)
        Q_vals.append(A*V*1000)  # л/с
    ax_hr.plot(h_vals/D_or_B, Q_vals, label=f"Сегмент {idx+1}")
ax_hr.set_xlabel("h/D")
ax_hr.set_ylabel("Q, л/с")
ax_hr.set_title("Зависимость Q от заполнения")
ax_hr.grid(True)
ax_hr.legend()
st.pyplot(fig_h_ratio)
