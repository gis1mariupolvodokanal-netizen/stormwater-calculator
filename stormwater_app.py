import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# Функция для гидравлического радиуса и площади
def hydraulic_params(shape, D, h):
    if shape == "Круглая":
        r = D / 2
        theta = 2 * np.arccos((r - h) / r)
        area = (r**2 / 2) * (theta - np.sin(theta))
        wetted_perimeter = r * theta
    else:  # Прямоугольная
        b = D
        area = b * h
        wetted_perimeter = b + 2 * h
    R = area / wetted_perimeter if wetted_perimeter > 0 else 0
    return area, R

# Расчет расхода
def manning_flow(shape, D, h, S, n=0.013):
    area, R = hydraulic_params(shape, D, h)
    Q = (1 / n) * area * (R**(2/3)) * np.sqrt(S)
    return Q

st.title("💧 Гидравлический расчет трубопровода")

# Ввод количества сегментов
num_segments = st.number_input("Количество сегментов", min_value=1, max_value=10, value=2)

segments = []
for i in range(num_segments):
    st.subheader(f"Сегмент {i+1}")
    shape = st.selectbox(f"Форма трубы (Сегмент {i+1})", ["Круглая", "Прямоугольная"], key=f"shape_{i}")
    D = st.number_input(f"Диаметр/ширина трубы, м (Сегмент {i+1})", min_value=0.1, value=0.5, step=0.1, key=f"D_{i}")
    h = st.number_input(f"Высота воды, м (Сегмент {i+1})", min_value=0.01, value=D/2, step=0.05, key=f"h_{i}")
    z1 = st.number_input(f"Высотная отметка 1-го колодца, м (Сегмент {i+1})", value=100.0, step=0.5, key=f"z1_{i}")
    z2 = st.number_input(f"Высотная отметка 2-го колодца, м (Сегмент {i+1})", value=99.0, step=0.5, key=f"z2_{i}")
    L = st.number_input(f"Длина трубы, м (Сегмент {i+1})", min_value=1.0, value=50.0, step=1.0, key=f"L_{i}")
    S = max((z1 - z2) / L, 1e-6)  # уклон
    Q = manning_flow(shape, D, h, S)
    segments.append({"shape": shape, "D": D, "h": h, "Q": Q, "S": S})

# Вывод результатов
st.header("📊 Результаты по сегментам")

q_values = [seg["Q"] for seg in segments]

cols = st.columns(num_segments)  # выводим поперечные сечения в строку
for i, seg in enumerate(segments):
    with cols[i]:
        st.markdown(f"**Сегмент {i+1}:** Q = {seg['Q']:.4f} м³/с = {seg['Q']*1000:.2f} л/с, уклон S = {seg['S']:.5f}")

        # Поперечное сечение трубы
        fig, ax = plt.subplots(figsize=(3, 3))
        if seg["shape"] == "Круглая":
            circle = plt.Circle((0, 0), seg["D"]/2, color="lightgrey", alpha=0.5)
            ax.add_patch(circle)
            theta = np.linspace(0, 2*np.pi, 200)
            x = (seg["D"]/2) * np.cos(theta)
            y = (seg["D"]/2) * np.sin(theta)
            ax.plot(x, y, 'k')
            ax.fill_between(x, y, -seg["D"]/2, where=(y <= seg["h"] - seg["D"]/2), color="blue", alpha=0.4)
        else:  # Прямоугольная труба
            ax.add_patch(plt.Rectangle((-seg["D"]/2, 0), seg["D"], seg["D"], fill=None, edgecolor='k'))
            ax.add_patch(plt.Rectangle((-seg["D"]/2, 0), seg["D"], seg["h"], color="blue", alpha=0.4))
        ax.set_xlim(-seg["D"], seg["D"])
        ax.set_ylim(0, seg["D"])
        ax.set_aspect("equal")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)

# График Q по сегментам
fig_seg, ax_seg = plt.subplots(figsize=(6, 3))
ax_seg.bar(range(1, num_segments+1), q_values, color="skyblue")
ax_seg.set_xlabel("Сегмент")
ax_seg.set_ylabel("Q, м³/с")
ax_seg.set_title("Расход Q по сегментам")
st.pyplot(fig_seg, use_container_width=True)

# График зависимости Q(h/D)
fig_ratio, ax_ratio = plt.subplots(figsize=(6, 3))
for i, seg in enumerate(segments):
    h_ratios = np.linspace(0.01, seg["D"], 30)
    Q_vals = [manning_flow(seg["shape"], seg["D"], h, seg["S"]) for h in h_ratios]
    ax_ratio.plot(h_ratios/seg["D"], Q_vals, label=f"Сегмент {i+1}")
ax_ratio.set_xlabel("h/D")
ax_ratio.set_ylabel("Q, м³/с")
ax_ratio.set_title("Зависимость Q от заполнения трубы")
ax_ratio.legend()
st.pyplot(fig_ratio, use_container_width=True)

# Суммарный расход
st.subheader(f"💡 Суммарный расход: {sum(q_values):.4f} м³/с = {sum(q_values)*1000:.2f} л/с")
