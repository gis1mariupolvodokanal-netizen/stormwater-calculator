import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Справочник коэффициентов шероховатости n =====
materials_n = {
    "Бетон": 0.013,
    "ПВХ": 0.009,
    "Чугун": 0.014,
    "Асбестоцемент": 0.011,
    "Сталь": 0.012
}

# ===== Настройки страницы =====
st.set_page_config(page_title="Калькулятор расхода ливневых стоков", layout="wide")
st.title("💧 Калькулятор расхода ливневых стоков (формула Маннинга)")

# ===== Ввод параметров =====
col1, col2 = st.columns(2)
with col1:
    D = st.number_input("Диаметр трубы, м", value=0.1, step=0.01, format="%.3f")
    h = st.number_input("Высота заполнения потоком, м", value=0.07, step=0.01, format="%.3f")
    material = st.selectbox("Материал трубы", list(materials_n.keys()))
with col2:
    top1 = st.number_input("Отметка верха 1-го колодца, м", value=246.0, step=0.01)
    depth1 = st.number_input("Глубина 1-го колодца, м", value=5.55, step=0.01)
    top2 = st.number_input("Отметка верха 2-го колодца, м", value=245.0, step=0.01)
    depth2 = st.number_input("Глубина 2-го колодца, м", value=5.57, step=0.01)
    length = st.number_input("Расстояние между колодцами, м", value=73.2, step=0.1)

# ===== Расчёт уклона =====
invert1 = top1 - depth1
invert2 = top2 - depth2
S = (invert1 - invert2) / length  # безразмерный уклон

# ===== Функция для расчёта геометрии и расхода =====
def manning_flow(D, h, n, S):
    r = D / 2
    theta = 2 * np.arccos(np.clip((r - h) / r, -1, 1))
    A = (r**2 / 2) * (theta - np.sin(theta))
    P = r * theta
    R = A / P
    V = (1 / n) * (R ** (2/3)) * np.sqrt(S)
    Q = A * V
    return A, P, R, V, Q

# ===== Расчёт для заданного уровня заполнения =====
n = materials_n[material]
A, P, R, V, Q_m3s = manning_flow(D, h, n, S)
Q_ls = Q_m3s * 1000  # л/с

st.subheader("📊 Результаты расчёта для заданного уровня заполнения")
st.write(f"Уклон S: `{S:.5f}` ({S*100:.3f} %)")
st.write(f"Площадь A: `{A:.6f}` м²")
st.write(f"Смоченный периметр P: `{P:.4f}` м")
st.write(f"Гидравлический радиус R: `{R:.4f}` м")
st.write(f"Скорость V: `{V:.3f}` м/с")
st.write(f"**Расход Q: `{Q_m3s:.5f}` м³/с  = `{Q_ls:.3f}` л/с**")

# ===== Таблица Q для разных h/D =====
ratios = np.linspace(0.05, 1.0, 200)
table_data = []
for ratio in ratios:
    hh = ratio * D
    A_i, P_i, R_i, V_i, Q_i = manning_flow(D, hh, n, S)
    table_data.append([ratio, hh, A_i, P_i, R_i, V_i, Q_i, Q_i*1000])

df = pd.DataFrame(table_data, columns=["h/D", "h (м)", "A (м²)", "P (м)", "R (м)", "V (м/с)", "Q (м³/с)", "Q (л/с)"])

# ===== Определение максимального расхода =====
max_row = df.loc[df["Q (м³/с)"].idxmax()]
max_ratio = max_row["h/D"]
max_Q_ls = max_row["Q (л/с)"]

st.subheader("⭐ Оптимальное заполнение для максимального расхода")
st.write(f"Максимальный расход достигается при `h/D ≈ {max_ratio:.2f}`")
st.write(f"**Qmax = {max_Q_ls:.3f} л/с**")

# ===== Интерактивный график зависимости Q от h/D с отметкой максимума =====
fig1, ax1 = plt.subplots(figsize=(6, 4))
ax1.plot(df["h/D"], df["Q (л/с)"], color="blue", label="Q(h)")
ax1.scatter(max_ratio, max_Q_ls, color="red", zorder=5, label="Максимальный расход")
ax1.annotate(f"Qmax={max_Q_ls:.1f} л/с\nh/D={max_ratio:.2f}", 
             xy=(max_ratio, max_Q_ls), xytext=(max_ratio+0.05, max_Q_ls),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=9)
ax1.set_xlabel("h/D")
ax1.set_ylabel("Q, л/с")
ax1.set_title("Расход Q в зависимости от h/D")
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.legend()
st.pyplot(fig1)

# ===== Чертёж поперечного сечения трубы =====
fig2, ax2 = plt.subplots(figsize=(5, 5))
r = D / 2
circle = plt.Circle((0, 0), r, color="lightgray", zorder=1)
ax2.add_patch(circle)

x = np.linspace(-r, r, 500)
y_circle = np.sqrt(r**2 - x**2)
y_water = np.clip(y_circle, -r, h - r)
ax2.fill_between(x, -r, y_water, color="blue", alpha=0.5)

# Подписи
ax2.annotate("D", xy=(0, r + 0.01), ha="center", fontsize=10)
ax2.annotate("h", xy=(r + 0.01, -r + h/2), fontsize=10)

ax2.set_aspect("equal", adjustable="datalim")
ax2.set_xlim(-r - 0.05, r + 0.05)
ax2.set_ylim(-r - 0.05, r + 0.05)
ax2.axis("off")
ax2.set_title("Поперечное сечение трубы", fontsize=12)
st.pyplot(fig2)

# ===== Таблица для отображения и скачивания =====
st.subheader("📋 Таблица расхода при разном заполнении")
st.dataframe(df.style.format({"h/D": "{:.2f}", "h (м)": "{:.3f}", "A (м²)": "{:.6f}", "P (м)": "{:.4f}",
                              "R (м)": "{:.4f}", "V (м/с)": "{:.3f}", "Q (м³/с)": "{:.5f}", "Q (л/с)": "{:.3f}"}))

csv = df.to_csv(index=False).encode("utf-8")
st.download_button("💾 Скачать таблицу в CSV", csv, "Q_table.csv", "text/csv")