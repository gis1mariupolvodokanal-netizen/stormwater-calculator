import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# Коэффициенты шероховатости n
materials_n = {
    "Бетон": 0.013,
    "ПВХ": 0.009,
    "Чугун": 0.014,
    "Асбестоцемент": 0.011,
    "Сталь": 0.012
}

st.set_page_config(page_title="Калькулятор расхода ливневых стоков", layout="wide")
st.title("💧 Калькулятор расхода ливневых стоков (формула Маннинга)")

# Ввод количества сегментов
num_segments = st.number_input("Количество сегментов трубы", min_value=1, max_value=10, value=1, step=1)

segments = []
for i in range(num_segments):
    st.subheader(f"Сегмент {i+1}")
    col1, col2 = st.columns(2)
    with col1:
        shape = st.selectbox(f"Форма трубы (сегмент {i+1})", ["Круглая", "Прямоугольная"], key=f"shape_{i}")
        if shape == "Круглая":
            D = st.number_input(f"Диаметр трубы, м (сегмент {i+1})", value=0.5, step=0.01, format="%.3f", key=f"D_{i}")
            h = st.number_input(f"Высота заполнения потоком, м (сегмент {i+1})", value=0.25, step=0.01, format="%.3f", key=f"h_{i}")
            B, H = None, None
        else:
            B = st.number_input(f"Ширина трубы, м (сегмент {i+1})", value=0.5, step=0.01, format="%.3f", key=f"B_{i}")
            H = st.number_input(f"Высота трубы, м (сегмент {i+1})", value=0.5, step=0.01, format="%.3f", key=f"H_{i}")
            h = st.number_input(f"Высота заполнения потоком, м (сегмент {i+1})", value=0.25, step=0.01, format="%.3f", key=f"h_{i}")
            D = None
        material = st.selectbox(f"Материал трубы (сегмент {i+1})", list(materials_n.keys()), key=f"mat_{i}")
    with col2:
        top1 = st.number_input(f"Высотная отметка верха 1-го колодца, м (сегмент {i+1})", value=246.0, step=0.01, key=f"top1_{i}")
        depth1 = st.number_input(f"Глубина 1-го колодца, м (сегмент {i+1})", value=5.0, step=0.01, key=f"depth1_{i}")
        top2 = st.number_input(f"Высотная отметка верха 2-го колодца, м (сегмент {i+1})", value=245.0, step=0.01, key=f"top2_{i}")
        depth2 = st.number_input(f"Глубина 2-го колодца, м (сегмент {i+1})", value=5.1, step=0.01, key=f"depth2_{i}")
        length = st.number_input(f"Расстояние между колодцами, м (сегмент {i+1})", value=50.0, step=0.1, key=f"len_{i}")

    segments.append({
        "shape": shape, "D": D, "B": B, "H": H, "h": h,
        "material": material, "top1": top1, "depth1": depth1,
        "top2": top2, "depth2": depth2, "length": length
    })

results = []
df_all = []

for idx, seg in enumerate(segments):
    st.markdown(f"---\n### 📊 Результаты для сегмента {idx+1}")
    invert1 = seg["top1"] - seg["depth1"]
    invert2 = seg["top2"] - seg["depth2"]
    S = (invert1 - invert2) / seg["length"]

    n = materials_n[seg["material"]]

    if seg["shape"] == "Круглая":
        r = seg["D"] / 2
        theta = 2 * np.arccos((r - seg["h"]) / r)
        A = (r**2 / 2) * (theta - np.sin(theta))
        P = r * theta
        R = A / P
    else:
        A = seg["B"] * seg["h"]
        P = seg["B"] + 2 * seg["h"]
        R = A / P

    V = (1 / n) * (R ** (2/3)) * (S ** 0.5)
    Q_m3s = A * V
    Q_ls = Q_m3s * 1000

    results.append(Q_ls)

    st.write(f"Уклон S: `{S:.5f}` ({S*100:.3f} %)")
    st.write(f"Площадь A: `{A:.6f}` м²")
    st.write(f"Смоченный периметр P: `{P:.4f}` м")
    st.write(f"Гидравлический радиус R: `{R:.4f}` м")
    st.write(f"Скорость V: `{V:.3f}` м/с")
    st.write(f"**Расход Q: `{Q_m3s:.5f}` м³/с  = `{Q_ls:.3f}` л/с**")

    # Рисуем графики рядом
    col1, col2 = st.columns(2)

    with col1:
        # Поперечное сечение
        fig, ax = plt.subplots(figsize=(3,3))
        if seg["shape"] == "Круглая":
            circle = plt.Circle((0, 0), seg["D"]/2, color="lightgray", zorder=1)
            ax.add_patch(circle)
            theta = 2 * np.arccos((seg["D"]/2 - seg["h"]) / (seg["D"]/2))
            x = np.linspace(-seg["D"]/2, seg["D"]/2, 200)
            y_max = np.sqrt((seg["D"]/2)**2 - x**2)
            y_min = -y_max
            ax.fill_between(x, y_min, y_max, where=(y_max <= (seg["h"] - seg["D"]/2)),
                            color="blue", alpha=0.5, zorder=2)
        else:
            ax.add_patch(plt.Rectangle((-seg["B"]/2, -seg["H"]/2), seg["B"], seg["H"],
                                       color="lightgray", zorder=1))
            ax.add_patch(plt.Rectangle((-seg["B"]/2, -seg["H"]/2), seg["B"], seg["h"],
                                       color="blue", alpha=0.5, zorder=2))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Поперечное сечение", fontsize=10)
        st.pyplot(fig)

    with col2:
        # Зависимость Q(h/D или h/H)
        ratios = np.linspace(0.05, 1.0, 100)
        Qs = []
        for ratio in ratios:
            hh = ratio * (seg["D"] if seg["shape"]=="Круглая" else seg["H"])
            if seg["shape"] == "Круглая":
                r = seg["D"]/2
                theta = 2 * np.arccos((r - hh) / r)
                A_i = (r**2 / 2) * (theta - np.sin(theta))
                P_i = r * theta
            else:
                A_i = seg["B"] * hh
                P_i = seg["B"] + 2*hh
            R_i = A_i / P_i
            V_i = (1/n) * (R_i**(2/3)) * (S**0.5)
            Qs.append(A_i * V_i)
        fig2, ax2 = plt.subplots(figsize=(3,3))
        ax2.plot(ratios, Qs, color="blue")
        ax2.set_xlabel("h/D" if seg["shape"]=="Круглая" else "h/H", fontsize=9)
        ax2.set_ylabel("Q (м³/с)", fontsize=9)
        ax2.grid(True)
        ax2.set_title("Q(h)", fontsize=10)
        st.pyplot(fig2)

    df_seg = pd.DataFrame({"Сегмент": [idx+1], "Форма": [seg["shape"]], "Q (л/с)": [Q_ls]})
    df_all.append(df_seg)

# Суммарный расход
st.subheader("💡 Суммарный расход всей линии")
total_Q = sum(results)
st.write(f"**Qсумм = {total_Q:.3f} л/с**")

# Таблица и выгрузка в Excel
df_all = pd.concat(df_all, ignore_index=True)
st.dataframe(df_all)

output = io.BytesIO()
with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
    df_all.to_excel(writer, index=False, sheet_name="Расчет")
st.download_button("💾 Скачать результаты в Excel", data=output.getvalue(),
                   file_name="расчет_ливневки.xlsx", mime="application/vnd.ms-excel")
