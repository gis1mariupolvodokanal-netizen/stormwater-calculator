import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

# ----------------------- Справочник шероховатости -----------------------
materials_n = {
    "Бетон": 0.013,
    "ПВХ": 0.009,
    "Чугун": 0.014,
    "Асбестоцемент": 0.011,
    "Сталь": 0.012
}

st.set_page_config(page_title="Калькулятор расхода ливневых стоков", layout="wide")
st.title("💧 Калькулятор расхода (Маннинг) — несколько сегментов, Excel и сечения")

# ----------------------- Вспомогательные функции -----------------------
def clip01(x):
    return max(0.0, min(1.0, x))

def circ_section_area_perimeter(D, h):
    """
    Круглая труба. D — диаметр, h — высота заполнения (0..D).
    Возвращает (A, P) — площадь и смоченный периметр.
    """
    if h <= 0:
        return 0.0, 0.0
    if h >= D:
        r = D / 2.0
        A_full = np.pi * r**2
        P_full = 2.0 * np.pi * r
        return A_full, P_full
    r = D / 2.0
    # Аргумент для arccos клиппируем от -1 до 1 для численной устойчивости
    arg = (r - h) / r
    arg = np.clip(arg, -1.0, 1.0)
    theta = 2.0 * np.arccos(arg)  # центральный угол смачиваемой части (рад)
    A = 0.5 * r**2 * (theta - np.sin(theta))
    P = r * theta
    return A, P

def rect_section_area_perimeter(B, H, h):
    """
    Прямоугольная труба (ширина B, высота H), h — уровень воды (0..H).
    Возвращает (A, P).
    """
    if h <= 0:
        return 0.0, 0.0
    if h >= H:
        A_full = B * H
        P_full = B + 2.0 * H
        return A_full, P_full
    A = B * h
    P = B + 2.0 * h
    return A, P

def manning_Q(A, P, S, n):
    if A <= 0 or P <= 0 or S <= 0:
        return 0.0, 0.0, 0.0
    R = A / P
    V = (1.0 / n) * (R ** (2.0 / 3.0)) * (S ** 0.5)
    Q = A * V
    return R, V, Q

def draw_circular_cross_section(ax, D, h):
    """
    Красивое поперечное сечение круга с заливкой воды до уровня h.
    Система координат: центр окружности (0,0), радиус R = D/2.
    Вода от y = -R до y = -R + h.
    """
    R = D / 2.0
    # Контур трубы
    circle = plt.Circle((0, 0), R, color="lightgray", zorder=1)
    ax.add_patch(circle)

    # Сетка точек по x и границы круга
    x = np.linspace(-R, R, 600)
    y_upper = np.sqrt(np.maximum(R**2 - x**2, 0.0))
    y_lower = -y_upper

    # Уровень воды
    y_surface = -R + h
    # Верх линии заливки — минимум между верхней границей круга и линией воды
    y_top = np.minimum(y_upper, y_surface)
    # Заполняем там, где верх заливки выше нижней границы
    fill_mask = y_top > y_lower
    ax.fill_between(x[fill_mask], y_lower[fill_mask], y_top[fill_mask], color="blue", alpha=0.5, zorder=2)

    # Декор
    ax.set_aspect("equal", adjustable="box")
    pad = 0.06 * D
    ax.set_xlim(-R - pad, R + pad)
    ax.set_ylim(-R - pad, R + pad)
    ax.axis("off")

def draw_rect_cross_section(ax, B, H, h):
    """
    Красивое поперечное сечение прямоугольника с заливкой воды до h.
    Прямоугольник центрирован по вертикали: от y=-H/2 до y=+H/2.
    Вода от y=-H/2 до y=-H/2 + h.
    """
    y_bottom = -H / 2.0
    # Контур трубы
    ax.add_patch(plt.Rectangle((-B/2.0, -H/2.0), B, H, facecolor="lightgray", edgecolor="none", zorder=1))
    # Заливка водой
    h_clamped = np.clip(h, 0.0, H)
    ax.add_patch(plt.Rectangle((-B/2.0, y_bottom), B, h_clamped, facecolor="blue", alpha=0.5, edgecolor="none", zorder=2))
    # Декор
    ax.set_aspect("equal", adjustable="box")
    pad_x = 0.06 * max(B, 1e-6)
    pad_y = 0.06 * max(H, 1e-6)
    ax.set_xlim(-B/2.0 - pad_x, B/2.0 + pad_x)
    ax.set_ylim(-H/2.0 - pad_y, H/2.0 + pad_y)
    ax.axis("off")

# ----------------------- Ввод данных -----------------------
num_segments = st.number_input("Количество сегментов", min_value=1, max_value=20, value=2, step=1)

segments = []
for i in range(num_segments):
    st.subheader(f"Сегмент {i+1}")
    c1, c2 = st.columns(2)

    with c1:
        shape = st.selectbox(f"Форма трубы (сегмент {i+1})", ["Круглая", "Прямоугольная"], key=f"shape_{i}")
        material = st.selectbox(f"Материал трубы (сегмент {i+1})", list(materials_n.keys()), key=f"mat_{i}")
        if shape == "Круглая":
            D = st.number_input(f"Диаметр, м (сегмент {i+1})", value=0.6, min_value=0.05, step=0.01, format="%.3f", key=f"D_{i}")
            h = st.number_input(f"Высота заполнения h, м (сегмент {i+1})", value=0.3, min_value=0.0, step=0.01, format="%.3f", key=f"h_{i}")
            B, H = None, None
        else:
            B = st.number_input(f"Ширина B, м (сегмент {i+1})", value=0.6, min_value=0.05, step=0.01, format="%.3f", key=f"B_{i}")
            H = st.number_input(f"Высота H, м (сегмент {i+1})", value=0.6, min_value=0.05, step=0.01, format="%.3f", key=f"H_{i}")
            h = st.number_input(f"Высота заполнения h, м (сегмент {i+1})", value=0.3, min_value=0.0, step=0.01, format="%.3f", key=f"hrect_{i}")

    with c2:
        top1 = st.number_input(f"Высотная отметка верха 1-го колодца, м (сегмент {i+1})", value=246.00, step=0.01, key=f"top1_{i}")
        depth1 = st.number_input(f"Глубина 1-го колодца, м (сегмент {i+1})", value=5.50, step=0.01, key=f"depth1_{i}")
        top2 = st.number_input(f"Высотная отметка верха 2-го колодца, м (сегмент {i+1})", value=245.00, step=0.01, key=f"top2_{i}")
        depth2 = st.number_input(f"Глубина 2-го колодца, м (сегмент {i+1})", value=5.60, step=0.01, key=f"depth2_{i}")
        length = st.number_input(f"Расстояние между колодцами, м (сегмент {i+1})", value=50.0, min_value=0.1, step=0.1, key=f"L_{i}")

    segments.append({
        "shape": shape, "material": material, "D": None if shape!="Круглая" else D,
        "B": None if shape!="Прямоугольная" else B,
        "H": None if shape!="Прямоугольная" else H,
        "h": h, "top1": top1, "depth1": depth1, "top2": top2, "depth2": depth2, "L": length
    })

# ----------------------- Расчёт по сегментам -----------------------
rows = []
q_lps_list = []  # для суммарного
qh_sheet_rows = []  # для листа Q(h)

for idx, seg in enumerate(segments, start=1):
    n = materials_n[seg["material"]]
    invert1 = seg["top1"] - seg["depth1"]
    invert2 = seg["top2"] - seg["depth2"]
    S_raw = (invert1 - invert2) / max(seg["L"], 1e-6)
    if S_raw <= 0:
        st.warning(f"Сегмент {idx}: уклон S ≤ 0 (S={S_raw:.6f}). Для расчёта использовано S=1e-6.")
    S = max(S_raw, 1e-6)

    # Клиппинг h (на всякий)
    if seg["shape"] == "Круглая":
        D = seg["D"]
        if D is None:
            st.error(f"Сегмент {idx}: не указан диаметр.")
            continue
        h = np.clip(seg["h"], 0.0, D)
        A, P = circ_section_area_perimeter(D, h)
    else:
        B, H = seg["B"], seg["H"]
        if B is None or H is None:
            st.error(f"Сегмент {idx}: не указаны B/H.")
            continue
        h = np.clip(seg["h"], 0.0, H)
        A, P = rect_section_area_perimeter(B, H, h)

    R, V, Q = manning_Q(A, P, S, n)
    Q_lps = Q * 1000.0
    q_lps_list.append(Q_lps)

    # ---------- Вывод результатов ----------
    st.markdown(f"---\n### 📊 Сегмент {idx}")
    st.write(f"Материал: **{seg['material']}**, n = `{n}`")
    st.write(f"Уклон S: `{S:.6f}` ({S*100:.3f} %)")
    st.write(f"Площадь A: `{A:.6f}` м², смоченный периметр P: `{P:.4f}` м, гидр. радиус R: `{R:.4f}` м")
    st.write(f"Скорость V: `{V:.3f}` м/с, **Расход Q: `{Q:.5f}` м³/с = `{Q_lps:.2f}` л/с**")

    # ---------- Графики рядом ----------
    g1, g2 = st.columns(2)

    with g1:
        fig, ax = plt.subplots(figsize=(3.2, 3.2))
        if seg["shape"] == "Круглая":
            draw_circular_cross_section(ax, D, h)
        else:
            draw_rect_cross_section(ax, B, H, h)
        ax.set_title("Поперечное сечение", fontsize=10)
        st.pyplot(fig)

    with g2:
        # Кривая Q(h) для данного сегмента
        ratios = np.linspace(0.05, 1.0, 120)  # гладкая кривая
        Qs = []
        for ratio in ratios:
            if seg["shape"] == "Круглая":
                hh = ratio * D
                Ai, Pi = circ_section_area_perimeter(D, hh)
            else:
                hh = ratio * H
                Ai, Pi = rect_section_area_perimeter(B, H, hh)
            _, _, Qi = manning_Q(Ai, Pi, S, n)
            Qs.append(Qi)
            # Соберём в Excel-лист Q(h)
            qh_sheet_rows.append({
                "Сегмент": idx,
                "ratio": ratio,
                "h (м)": hh,
                "Q (м³/с)": Qi
            })

        fig2, ax2 = plt.subplots(figsize=(4.0, 3.0))
        ax2.plot(ratios, Qs)
        ax2.set_xlabel("h/D" if seg["shape"] == "Круглая" else "h/H")
        ax2.set_ylabel("Q, м³/с")
        ax2.set_title("Q(h) данного сегмента", fontsize=10)
        ax2.grid(True)
        st.pyplot(fig2)

    # строка в основную таблицу
    rows.append({
        "Сегмент": idx,
        "Форма": seg["shape"],
        "Материал": seg["material"],
        "S (-)": S,
        "A (м²)": A,
        "P (м)": P,
        "R (м)": R,
        "V (м/с)": V,
        "Q (м³/с)": Q,
        "Q (л/с)": Q_lps
    })

# ----------------------- Итоги и общие графики -----------------------
if rows:
    df = pd.DataFrame(rows)
    st.subheader("📋 Таблица результатов по сегментам")
    st.dataframe(df.style.format({
        "S (-)": "{:.6f}", "A (м²)": "{:.6f}", "P (м)": "{:.4f}",
        "R (м)": "{:.4f}", "V (м/с)": "{:.3f}", "Q (м³/с)": "{:.5f}", "Q (л/с)": "{:.2f}"
    }))

    total_Q_lps = df["Q (л/с)"].sum()
    st.markdown(f"### 💡 Суммарный расход по линии: **{total_Q_lps:.2f} л/с**")

    cA, cB = st.columns(2)

    with cA:
        # Столбчатый график Q по сегментам
        figb, axb = plt.subplots(figsize=(4.2, 3.0))
        axb.bar(df["Сегмент"].astype(str), df["Q (м³/с)"])
        axb.set_xlabel("Сегмент")
        axb.set_ylabel("Q, м³/с")
        axb.set_title("Расход Q по сегментам", fontsize=10)
        axb.grid(axis="y", alpha=0.3)
        st.pyplot(figb)

    with cB:
        # Суммарная Q(h) для «эталонного» сегмента? — не суммируем, показываем пример для 1-го,
        # т.к. уклоны/размеры/материалы отличаются. В Excel есть полные кривые для каждого.
        st.info("Кривые Q(h) для каждого сегмента сохранены в Excel (лист «Q(h)»).")

    # ----------------------- Выгрузка в Excel -----------------------
    qh_df = pd.DataFrame(qh_sheet_rows) if qh_sheet_rows else pd.DataFrame(
        columns=["Сегмент", "ratio", "h (м)", "Q (м³/с)"]
    )
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Сегменты")
            qh_df.to_excel(writer, index=False, sheet_name="Q(h)")
        xlsx_bytes = output.getvalue()
        st.download_button(
            "💾 Скачать результаты в Excel",
            data=xlsx_bytes,
            file_name="расчет_ливневки.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    except Exception as e:
        st.error(f"Экспорт в Excel не удался: {e}")
        st.download_button(
            "💾 Скачать результаты в CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="расчет_ливневки.csv",
            mime="text/csv",
        )
