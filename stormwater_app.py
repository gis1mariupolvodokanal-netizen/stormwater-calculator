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
    arg = np.clip((r - h) / r, -1.0, 1.0)
    theta = 2.0 * np.arccos(arg)
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

def draw_circular_cross_section(ax, D, h, A, P, R, theta):
    """
    Поперечное сечение круга с заливкой воды и аннотациями.
    """
    R_circ = D / 2.0
    y_center = R_circ
    
    circle = plt.Circle((0, y_center), R_circ, color="lightgray", zorder=1)
    ax.add_patch(circle)

    x = np.linspace(-R_circ, R_circ, 600)
    y_upper = np.sqrt(np.maximum(R_circ**2 - x**2, 0.0)) + y_center
    y_lower = -y_upper + 2 * y_center
    y_surface = h
    
    y_top = np.minimum(y_upper, y_surface)
    fill_mask = y_top > y_lower
    ax.fill_between(x[fill_mask], y_lower[fill_mask], y_top[fill_mask], color="skyblue", alpha=0.8, zorder=2)
    
    ax.plot([-np.sqrt(np.maximum(R_circ**2 - (h-y_center)**2, 0.0)), np.sqrt(np.maximum(R_circ**2 - (h-y_center)**2, 0.0))], [h, h], color='blue', linestyle='--', zorder=3)
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    textstr = f"θ = {theta:.3f} rad\nA = {A:.6f} м²\nP = {P:.5f} м\nR = {R:.5f} м"
    ax.text(-R_circ, h/2, textstr, fontsize=8, verticalalignment='center', bbox=props, zorder=4)

    if h > 0:
        ax.annotate('', xy=(0, h), xytext=(0, 0), arrowprops=dict(facecolor='black', shrink=0.05))
        ax.text(0.01, h/2, f'h = {h:.3f} м', fontsize=8, verticalalignment='center', zorder=4)
    
    ax.annotate('', xy=(R_circ, 0), xytext=(R_circ, D), arrowprops=dict(arrowstyle='<->'))
    ax.text(R_circ + 0.01, D/2, f'D = {D:.3f} м', fontsize=8, verticalalignment='center')
    
    ax.set_aspect("equal", adjustable="box")
    pad = 0.1 * D
    ax.set_xlim(-R_circ - pad, R_circ + pad)
    ax.set_ylim(-pad, D + pad)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м (низ лотка = 0)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)


def draw_rect_cross_section(ax, B, H, h):
    """
    Поперечное сечение прямоугольника с заливкой воды.
    """
    y_bottom = 0
    
    ax.add_patch(plt.Rectangle((-B/2.0, y_bottom), B, H, facecolor="lightgray", edgecolor="none", zorder=1))
    
    h_clamped = np.clip(h, 0.0, H)
    ax.add_patch(plt.Rectangle((-B/2.0, y_bottom), B, h_clamped, facecolor="skyblue", alpha=0.8, edgecolor="none", zorder=2))
    
    ax.annotate('', xy=(-B/2, y_bottom), xytext=(-B/2, h_clamped), arrowprops=dict(facecolor='black', shrink=0.05))
    ax.text(-B/2 + 0.01, h_clamped/2, f'h = {h_clamped:.3f} м', fontsize=8, verticalalignment='center')
    
    ax.annotate('', xy=(-B/2, y_bottom), xytext=(B/2, y_bottom), arrowprops=dict(arrowstyle='<->'))
    ax.text(0, y_bottom - 0.01, f'B = {B:.3f} м', fontsize=8, verticalalignment='top')

    ax.set_aspect("equal", adjustable="box")
    pad_x = 0.1 * max(B, 1e-6)
    pad_y = 0.1 * max(H, 1e-6)
    ax.set_xlim(-B/2.0 - pad_x, B/2.0 + pad_x)
    ax.set_ylim(-pad_y, H + pad_y)
    ax.set_xlabel("x, м")
    ax.set_ylabel("y, м (низ лотка = 0)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

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
            if f"D_{i}" not in st.session_state:
                st.session_state[f"D_{i}"] = 0.6
            D = st.number_input(f"Диаметр, м (сегмент {i+1})", value=st.session_state[f"D_{i}"], min_value=0.01, step=0.01, format="%.3f", key=f"D_{i}")
            
            # Начальное значение h не может быть больше D
            h_initial = min(0.3, D)
            h = st.number_input(f"Высота заполнения h, м (сегмент {i+1})", value=h_initial, min_value=0.0, max_value=D, step=0.01, format="%.3f", key=f"h_{i}")
            B, H = None, None
            
        else:
            if f"B_{i}" not in st.session_state:
                st.session_state[f"B_{i}"] = 0.6
            B = st.number_input(f"Ширина B, м (сегмент {i+1})", value=st.session_state[f"B_{i}"], min_value=0.01, step=0.01, format="%.3f", key=f"B_{i}")
            
            if f"H_{i}" not in st.session_state:
                st.session_state[f"H_{i}"] = 0.6
            H = st.number_input(f"Высота H, м (сегмент {i+1})", value=st.session_state[f"H_{i}"], min_value=0.01, step=0.01, format="%.3f", key=f"H_{i}")

            h_initial = min(0.3, H)
            h = st.number_input(f"Высота заполнения h, м (сегмент {i+1})", value=h_initial, min_value=0.0, max_value=H, step=0.01, format="%.3f", key=f"hrect_{i}")

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
q_lps_list = []
qh_sheet_rows = []

for idx, seg in enumerate(segments, start=1):
    n = materials_n[seg["material"]]
    
    if seg["D"] is not None and seg["D"] <= 0:
        st.warning(f"Сегмент {idx}: Диаметр должен быть больше нуля.")
        continue
    if seg["B"] is not None and seg["B"] <= 0 or seg["H"] is not None and seg["H"] <= 0:
        st.warning(f"Сегмент {idx}: Ширина и высота должны быть больше нуля.")
        continue
    if seg["L"] <= 0:
        st.warning(f"Сегмент {idx}: Длина должна быть больше нуля.")
        continue
        
    invert1 = seg["top1"] - seg["depth1"]
    invert2 = seg["top2"] - seg["depth2"]
    
    S_raw = (invert1 - invert2) / seg["L"]
    S = max(S_raw, 1e-6)
    if S_raw <= 0:
        st.warning(f"Сегмент {idx}: уклон S ≤ 0 (S={S_raw:.6f}). Для расчёта использовано S=1e-6.")

    h_input = seg["h"]
    theta = None
    if seg["shape"] == "Круглая":
        D = seg["D"]
        if D is None: continue
        A, P = circ_section_area_perimeter(D, h_input)
        r = D / 2.0
        arg = np.clip((r - h_input) / r, -1.0, 1.0)
        theta = 2.0 * np.arccos(arg)
    else:
        B, H = seg["B"], seg["H"]
        if B is None or H is None: continue
        A, P = rect_section_area_perimeter(B, H, h_input)

    R, V, Q = manning_Q(A, P, S, n)
    Q_lps = Q * 1000.0
    q_lps_list.append(Q_lps)

    st.markdown(f"---\n### 📊 Сегмент {idx}")
    st.write(f"Материал: **{seg['material']}**, n = `{n}`")
    st.write(f"Уклон S: `{S:.6f}` ({S*100:.3f} %)")
    st.write(f"Площадь A: `{A:.6f}` м², смоченный периметр P: `{P:.4f}` м, гидр. радиус R: `{R:.4f}` м")
    st.write(f"Скорость V: `{V:.3f}` м/с, **Расход Q: `{Q:.5f}` м³/с = `{Q_lps:.2f}` л/с**")

    g1, g2 = st.columns(2)

    with g1:
        fig, ax = plt.subplots(figsize=(5, 5))
        if seg["shape"] == "Круглая":
            draw_circular_cross_section(ax, D, h_input, A, P, R, theta)
        else:
            draw_rect_cross_section(ax, B, H, h_input)
        ax.set_title(f"Поперечное сечение трубы {seg['shape']} h={h_input:.2f} м", fontsize=10)
        st.pyplot(fig)

    with g2:
        if seg["shape"] == "Круглая":
            max_dim = D
        else:
            max_dim = H
        
        ratios = np.linspace(0.01, 1.0, 100)
        Qs = []
        for ratio in ratios:
            hh = ratio * max_dim
            if seg["shape"] == "Круглая":
                Ai, Pi = circ_section_area_perimeter(D, hh)
            else:
                Ai, Pi = rect_section_area_perimeter(B, H, hh)
            _, _, Qi = manning_Q(Ai, Pi, S, n)
            Qs.append(Qi * 1000.0)
            
            qh_sheet_rows.append({
                "Сегмент": idx,
                "ratio": ratio,
                "h (м)": hh,
                "Q (м³/с)": Qi
            })
            
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.plot(ratios, Qs, marker='o', markersize=4)
        ax2.set_xlabel("h/D" if seg["shape"] == "Круглая" else "h/H")
        ax2.set_ylabel("Q, л/с")
        ax2.set_title(f"Q(h) для сегмента {idx}", fontsize=10)
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        current_ratio = h_input / max_dim if max_dim > 0 else 0
        current_Q = Q_lps
        ax2.plot(current_ratio, current_Q, 'ro', label=f'Текущее Q = {current_Q:.2f} л/с')
        ax2.legend()
        
        st.pyplot(fig2)

    rows.append({
        "Сегмент": idx,
        "Форма": seg["shape"],
        "Материал": seg["material"],
        "Размер": f"D={D:.3f} м" if seg["shape"] == "Круглая" else f"B={B:.3f}м, H={H:.3f}м",
        "h (м)": h_input,
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
        "h (м)": "{:.3f}", "S (-)": "{:.6f}", "A (м²)": "{:.6f}", "P (м)": "{:.4f}",
        "R (м)": "{:.4f}", "V (м/с)": "{:.3f}", "Q (м³/с)": "{:.5f}", "Q (л/с)": "{:.2f}"
    }))

    total_Q_lps = df["Q (л/с)"].sum()
    st.markdown(f"### 💡 Суммарный расход по линии: **{total_Q_lps:.2f} л/с**")

    cA, cB = st.columns(2)

    with cA:
        figb, axb = plt.subplots(figsize=(4.2, 3.0))
        axb.bar(df["Сегмент"].astype(str), df["Q (л/с)"])
        axb.set_xlabel("Сегмент")
        axb.set_ylabel("Q, л/с")
        axb.set_title("Расход Q по сегментам", fontsize=10)
        axb.grid(axis="y", alpha=0.3)
        st.pyplot(figb)

    with cB:
        st.info("Кривые Q(h) для каждого сегмента сохранены в Excel (лист «Q(h)»).")

    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name="Сегменты")
            qh_df = pd.DataFrame(qh_sheet_rows)
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
