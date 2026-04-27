import cv2
import numpy as np
import os
from datetime import datetime
from ultralytics import FastSAM

input_dir = 'data/pile_of_metal_washers'

# ==============================================================
#  HIPERPARÁMETROS — solo tocar estas líneas entre experimentos
# ==============================================================

# --- Clásico (HoughCircles) ---
MIN_DIST_RATIO  = 0.18  # distancia mínima entre centros (% del lado corto)
                          #   bajo  (0.05-0.08) → más detecciones, más falsos positivos
                          #   alto  (0.14-0.20) → menos detecciones, más pérdidas
PARAM2          = 45      # umbral del acumulador de Hough
                          #   bajo  (25-32) → muy sensible, detecta arcos débiles
                          #   alto  (42-50) → solo acepta círculos muy nítidos
MIN_RADIUS_RATIO = 0.04  # radio mínimo aceptado (% del lado corto)
MAX_RADIUS_RATIO = 0.22  # radio máximo aceptado (% del lado corto)

# --- IA (FastSAM) ---
CONF_IA         = 0.75   # confianza mínima de segmentación
                          #   bajo  (0.3-0.45) → detecta más piezas, incluso parciales
                          #   alto  (0.65-0.80) → solo piezas muy visibles y claras
CIRCULARIDAD_MIN = 0.50  # filtro de forma: 1.0 = círculo perfecto, 0.0 = cualquier forma

# ==============================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('data/resultados_bin_picking', timestamp)
os.makedirs(output_dir, exist_ok=True)

print("=== EXPERIMENTO 1: BIN PICKING ===")
print("Comparando Visión Clásica vs Deep Learning en múltiples imágenes...\n")
print(f"Config clásico → minDist={MIN_DIST_RATIO*100:.0f}%, param2={PARAM2}")
print(f"Config IA      → conf={CONF_IA}, circularidad>{CIRCULARIDAD_MIN}\n")

model = FastSAM('FastSAM-s.pt')

COLORES = [
    (255, 80,  80),  (80,  255, 80),  (80,  80,  255),
    (255, 255, 80),  (255, 80,  255), (80,  255, 255),
    (255, 160, 80),  (160, 80,  255), (80,  255, 160),
    (200, 200, 80),  (80,  200, 200), (200, 80,  200),
]

def filtrar_circulos_anidados(circulos):
    """Elimina círculos con solapamiento > 60% respecto a uno más grande (agujeros de arandelas)."""
    if len(circulos) == 0:
        return circulos
    circulos = sorted(circulos, key=lambda c: c[2], reverse=True)
    validos = []
    for cx, cy, r in circulos:
        solapado = False
        for px, py, pr in validos:
            distancia = np.sqrt((cx - px)**2 + (cy - py)**2)
            # Área de intersección aproximada entre dos círculos
            if distancia < pr:
                # El centro de este círculo está dentro del padre → muy solapado
                solapado = True
                break
            # Solapamiento por distancia entre bordes
            if distancia < (pr + r) * 0.6:
                solapado = True
                break
        if not solapado:
            validos.append((cx, cy, r))
    return validos

for filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, filename)
    if not os.path.isfile(image_path):
        continue

    print(f"Procesando: {filename}")

    img_original = cv2.imread(image_path)
    if img_original is None:
        print(f"  [!] Aviso: OpenCV no pudo leer '{filename}'. Saltando...\n")
        continue

    h, w = img_original.shape[:2]
    area_imagen = h * w

    # --- 1. VISIÓN CLÁSICA: HoughCircles + filtro de anidados ---
    img_clasica = img_original.copy()
    gray = cv2.cvtColor(img_clasica, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    circulos_raw = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * MIN_DIST_RATIO),
        param1=80,
        param2=PARAM2,
        minRadius=int(min(h, w) * MIN_RADIUS_RATIO),
        maxRadius=int(min(h, w) * MAX_RADIUS_RATIO),
    )

    num_clasico = 0
    if circulos_raw is not None:
        circulos_lista = [tuple(c) for c in np.round(circulos_raw[0]).astype(int)]
        circulos_filtrados = filtrar_circulos_anidados(circulos_lista)

        for i, (cx, cy, r) in enumerate(circulos_filtrados):
            cv2.circle(img_clasica, (cx, cy), r, (0, 255, 0), 2)
            cv2.circle(img_clasica, (cx, cy), 3, (0, 255, 0), -1)
            cv2.putText(img_clasica, f'#{i+1}', (cx - 12, cy - r - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            num_clasico += 1

    cv2.putText(img_clasica, f'Piezas detectadas: {num_clasico}',
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img_clasica, 'Vision Clasica (HoughCircles)',
                (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    nombre_base = os.path.splitext(filename)[0]
    out_clasica = os.path.join(output_dir, f'clasico_{nombre_base}.jpg')
    cv2.imwrite(out_clasica, img_clasica)
    print(f"  -> Clásico: {num_clasico} arandelas detectadas")

    # --- 2. DEEP LEARNING (IA): máscaras manuales ---
    results = model(image_path, conf=CONF_IA, iou=0.9, verbose=False)
    img_ia = img_original.copy()
    overlay = img_ia.copy()

    mascaras = results[0].masks
    num_ia = 0

    if mascaras is not None:
        masks_data = mascaras.data.cpu().numpy()

        mascaras_validas = []
        for m in masks_data:
            m_resized = cv2.resize(m, (w, h))
            binaria = (m_resized > 0.5).astype(np.uint8)
            area = binaria.sum()
            if not (0.005 * area_imagen < area < 0.25 * area_imagen):
                continue
            # Filtro de circularidad: descarta fragmentos irregulares
            contornos_m, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contornos_m:
                continue
            perimetro = cv2.arcLength(contornos_m[0], True)
            if perimetro == 0:
                continue
            circularidad = 4 * np.pi * area / (perimetro ** 2)
            if circularidad < CIRCULARIDAD_MIN:
                continue
            mascaras_validas.append(m_resized)

        # Suprimir máscaras cuyo centroide cae dentro de una máscara más grande (agujeros)
        mascaras_validas = sorted(mascaras_validas, key=lambda m: (m > 0.5).sum(), reverse=True)
        mascaras_finales = []
        for m in mascaras_validas:
            binaria = m > 0.5
            ys, xs = np.where(binaria)
            if len(xs) == 0:
                continue
            cx_m, cy_m = int(xs.mean()), int(ys.mean())
            dentro = any(grande[cy_m, cx_m] > 0.5 for grande in mascaras_finales)
            if not dentro:
                mascaras_finales.append(m)
        mascaras_validas = mascaras_finales

        for i, m in enumerate(mascaras_validas):
            color = COLORES[i % len(COLORES)]
            mascara_bool = m > 0.5
            overlay[mascara_bool] = color
            num_ia += 1

        cv2.addWeighted(overlay, 0.45, img_ia, 0.55, 0, img_ia)

        for i, m in enumerate(mascaras_validas):
            mascara_bool = m > 0.5
            ys, xs = np.where(mascara_bool)
            if len(xs) == 0:
                continue
            cx, cy = int(xs.mean()), int(ys.mean())
            cv2.putText(img_ia, f'#{i+1}', (cx - 12, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.putText(img_ia, f'Piezas detectadas: {num_ia}',
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(img_ia, 'Deep Learning (FastSAM)',
                (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

    out_ia = os.path.join(output_dir, f'ia_{nombre_base}.jpg')
    cv2.imwrite(out_ia, img_ia)
    print(f"  -> IA: {num_ia} piezas detectadas")

    # --- 3. COMPARATIVA LADO A LADO ---
    h_c, w_c = img_clasica.shape[:2]
    h_i, w_i = img_ia.shape[:2]
    h_target = max(h_c, h_i)

    if h_c != h_target:
        img_clasica = cv2.copyMakeBorder(img_clasica, 0, h_target - h_c, 0, 0,
                                          cv2.BORDER_CONSTANT, value=(0, 0, 0))
    if h_i != h_target:
        img_ia = cv2.copyMakeBorder(img_ia, 0, h_target - h_i, 0, 0,
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0))

    separador = np.zeros((h_target, 4, 3), dtype=np.uint8)
    comparativa = np.hstack([img_clasica, separador, img_ia])

    out_comp = os.path.join(output_dir, f'comparativa_{nombre_base}.jpg')
    cv2.imwrite(out_comp, comparativa)
    print(f"  -> Comparativa guardada\n")

print(f"¡Terminado! Resultados guardados en '{output_dir}'.")
