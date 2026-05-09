import cv2
import numpy as np
import os
from datetime import datetime
from skimage.morphology import skeletonize as sk_skeletonize

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join('data/resultados_sellados', timestamp)
os.makedirs(output_dir, exist_ok=True)

print("=== EXPERIMENTO 2: INSPECCIÓN DE SELLADOS (VMT ACS) ===\n")

# ==============================================================
#  HIPERPARÁMETROS — solo tocar estas líneas entre experimentos
# ==============================================================
UMBRAL_FIJO      = 120   # umbral para binarización clásica (0-255)
                          #   bajo  → detecta más, más sensible al ruido
                          #   alto  → detecta menos, pierde cordones poco contrastados
ANCHO_NOMINAL    = 30    # ancho esperado del cordón en píxeles
TOLERANCIA_ANCHO = 0.40  # variación máxima aceptable del ancho (0.40 = 40%)
                          #   bajo  → detecta engrosamientos pequeños (más alertas)
                          #   alto  → solo alerta engrosamientos grandes
MIN_GAP_PX       = 8     # píxeles mínimos de hueco para alertar interrupción
                          #   bajo  → detecta huecos pequeños (ruido puede activarlo)
                          #   alto  → solo alerta interrupciones grandes
# ==============================================================

IMG_H, IMG_W = 220, 750

def generar_cordon(con_interrupcion=False, con_engrosamiento=False,
                   con_desviacion=False, seed=42):
    """Genera imagen sintética de un cordón de silicona con defectos opcionales."""
    np.random.seed(seed)
    img = np.full((IMG_H, IMG_W, 3), 215, dtype=np.uint8)

    cy_base = IMG_H // 2

    for x in range(IMG_W):
        # Trayectoria: recta o sinusoidal
        desv = int(20 * np.sin(2 * np.pi * x / IMG_W)) if con_desviacion else 0
        cy = cy_base + desv

        # Ancho del cordón
        w = ANCHO_NOMINAL
        if con_engrosamiento and 300 < x < 430:
            factor = 1.0 + 0.95 * np.sin(np.pi * (x - 300) / 130)
            w = int(ANCHO_NOMINAL * factor)

        # Interrupción: no pintar esa zona
        if con_interrupcion and 160 < x < 215:
            continue

        y1 = max(0, cy - w // 2)
        y2 = min(IMG_H, cy + w // 2)
        img[y1:y2, x] = [55, 58, 65]

    # Ruido gaussiano para realismo
    ruido = np.random.normal(0, 14, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    return img


def analisis_clasico(img):
    """
    Método clásico: umbral fijo.
    Solo detecta SI hay cordón o NO — no puede localizar ni clasificar defectos.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, UMBRAL_FIJO, 255, cv2.THRESH_BINARY_INV)

    resultado = img.copy()
    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos pequeños (ruido)
    contornos_validos = [c for c in contornos if cv2.contourArea(c) > 200]
    cv2.drawContours(resultado, contornos_validos, -1, (0, 220, 0), 2)

    hay_cordon = len(contornos_validos) > 0
    estado = 'DETECTADO' if hay_cordon else 'NO DETECTADO'
    color_estado = (0, 220, 0) if hay_cordon else (0, 0, 220)

    cv2.putText(resultado, f'Cordon: {estado}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_estado, 2)
    cv2.putText(resultado, f'Umbral fijo = {UMBRAL_FIJO}',
                (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)
    cv2.putText(resultado, 'Metodo Clasico (Threshold)',
                (10, IMG_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    return resultado


def analisis_skeletonize(img):
    """
    Método avanzado: Skeletonize + DistanceTransform.
    Detecta y localiza interrupciones y engrosamientos con precisión de pixel.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu: umbral automático adaptado a la imagen
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Skeletonize: reduce el cordón a una línea de 1 pixel (eje central)
    skeleton = sk_skeletonize(thresh.astype(bool)).astype(np.uint8) * 255

    # DistanceTransform: en cada pixel del skeleton, valor = distancia al borde
    # → ancho local = 2 × distancia
    dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)

    # Recopilar zonas de interrupción y engrosamiento
    zonas_interrupcion = []
    zonas_engrosamiento = []

    tiene_cordon_prev = False
    inicio_gap = None

    anchos_por_col = {}
    centros_por_col = {}

    for x in range(IMG_W):
        col_skel = skeleton[:, x]
        puntos = np.where(col_skel > 0)[0]

        if len(puntos) == 0:
            if tiene_cordon_prev and inicio_gap is None:
                inicio_gap = x
            tiene_cordon_prev = False
        else:
            if inicio_gap is not None:
                gap = x - inicio_gap
                if gap >= MIN_GAP_PX:
                    zonas_interrupcion.append((inicio_gap, x))
                inicio_gap = None
            tiene_cordon_prev = True

            cy = int(np.mean(puntos))
            centros_por_col[x] = cy
            ancho_local = 2.0 * dist[cy, x]
            anchos_por_col[x] = ancho_local

            if ancho_local > ANCHO_NOMINAL * (1 + TOLERANCIA_ANCHO):
                zonas_engrosamiento.append(x)

    # --- Dibujar resultado ---
    resultado = img.copy()

    # Fondo semi-transparente para zonas de interrupción
    overlay = resultado.copy()
    for (x0, x1) in zonas_interrupcion:
        cv2.rectangle(overlay, (x0, 0), (x1, IMG_H), (0, 0, 180), -1)
    cv2.addWeighted(overlay, 0.35, resultado, 0.65, 0, resultado)

    # Línea skeleton coloreada por estado
    for x, cy in centros_por_col.items():
        ancho = anchos_por_col.get(x, 0)
        if x in zonas_engrosamiento:
            color = (0, 130, 255)   # naranja → engrosamiento
        else:
            color = (0, 220, 0)     # verde → OK
        cv2.circle(resultado, (x, cy), 1, color, -1)

    # Etiquetas de zonas
    for (x0, x1) in zonas_interrupcion:
        xm = (x0 + x1) // 2
        cv2.putText(resultado, f'INTERRUPCION ({x1-x0}px)',
                    (max(0, xm - 55), IMG_H - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 80, 255), 1)

    if zonas_engrosamiento:
        x0e = min(zonas_engrosamiento)
        x1e = max(zonas_engrosamiento)
        xm = (x0e + x1e) // 2
        cv2.putText(resultado, f'ENGROSAMIENTO',
                    (max(0, xm - 45), 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 130, 255), 1)

    # Resumen de alertas
    alertas = []
    for (x0, x1) in zonas_interrupcion:
        alertas.append(f'Interrupcion x={x0}-{x1} ({x1-x0}px)')
    if zonas_engrosamiento:
        alertas.append(f'Engrosamiento x={min(zonas_engrosamiento)}-{max(zonas_engrosamiento)}')

    n = len(alertas)
    color_n = (0, 220, 0) if n == 0 else (0, 80, 255)
    cv2.putText(resultado, f'Alertas: {n}',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color_n, 2)
    y = 58
    for a in alertas:
        cv2.putText(resultado, a, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)
        y += 22

    cv2.putText(resultado, 'Metodo Avanzado (Skeletonize + Otsu)',
                (10, IMG_H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
    cv2.putText(resultado, 'Verde=OK  Naranja=Engrosamiento  Rojo=Interrupcion',
                (10, IMG_H - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    return resultado, alertas


# --- Escenarios ---
escenarios = [
    {'nombre': '1_perfecto',
     'interrupcion': False, 'engrosamiento': False, 'desviacion': False},
    {'nombre': '2_interrupcion',
     'interrupcion': True,  'engrosamiento': False, 'desviacion': False},
    {'nombre': '3_engrosamiento',
     'interrupcion': False, 'engrosamiento': True,  'desviacion': False},
    {'nombre': '4_combinado',
     'interrupcion': True,  'engrosamiento': True,  'desviacion': True},
]

for esc in escenarios:
    print(f"Procesando: {esc['nombre']}")

    img = generar_cordon(
        con_interrupcion=esc['interrupcion'],
        con_engrosamiento=esc['engrosamiento'],
        con_desviacion=esc['desviacion']
    )

    clasico   = analisis_clasico(img)
    avanzado, alertas = analisis_skeletonize(img)

    # Guardar individuales
    cv2.imwrite(os.path.join(output_dir, f'original_{esc["nombre"]}.jpg'), img)
    cv2.imwrite(os.path.join(output_dir, f'clasico_{esc["nombre"]}.jpg'),  clasico)
    cv2.imwrite(os.path.join(output_dir, f'avanzado_{esc["nombre"]}.jpg'), avanzado)

    # Comparativa lado a lado: original | clásico | avanzado
    sep = np.zeros((IMG_H, 4, 3), dtype=np.uint8)
    comparativa = np.hstack([img, sep, clasico, sep, avanzado])
    cv2.imwrite(os.path.join(output_dir, f'comparativa_{esc["nombre"]}.jpg'), comparativa)

    resumen = alertas if alertas else ['Sin defectos']
    for r in resumen:
        print(f'  -> {r}')
    print()

print(f"Terminado! Resultados en '{output_dir}'.")
