import cv2
import os
from ultralytics import FastSAM

# Carpetas de entrada y salida
input_dir = 'data/pile_of_metal_washers'
output_dir = 'data/resultados_bin_picking'
os.makedirs(output_dir, exist_ok=True) # Crea la carpeta si no existe

print("=== EXPERIMENTO 1: BIN PICKING ===")
print("Comparando Visión Clásica vs Deep Learning en múltiples imágenes...\n")

# Cargamos el modelo de IA una sola vez
model = FastSAM('FastSAM-s.pt')

# Recorremos todas las imágenes de la carpeta
for filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, filename)
    if not os.path.isfile(image_path): 
        continue
    
    print(f"Procesando: {filename}")
    
    # --- 1. VISIÓN CLÁSICA ---
    img_clasica = cv2.imread(image_path)
    if img_clasica is None:
        print(f"  [!] Aviso: OpenCV no pudo leer '{filename}' (quizá por el formato). Saltando...\n")
        continue
        
    gray = cv2.cvtColor(img_clasica, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cv2.drawContours(img_clasica, contours, -1, (0, 255, 0), 2)
    
    out_clasica = os.path.join(output_dir, f'clasico_{filename}.jpg')
    cv2.imwrite(out_clasica, img_clasica)
    print(f"  -> Resultado clásico guardado")
    
    # --- 2. DEEP LEARNING (IA) ---
    results = model(image_path, conf=0.4, iou=0.9, verbose=False) # verbose=False para no ensuciar la terminal
    out_ia = os.path.join(output_dir, f'ia_{filename}.jpg')
    results[0].save(out_ia)
    print(f"  -> Resultado IA guardado\n")

print(f"¡Terminado! Revisa la carpeta '{output_dir}' para elegir la mejor comparativa para la presentación.")