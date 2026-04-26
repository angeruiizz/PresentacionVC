import cv2
import os

# Carpetas de entrada y salida
input_dir = 'data/metal_surface_scratch'
output_dir = 'data/resultados_defectos'
os.makedirs(output_dir, exist_ok=True)

print("=== EXPERIMENTO 2: DETECCIÓN DE DEFECTOS ===")
print("Aplicando umbrales clásicos a múltiples imágenes para demostrar su limitación...\n")

for filename in os.listdir(input_dir):
    image_path = os.path.join(input_dir, filename)
    if not os.path.isfile(image_path): 
        continue
    
    print(f"Procesando: {filename}")
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"  [!] Aviso: OpenCV no pudo leer '{filename}'. Saltando...\n")
        continue
        
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Aplicamos el umbral fijo
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    img_resultado = img.copy()
    defectos_encontrados = 0
    
    for cnt in contours:
        if cv2.contourArea(cnt) > 50: 
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_resultado, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(img_resultado, 'Defecto', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            defectos_encontrados += 1
            
    out_path = os.path.join(output_dir, f'resultado_{filename}.jpg')
    cv2.imwrite(out_path, img_resultado)
    
    print(f"  -> {defectos_encontrados} 'defectos' detectados (incluyendo falsos positivos por la luz).")
    print(f"  -> Guardado en '{out_path}'\n")

print("="*60)
print("CONCLUSIÓN: Notaréis que el umbral '100' funciona bien en algunas fotos y fatal en otras")
print("dependiendo de la iluminación. Esta es la justificación perfecta para la presentación")
print("sobre por qué se necesitan algoritmos de Deep Learning que se adapten a la luz.")
print("="*60)