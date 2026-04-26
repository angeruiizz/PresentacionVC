# Memoria del Proyecto: Visión por Computador en la Industria

## De las técnicas deterministas clásicas al Deep Learning

### Datos del Documento
- **Asignatura:** Visión por Computador 
- **Universidad:** Universitat Rovira i Virgili (URV)
- **Autores:** Marina Oteiza Álvarez, Susana Triviño Nortes, Angelina Ruiz Jiménez

---

## 1. Introducción y Objetivos

El presente documento detalla la investigación y los experimentos prácticos realizados a raíz del análisis del artículo *"Sistemas de Visión Artificial Industrial y Medición por Láser"* (VMT / Pepperl+Fuchs). El artículo original expone el paradigma de la automatización mediante sistemas clásicos de visión artificial para el guiado de robots (Robot-Vision) y el control de calidad (Machine-Vision). 

Aunque estas soluciones clásicas son robustas en entornos controlados, presentan limitaciones severas ante variaciones de iluminación y solapamiento de piezas. Por ello, el objetivo de este trabajo ha sido **proponer e implementar soluciones alternativas** basadas en Inteligencia Artificial y Deep Learning, apoyándonos en literatura científica reciente, para demostrar empíricamente el salto cualitativo que estas nuevas tecnologías aportan a la industria.

## 2. Estructura del Proyecto y Rutas Relativas

El proyecto está estructurado de la siguiente manera para garantizar su correcta trazabilidad. Todas las rutas mencionadas a continuación son relativas a la raíz del repositorio (`./`):

* **`./data/`**: Directorio principal de imágenes.
  * `./data/pile_of_metal_washers/`: Contiene las imágenes originales de prueba para el experimento de *Bin Picking*.
  * `./data/metal_surface_scratch/`: Contiene las imágenes originales de prueba para el experimento de detección de defectos.
  * `./data/resultados_bin_picking/`: Directorio autogenerado tras la ejecución del código. Contiene las comparativas visuales (Clásico vs IA) del experimento 1.
  * `./data/resultados_defectos/`: Directorio autogenerado tras la ejecución del código. Contiene los resultados visuales del experimento 2.
* **`./src/`**: Código fuente de los algoritmos.
  * `./src/experimento_bin_picking.py`: Script de Python para segmentación de instancias.
  * `./src/experimento_defectos.py`: Script de Python para detección de anomalías.
* **`./docs/`**: Documentación teórica del proyecto.
  * `./docs/Presentacion_VC.pdf`: Presentación de diapositivas utilizada en la exposición oral.
  * `./docs/Memoria_Experimentos.pdf`: Memoria descriptiva de los métodos y conclusiones.
  * `./docs/*.pdf`: Artículos científicos (*papers*) utilizados como referencia bibliográfica.
* **`./requirements.txt`**: Archivo con el listado estricto de dependencias de Python necesarias.

---

## 3. Cómo Ejecutar los Experimentos

Sigue estos pasos desde tu terminal para replicar los experimentos en tu máquina local. **Asegúrate de estar situado en la raíz del repositorio** antes de empezar.

### Paso 1: Preparar el entorno virtual
Para evitar conflictos con las librerías de tu sistema operativo, recomendamos crear y activar un entorno virtual:

```bash
#lonar el repositorio**
git clone [Añadir vuestro enlace de GitHub aquí]
cd PresentacionVC

# Crear el entorno virtual en la raíz del proyecto
python3 -m venv .venv

# Activar el entorno virtual (Linux / macOS)
source .venv/bin/activate

# Activar el entorno virtual (Windows)
.venv\Scripts\activate

#Instalar dependencias
pip install -r ./requirements.txt

#Ejecutar experimentos
python ./src/experimento_bin_picking.py
python ./src/experimento_defectos.py
```

---

## 4. Experimento 1: Segmentación de Instancias para *Bin Picking*

El "Bin Picking" requiere que el robot identifique piezas individuales dentro de un contenedor desordenado.

### Metodología
Se ha desarrollado un script en Python (`experimento_bin_picking.py`) que procesa imágenes de arandelas metálicas amontonadas. 
- **Enfoque Clásico:** Se aplicó un filtro Gaussiano y el algoritmo de detección de bordes de Canny mediante OpenCV.
- **Enfoque Moderno (IA):** Se implementó *FastSAM* (Segment Anything Model), un modelo fundacional *Zero-Shot* de la librería Ultralytics.

### Resultados obtenidos (data\resultados_bin_picking/)

![Imagen Original]
Imagen original de piezas metálicas amontonadas.*

![Resultado Clásico]
Resultado de la Visión Clásica (Filtro Canny). El sistema es incapaz de separar las piezas debido a los reflejos y sombras, detectando un exceso de ruido perjudicial para la cinemática del robot.*

![Resultado IA]
Resultado con FastSAM. La red neuronal aísla perfectamente cada instancia de forma robusta, generando máscaras precisas ignorando las oclusiones y cambios de luz.*

### Ajuste de hiperparametros

iteración 20260426_224534
Clásico: Canny → HoughCircles (minDist=7%, param2=35, minRadius=3%)
IA: .plot() → renderizado manual con paleta de colores + transparencia 45%
Filtro área máscaras: 0.3%–25% de imagen
Problema: HoughCircles detectaba agujeros interiores (doble conteo)


Iteración 20260426_225935
filtrar_circulos_anidados: descarta círculos contenidos dentro de otro (distancia + r < pr * 1.05)
Resultado: 16 clásico / 22 IA (real: 10)


Iteración 20260426_231006
minDist: 7% → 12% (más separación entre centros)
param2: 35 → 40 (más estricto, menos falsos positivos)
Filtro anidados: contención exacta → solapamiento 60% (más tolerante a descentrados)
conf IA: 0.4 → 0.6
Filtro de circularidad 4πA/P² > 0.35 → descarta fragmentos irregulares
Resultado: 8 clásico / 17 IA (real: 10)


Iteración 20260426_231207
param2: 40 → 36 
IA: supresión por centroide → máscaras ordenadas por área, se descarta cualquiera cuyo centro cae dentro de una ya aceptada → elimina agujeros interiores como el #65
Resultado: ~10 clásico / ~10 IA 

## 4. Experimento 2: Control de Calidad e Integridad (Defectos)

La inspección de calidad busca identificar arañazos o marcas en superficies mecánicas.

### Metodología
A través del script `experimento_defectos.py`, analizamos imágenes de superficies metálicas con rasguños utilizando el método tradicional de umbralización (*Thresholding*).

### Resultados obtenidos (data\resultados_defectos)

![Defecto Original]
Imagen de una superficie metálica con defectos.*

![Detección Thresholding] 
Detección clásica por Thresholding.*

### Análisis del resultado
Aunque el algoritmo clásico ha logrado enmarcar el defecto, el código requiere introducir un valor de umbral rígido programado manualmente (en nuestro caso, `100`). Hemos comprobado que si la imagen es ligeramente más oscura o más clara, este valor deja de funcionar, generando falsos positivos o ignorando el rasguño. Esto corrobora empíricamente las referencias analizadas: para una línea de producción real, es imperativo el uso de IA (Aprendizaje No Supervisado) capaz de adaptarse dinámicamente a la variabilidad lumínica y de textura de la fábrica.
