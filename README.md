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
git clone https://github.com/Sus306/PresentacionVC.git
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

## 4. Experimento 1 — Bin Picking

El "Bin Picking" requiere que el robot identifique piezas individuales dentro de un contenedor desordenado.

### Metodología
Se ha desarrollado un script en Python (`experimento_bin_picking.py`) que procesa imágenes de arandelas metálicas amontonadas. 
- **Enfoque Clásico:** Se aplicó un filtro Gaussiano y el algoritmo de detección de bordes de Canny mediante OpenCV.
- **Enfoque Moderno (IA):** Se implementó *FastSAM* (Segment Anything Model), un modelo fundacional *Zero-Shot* de la librería Ultralytics.

Comparativa de 3 iteraciones de ajuste de hiperparámetros sobre 3 escenas distintas.  
Ground truth conocido únicamente en la **imagen mixta** (10 piezas reales).

---

## Iteración 1 — Configuración base (permisiva)

```
MIN_DIST_RATIO   = 0.07
PARAM2           = 28
CONF_IA          = 0.4
CIRCULARIDAD_MIN = 0.25
```

| Imagen | Clásico | IA | Real |
|---|---|---|---|
| Apiladas | 124 | 68 | — |
| Dispersas | ~30 | 13 | — |
| Mixta | 11 | 10 | 10 |

### Imagen apiladas
`param2=28` acepta arcos muy débiles → HoughCircles detecta 124 círculos, incluyendo agujeros interiores y reflejos. `conf=0.4` hace que FastSAM acepte máscaras poco fiables → 68 detecciones con fragmentos y agujeros incluidos. Ambos métodos sobredetectan severamente.

### Imagen dispersas
HoughCircles detecta múltiples círculos por arandela (anillo exterior + agujero interior) al no tener filtro de anidados. FastSAM con `circularidad=0.25` acepta formas poco redondas pero contiene el ruido.

### Imagen mixta
Mejor resultado del clásico en esta escena (11 vs 10 real). Las piezas dispersas tienen borde completo → suficientes votos. FastSAM también acierta con 10. La configuración permisiva funciona bien cuando no hay oclusión.

> **Conclusión:** Umbral bajo = máxima sensibilidad → ambos métodos sobredetectan con oclusión. El clásico confunde agujeros con piezas, la IA acepta fragmentos irregulares. Funciona bien solo en escenas simples.

---

## Iteración 2 — Ajuste moderado

```
MIN_DIST_RATIO   = 0.12
PARAM2           = 36
CONF_IA          = 0.6
CIRCULARIDAD_MIN = 0.35
```

| Imagen | Clásico | IA | Real |
|---|---|---|---|
| Apiladas | 20 | 33 | — |
| Dispersas | ~10 | 13 | — |
| Mixta | 8 | **10** ✓ | 10 |

### Imagen apiladas
`minDist=12%` reduce falsos positivos del clásico (124→20) pero sigue subestimando por oclusión estructural. FastSAM baja de 68 a 33 gracias a `conf=0.6` — descarta máscaras poco confiables — pero sigue sobredetectando en escenas densas.

### Imagen dispersas
`param2=36` con `minDist=12%` reduce duplicados en el clásico. FastSAM mantiene resultados estables — `circularidad=0.35` empieza a filtrar mejor los fragmentos irregulares entre piezas de distintos tamaños.

### Imagen mixta
FastSAM alcanza exactamente 10 — **mejor resultado de la IA en toda la experimentación**. Los filtros más estrictos eliminan fragmentos sin perder piezas reales. HoughCircles baja a 8 — `param2=36` es más exigente y pierde 2 piezas con borde parcial.

> **Conclusión:** Subir `param2` y `conf` reduce el ruido en ambos métodos. El clásico empieza a perder piezas reales mientras la IA mantiene precisión. Primera divergencia clara entre métodos.

---

## Iteración 3 — Ajuste estricto

```
MIN_DIST_RATIO   = 0.18
PARAM2           = 45
CONF_IA          = 0.75
CIRCULARIDAD_MIN = 0.50
```

| Imagen | Clásico | IA | Real |
|---|---|---|---|
| Apiladas | 17 | 20 | — |
| Dispersas | ~11 | 13 | — |
| Mixta | 8 | **10** ✓ | 10 |

### Imagen apiladas
Configuración muy restrictiva — HoughCircles baja a 17, perdiendo muchas piezas reales al exigir bordes muy nítidos que la oclusión impide. FastSAM también baja a 20, su mínimo histórico en esta imagen, pero sigue siendo más cercano a la realidad que el clásico.

### Imagen dispersas
`circularidad=0.50` es el filtro más estricto probado — FastSAM mantiene 13, estable desde la iteración anterior. El clásico con `param2=45` pierde piezas pequeñas y de borde irregular que antes detectaba.

### Imagen mixta
FastSAM mantiene 10 por tercera iteración consecutiva — demuestra robustez ante cambios de hiperparámetros en escenas sin oclusión severa. HoughCircles se queda en 8 igual que en la iteración anterior — el ajuste fino ya no mejora ni empeora al clásico en esta escena.

> **Conclusión:** Configuración muy estricta — FastSAM se mantiene estable, HoughCircles pierde demasiadas piezas reales. El ajuste fino penaliza más al método clásico. La IA generaliza mejor ante distintas configuraciones.

---

## Conclusión global del experimento

| | HoughCircles | FastSAM |
|---|---|---|
| Escenas simples (sin oclusión) | ✓ Funciona bien con param2 bajo | ✓ Funciona bien en todas las configuraciones |
| Escenas con oclusión densa | ✗ Falla estructuralmente | ⚠ Sobredetecta pero se acerca más |
| Sensibilidad a hiperparámetros | Alta — resultados muy variables | Baja — resultados estables |
| Ground truth (10 piezas) | Máximo: 11 (It.1) — Mínimo: 8 | Máximo: 10 ✓ (It.2,3,4) |

### Por qué falla el clásico con oclusión
HoughCircles necesita ver suficiente borde continuo para acumular votos. Con piezas apiladas, la oclusión rompe ese borde → pocos votos por círculo → el sistema falla independientemente del `param2`. No es un problema de configuración, es una limitación estructural del método.

### Por qué la IA generaliza mejor
FastSAM segmenta regiones coherentes de textura, no formas geométricas predefinidas. Con oclusión parcial, la región sigue siendo visible → FastSAM la segmenta aunque el borde esté incompleto. Los filtros post-proceso (área, circularidad, centroide) corrigen los falsos positivos sin necesidad de reentrenamiento.


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
