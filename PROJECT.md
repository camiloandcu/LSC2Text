# LSC2Text: Reconocimiento y Traducción de Lenguaje de Señas Colombiano a Texto

**Tabla de Contenidos**

1. [¿Por qué LSC2Text?](#por-qué-lsc2text)
2. [El Problema y la Motivación](#el-problema-y-la-motivación)
3. [El Dataset](#el-dataset)
4. [Preprocesamiento de Imágenes](#preprocesamiento-de-imágenes)
5. [Extracción de Features](#extracción-de-features)
6. [Escalado de Features](#escalado-de-features)
7. [Máquina de Vectores de Soporte (SVM)](#modelos-de-aprendizaje-automático)
8. [Optimización de Parámetros](#optimización-de-parámetros)
9. [Arquitectura del Sistema](#arquitectura-del-sistema)

---

## Por qué LSC2Text?

LSC2Text es un sistema de reconocimiento automático de lengua de signos que convierte imágenes de gestos en texto. Este proyecto demuestra que es posible combinar técnicas de visión por computadora con aprendizaje automático para resolver un problema de accesibilidad real.

---

## El Problema y la Motivación

### ¿Qué problema estamos resolviendo?

La lengua de signos es la lengua natural de las personas sordas, pero muchos sistemas tecnológicos aún no la integran adecuadamente. Los traductores automáticos de lenguaje de signos a texto podrían mejorar significativamente la accesibilidad en comunicación en tiempo real, educación y redes sociales.

LSC2Text se enfoca en:

- **Reconocimiento visual de signos**: Procesar imágenes de signos del lenguaje de signos para identificar qué seña se está realizando
- **Clasificación automática**: Agrupar signos en categorías predefinidas con confianza medible
- **Aplicación práctica**: Crear una API y frontend accesibles para demostración

### ¿Por qué ahora?

- **Disponibilidad de datos**: Los datasets de lengua de signos son ahora más accesibles (ej: LSC70)
- **Hardware asequible**: GPUs y CPUs modernas hacen factible el procesamiento en tiempo real
- **Técnicas comprobadas**: Métodos como HOG y LBP funcionan bien incluso sin redes neuronales profundas
- **Oportunidad de impacto**: Cada mejora en el reconocimiento automático contribuye a la inclusión

### Contexto: Lingüística de la Lengua de Signos

La lengua de signos no es un sistema de mímica, sino una lengua con estructura gramatical compleja:

- **Parámetros articulatorios**: Ubicación, movimiento, forma de la mano, orientación y expresión facial
- **Composición de signos**: Cada seña combine estos parámetros simultáneamente (no secuencialmente como en lengua hablada)
- **Variabilidad individual**: Diferentes personas pueden realizar el mismo signo con variaciones personales

Este proyecto se enfoca en clasificar signos individuales, que es el primer paso hacia traducción de frases completas.

---

## El Dataset

### Descripción General

El proyecto utiliza el dataset **LSC70**, un conjunto de imágenes de Lenguaje de Señas Colombiana:

- **Origen**: Compilado por investigadores del procesamiento de lengua de signos
- **Cobertura**: 70 signos diferentes de uso frecuente
- **Composición**:
  - Total de imágenes: ~7,000
  - Signos únicos: 70 categorías
  - Variaciones: Múltiples observaciones por signo (diferentes personas, ángulos, condiciones de iluminación)

### Estructura de Carpetas

```
data/
├── raw/
│   ├── dataset.csv          # Índice original del dataset
│   ├── LSC70/
│   │   ├── LSC70AN/        # Señas del alfabeto
│   │   ├── LSC70ANH/       # Señas del alfabeto, mano
│   │   └── LSC70W/         # Señas generales
│   └── ...
├── processed/
│   └── dataset_lsc70anh_abcde.csv   # Índice con rutas
└── splits/
    └── dataset_lsc70anh_abcde.csv   # División train/val
```

### División del Dataset

El dataset se divide en tres conjuntos:

- **Entrenamiento (Train)**: 80% usado para entrenar el modelo
- **Validación (Validation)**: 20% usado para evaluar el modelo

Se usó StratifiedGroupKFold, porque se quería mantener un balance de las señas y que tampoco hubiese un leakage de data por usar participantes tanto en `val` como en `train`.

Ver: [scripts/split_dataset.py](scripts/split_dataset.py) para la lógica de división

---

## Preprocesamiento de Imágenes

### ¿Por qué Preprocesamiento?

Las imágenes crudas del dataset varían en:

- **Tamaño**: Diferentes resoluciones y relaciones de aspecto
- **Iluminación**: Brillo variable según condiciones de captura
- **Fondo**: Fondos heterogéneos
- **Escala**: Tamaño relativo del objeto en la imagen

El preprocesamiento normaliza estas variaciones para mejorar la consistencia del modelo.

### Pasos de Preprocesamiento

1. **Carga de Imagen**: Leer imagen en RGB o escala de grises según el contexto
2. **Conversión a Escala de Grises**: Reducir información a una sola dimensión (luminancia)
3. **Redimensionamiento**: Llevar todas las imágenes a 128×128 píxeles
4. **Normalización de Intensidad**: Escalar valores de píxeles a [0, 1]

Estos pasos se implementan en: [src/ml/preprocessing.py](src/ml/preprocessing.py)

### Decisiones de Diseño

- **128×128 píxeles**: Balance entre detalles y eficiencia computacional
- **Escala de grises**: La ubicación y forma del signo son más importantes que el color
- **Normalización lineal**: Rápida y predecible sin suposiciones de distribución

---

## Extracción de Features

Después del preprocesamiento, extraemos características numéricas que capturan la "forma" y "estructura" del signo. Usamos dos métodos complementarios:

### 1. HOG - Histograma de Gradientes Orientados

**¿Qué es?** HOG cuantifica la dirección de los cambios de intensidad en la imagen.

**Ecuación Base:**

```
HOG_bin(x,y) = ∑(w(i,j) × δ(∇I(i,j), θ_bin))
```

Donde:

- `∇I(i,j)` = gradiente en píxel (i,j) = [∂I/∂x, ∂I/∂y]
- `θ_bin` = orientación del bin (ej: 0°, 45°, 90°, ..., 315°)
- `w(i,j)` = peso (magnitud del gradiente)
- `δ()` = función que asigna píxeles a bins

**Pasos prácticos:**

1. Calcular gradientes (cambios de intensidad) en X e Y
2. Calcular magnitud y ángulo de cada píxel
3. Agrupar en histogramas por orientación (ej: 8 o 9 bins)
4. Promediar en celdas (ej: bloques de 8×8 píxeles)
5. Concatenar todos los histogramas

**¿Por qué HOG para signos?**

- Las manos tienen contornos y bordes claros
- La orientación de dedos y palma son críticas para la forma del signo
- HOG es robusto a pequeñas variaciones de escala

### 2. LBP - Patrón Binario Local

**¿Qué es?** LBP captura texturas comparando píxeles con sus vecinos.

**Ecuación Base:**

```
LBP(x,y) = ∑(s(I(neighbor) - I(x,y)) × 2^i)
```

Donde:

- `I(x,y)` = intensidad del píxel central
- `I(neighbor)` = intensidad del vecino i (ej: 8 vecinos en patrón de círculo)
- `s(x)` = función de signo: 1 si x ≥ 0, 0 si x < 0
- Resultado: número de 0 a 255 (8 vecinos = 8 bits)

**Pasos prácticos:**

1. Para cada píxel, comparar con sus 8 vecinos
2. Si vecino ≥ píxel central: bit=1, si no: bit=0
3. Convertir secuencia de bits a número decimal (0-255)
4. Histograma de valores LBP en regiones
5. Concatenar histogramas

**¿Por qué LBP para signos?**

- Captura microestructura de piel y texturas de manos
- Distintos signos tienen diferentes "texturas" de patrones de luz
- Computacionalmente muy rápido

### Configuración de Features

Ambos métodos se configuran mediante `FeatureConfig` en [src/ml/feature_extraction.py](src/ml/feature_extraction.py):

```python
@dataclass
class FeatureConfig:
    window_size: int = 64          # Tamaño de ventana (píxeles)
    stride: int = 8                # Paso entre ventanas (píxeles)
    num_cells: int = 8             # Células por dimensión en HOG
    hog_bins: int = 9              # Bins de orientación en HOG
    lbp_radius: int = 3            # Radio de vecindad para LBP
```

**Impacto de parámetros:**

- `window_size` mayor → features más globales pero menos locales
- `stride` menor → mayor sobreposición, features más densas
- `num_cells` mayor → histogramas más granulares
- `hog_bins` → balance entre detalles de orientación y variabilidad

---

## Escalado de Features

### ¿Por qué Escalado?

Después de extraer features HOG+LBP, obtenemos un vector de ~1344 dimensiones con valores heterogéneos:

- HOG: valores 0-9 (histogramas de 9 bins)
- LBP: valores 0-255 (suma de histogramas)

Modelos como SVM son **sensibles a la escala**: si una feature tiene rango [0,255] y otra [0,9], el modelo puede sobrepesarla.

### StandardScaler

Transformamos cada feature `x` a:

```
x_scaled = (x - mean) / std
```

Donde:

- `mean` = promedio de esa feature en TRAIN
- `std` = desviación estándar en TRAIN
- Resultado: media 0, desviación estándar 1

**Estadísticas de Escalado:**

Para el dataset LSC70ANH:

- **Dimensionalidad**: 1,344 features (combined HOG + LBP)
- **Media de features originales**: rango variado [0-255] según feature
- **Desv. Estándar pre-escalado**: típicamente 20-80 para HOG, 30-100 para LBP
- **Datos de entrenamiento**: ~4,200 imágenes utilizadas para calcular estadísticas

Las estadísticas se calculan SOLO en train set y se aplican consistentemente a validation y test.

**Decisiones:**

- Calculamos mean/std **SOLO en TRAIN**
- Aplicamos IGUALES a VALIDATION y TEST
- Esto evita "contaminación" de información de test en el modelo

**Impacto:**

- SVM converge más rápido (típicamente 2-3x más rápido)
- Mejor regularización (C se interpreta uniformemente)
- Mejores garantías de generalización

---

## Máquina de Vectores de Soporte (SVM)

**Idea intuitiva:** Encontrar un hiperplano (línea/plano) que mejor separe las clases.

**Función de decisión:**

```
f(x) = sign(w·φ(x) + b)
```

Donde:

- `w` = vector de pesos (normal del hiperplano)
- `φ(x)` = transformación vía kernel
- `b` = bias (desplazamiento)
- `sign()` = función signo (clase +1 o -1)

**Hiperparámetros principales:**

| Parámetro | Rango           | Significado                                                             |
| --------- | --------------- | ----------------------------------------------------------------------- |
| `C`       | 0.001 - 1000    | Penalización de errores; C pequeño = margen grande (más regularización) |
| `kernel`  | 'linear', 'rbf' | 'linear' = separación lineal simple; 'rbf' = separación no-lineal       |

**¿Por qué SVM?**

- Excelente con features bien diseñadas (HOG+LBP)
- Interpretable en dimensiones altas
- No requiere muchas muestras si features son buenas

---

## Optimización de Parámetros

### Búsqueda Bayesiana con Optuna

**¿Qué es Optuna?** Una librería Python que automáticamente busca hiperparámetros óptimos usando:

1. **Prueba aleatoria inicial** para explorar el espacio
2. **Observar resultados** para aprender qué zonas son prometedoras
3. **Enfoque en zonas prometedoras** para explotar el mejor comportamiento

**Objetivos de Optimización:**

La búsqueda de Optuna optimiza múltiples métricas según el contexto:

- **Exactitud (Accuracy)**: Porcentaje de predicciones correctas en el conjunto de validación
- **Macro F1-Score**: Media armónica ponderada, importante para clases desbalanceadas
- **Tiempo de entrenamiento**: Minimizar para iteraciones rápidas sin sacrificar calidad

### Optimización de Features

También optimizamos parámetros de HOG/LBP:

- `window_size`: ¿Qué resolución de ventana?
- `stride`: ¿Sobreposición completa o parcial?
- `num_cells`: ¿Cuántos histogramas?

Estos afectan el tamaño y calidad del vector de features.

**Proceso de Optimización de Features:**

Se ejecutaron **30 trials** de Optuna probando diferentes combinaciones, con un modelo de regresión de prueba:

- `lbp_radius`: 1-3 (radio de vecindad)
- `hog_orientations`: 6-12 (número de bins de orientación)
- `hog_pixels_per_cell`: 4-16 (granularidad de células)
- `hog_cells_per_block`: 1-3 (tamaño de bloques de normalización)

Cada trial:

1. Genera features con los parámetros propuestos
2. Entrena modelo SVM en train set
3. Evalúa en validation set
4. Registra la exactitud

Los mejores trials convergen hacia parámetros que capturan características de signos efectivamente.

### Optimización Conjunta

El verdadero poder es optimizar **feature parameters + model hyperparameters juntos**:

1. Generar features con ciertos parámetros
2. Entrenar modelo con ciertos hiperparámetros
3. Medir desempeño en validación
4. Repetir ajustando todo

Esto se implementa en: [scripts/optimize_train.py](scripts/optimize_train.py)

### Resultados de Optimización

**Optimización de Features (HOG+LBP parameters):**

![Feature Optimization Results](artifacts/experiments/feature_optimization/plots/best_trial.png)

Esta imagen muestra cómo el desempeño mejora a través de iteraciones de Optuna, descubriendo la mejor combinación de tamaño de ventana, stride y configuración de HOG/LBP.

**Optimización de Hiperparámetros del Modelo:**

![Model Hyperparameter Optimization](artifacts/experiments/hyperparam_optimization/best.png)

Esta imagen muestra el reporte de clasificación devuelto por el mejor modelo obtenido.

### Parámetros Óptimos Encontrados

**Configuración de Features (HOG+LBP):**

```json
{
  "lbp_radius": 3,
  "hog_orientations": 11,
  "hog_pixels_per_cell": 4,
  "hog_cells_per_block": 2
}
```

**Interpretación:**

- `lbp_radius=3`: Los vecinos de LBP se toman en un radio de 3 píxeles (8 vecinos)
- `hog_orientations=11`: 11 bins de orientación (en lugar del estándar 9)
- `hog_pixels_per_cell=4`: Células pequeñas de 4×4 píxeles para mayor granularidad
- `hog_cells_per_block=2`: Bloques de 2×2 células para normalización

---

## Arquitectura del Sistema

### Flujo Completo

```
┌─────────────────────────────────────────────────────────┐
│ Usuario sube imagen                                     │
│ (archivo PNG/JPG)                                       │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│ Preprocesamiento                                        │
│ • Carga de imagen                                       │
│ • Conversión a escala de grises                         │
│ • Redimensionamiento a 128×128                          │
│ • Normalización de intensidad                           │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│ Extracción de Features                                  │
│ • HOG (Histogramas de Gradientes Orientados)           │
│ • LBP (Patrones Binarios Locales)                      │
│ → Vector de ~1344 dimensiones                           │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│ Escalado (StandardScaler)                              │
│ • Media 0, Desviación Estándar 1                        │
│ → Vector preparado para modelo                          │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│ Predicción del Modelo                                   │
│ • SVM o MLP (según selección)                          │
│ • Salida: Probabilidad para cada signo (70 clases)     │
│ → Top-3 predicciones ordenadas                          │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│ API FastAPI                                             │
│ Endpoint: POST /predict                                 │
│ Respuesta: JSON con predicciones y confianzas          │
└──────────────────┬──────────────────────────────────────┘
                   ↓
┌─────────────────────────────────────────────────────────┐
│ Frontend (Server-Rendered HTML)                         │
│ • Jinja2 templates                                      │
│ • Interfaz de carga de imágenes                        │
│ • Visualización de resultados con confianzas           │
│ → Mostrar imagen + predicciones al usuario             │
└─────────────────────────────────────────────────────────┘
```

### Backend FastAPI

El backend se implementa en: [src/api/api.py](src/api/api.py)

**Endpoints principales:**

| Método | Ruta                | Descripción                                  |
| ------ | ------------------- | -------------------------------------------- |
| `GET`  | `/`                 | Página de carga de imágenes (HTML)           |
| `POST` | `/frontend/predict` | Procesa imagen del formulario, devuelve HTML |
| `POST` | `/predict`          | API JSON - recibe imagen, devuelve JSON      |
| `GET`  | `/health`           | Estado del servicio                          |
| `GET`  | `/metadata`         | Info del servicio (modelo, versión)          |

### Frontend (Jinja2 Templates)

Plantillas en: [src/api/templates/](src/api/templates/)

- `base.html`: Layout base con estilos (paleta de colores natural)
- `upload.html`: Formulario de carga
- `result.html`: Muestra predicciones + imagen subida
- `error.html`: Manejo de errores

### Visualización de Resultados

Las predicciones se muestran como:

1. **Ranking Top-3** de signos predichos
2. **Confianzas** expresadas como porcentajes
3. **Navegación** para hacer otra predicción
