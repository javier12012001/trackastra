# Trackastra - Extensión : Segmentación TFG, Hiperparámetros y Benchmark

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Napari](https://img.shields.io/badge/Visualización-Napari-indigo)
![Status](https://img.shields.io/badge/Estado-Desarrollo%20TFG-green)

Este repositorio es una **versión extendida y fork del proyecto original [Trackastra](https://github.com/trackastra/trackastra)** (Helmholtz AI). 

Ha sido desarrollado como parte de un **Proyecto de la asignatura Procesamiento de Imagenes Digitales** con el objetivo de comparar estrategias de segmentación, optimizar hiperparámetros y evaluar métricas de tracking en secuencias de imágenes biológicas.

---

##  Novedades de esta versión

Esta extensión añade una capa de experimentación sobre el framework original:

### 1. Nuevos Módulos de Segmentación
Implementación de métodos alternativos para alimentar el algoritmo de tracking:
* **Watershed Mejorado:** Preprocesamiento con normalización 8-bits, filtros de suavizado y umbralización automática.
* **Integración con Cellpose:** Uso directo del modelo `cyto3` para segmentaciones basadas en Deep Learning.
* **Pipeline Híbrido:** Capacidad de inyectar estas máscaras en Trackastra para medir el impacto de la segmentación en la calidad del tracking.

### 2. Control de Hiperparámetros
Interfaz centralizada para ajustar variables críticas sin modificar el código fuente:
* Filtros de preprocesamiento.
* Parámetros del algoritmo Watershed.
* Configuración de inferencia de Cellpose (diámetro, flow threshold).
* Configuración interna del tracking de Trackastra.

### 3. Benchmark Unificado (`benchmark3.ipynb`)
Un flujo de trabajo completo que automatiza:
1.  **Carga** de imágenes.
2.  **Segmentación** (Watershed vs Cellpose).
3.  **Tracking** con Trackastra.
4.  **Conversión** a formato estándar CTC (Cell Tracking Challenge).
5.  **Evaluación** automática de métricas SEG (Segmentación) y TRA (Tracking).
6.  **Visualización** interactiva en napari.

---

##  Métodos Incluidos

| Método | Descripción |
| :--- | :--- |
| **Watershed** | Normalización, filtrado y transformación de distancia para separar objetos conectados. Ideal para imágenes limpias. |
| **Cellpose** | Modelo `cyto3` preentrenado. Robusto para morfologías celulares complejas y alta densidad. |
| **Trackastra** | Algoritmo de tracking principal. En esta versión es agnóstico a la fuente de la máscara. |
| **Métricas CTC** | Implementación del cálculo oficial de métricas TRA y SEG para validación científica. |

---
## Enlace al repositorio

https://github.com/javier12012001/trackastra

##  Estructura del Repositorio
### Estructura del Repositorio

```text
trackastra
├── Codigo de segmentacion de mascaras TFG
│   ├── CellposeTrainedModel.py
│   └── Script de ejecución de mascaras.txt
├── Documentación Código.pdf
├── benchmark_final.ipynb
├── enlaces a bases de datos.txt
└── hiperparámetros.ipynb
