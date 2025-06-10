# 🧠 Predictor de Diabetes con IA (Django + H2O.ai)

Este proyecto es una aplicación web construida con Django que utiliza la plataforma de Machine Learning H2O.ai para entrenar un modelo de Deep Learning (Red Neuronal) capaz de predecir la probabilidad de que un paciente tenga diabetes, basándose en datos médicos.

La aplicación ofrece una interfaz interactiva para cargar datos, visualizar el proceso de entrenamiento del modelo en tiempo real y analizar los resultados a través de un dashboard con métricas y gráficos detallados.

**Autor:** [Diego Zulueta Mori](https://github.com/DiegoM0R1) <br>
**Repositorio del Proyecto:** [https://github.com/DiegoM0R1/diabetes_predictor](https://github.com/DiegoM0R1/diabetes_predictor)

## 📜 Tabla de Contenidos
- [Características Principales](#-características-principales)
- [Arquitectura del Sistema](#-arquitectura-del-sistema)
- [Stack Tecnológico](#-stack-tecnológico)
- [Prerrequisitos](#-prerrequisitos)
- [Instalación y Puesta en Marcha](#-instalación-y-puesta-en-marcha)
- [Modo de Uso](#-modo-de-uso)
- [Vistas de la Aplicación](#-vistas-de-la-aplicación)
- [Licencia](#-licencia)

## ✨ Características Principales

* [cite_start]**Carga de Datos Flexible:** Permite a los usuarios subir sus propios conjuntos de datos en formato CSV. 
* **Entrenamiento Asíncrono:** El entrenamiento del modelo se ejecuta en un proceso en segundo plano para no bloquear la interfaz de usuario.
* [cite_start]**Visualización en Tiempo Real:** Muestra el progreso del entrenamiento y la curva de aprendizaje (error vs. épocas) en vivo gracias a Django Channels (WebSockets). 
* **Dashboard Interactivo:** Presenta un análisis completo del rendimiento del modelo, incluyendo:
    * Métricas clave (Precisión, AUC, Sensibilidad, Especificidad).
    * Matriz de Confusión.
    * Gráfico de Importancia de Características (Feature Importance).
    * Estadísticas poblacionales y clínicas.

##  diagrama de arquitectura Arquitectura del Sistema

El sistema sigue una arquitectura web moderna que separa las responsabilidades y permite la comunicación en tiempo real.

![Diagrama de Flujo y Arquitectura](https://i.imgur.com/eB3t9sY.png)

> *Diagrama creado con Eraser.io. Puedes consultar la versión interactiva [aquí](https://app.eraser.io/workspace/PR6ZvdDFD3uYdrNKkLrL).*

## 🛠️ Stack Tecnológico

* **Backend:** Python, Django
* **Machine Learning:** H2O.ai
* **Comunicación en Tiempo Real:** Django Channels
* **Frontend:** HTML5, CSS3, JavaScript
* **Visualización de Datos:** Chart.js

## 📋 Prerrequisitos

Antes de empezar, asegúrate de tener instalado lo siguiente en tu sistema:
* Python (versión 3.8 o superior)
* `pip` (gestor de paquetes de Python)
* Java Development Kit (JDK) (versión 8 o superior), ya que es un requisito para H2O.ai.

## 🚀 Instalación y Puesta en Marcha

Sigue estos pasos para configurar el proyecto en tu entorno local:

1.  **Clona el repositorio:**
    ```bash
    git clone [https://github.com/DiegoM0R1/diabetes_predictor.git](https://github.com/DiegoM0R1/diabetes_predictor.git)
    cd diabetes_predictor
    ```

2.  **Crea y activa un entorno virtual:**
    * **Windows:**
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    * **macOS / Linux:**
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3.  **Instala las dependencias:**
    *(Asegúrate de tener un archivo `requirements.txt` en tu repositorio. Si no lo tienes, puedes crearlo con `pip freeze > requirements.txt`)*
    ```bash
    pip install -r requirements.txt
    ```

4.  **Realiza las migraciones de la base de datos de Django:**
    ```bash
    python manage.py migrate
    ```

5.  **Inicia el servidor de desarrollo:**
    ```bash
    python manage.py runserver
    ```

¡Listo! La aplicación debería estar corriendo en `http://127.0.0.1:8000/`.

## 👩‍💻 Modo de Uso

1.  Abre tu navegador y ve a `http://127.0.0.1:8000/`.
2.  [cite_start]En la página principal, sube un archivo CSV con datos de pacientes. [cite: 20] [cite_start]Si no tienes uno, puedes usar el dataset de ejemplo de Pima Indians, que puedes descargar [aquí](https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv). 
3.  Haz clic en "Iniciar Procesamiento y Predicción".
4.  Serás redirigido a una página que muestra el progreso del entrenamiento en tiempo real.
5.  Una vez finalizado, la aplicación te llevará automáticamente al dashboard de resultados para que analices el rendimiento del modelo.

## 🖼️ Vistas de la Aplicación

**1. Página de Carga de Datos**
![Página de Carga](https://i.imgur.com/rXo2VlR.png)

**2. Página de Procesamiento en Tiempo Real**
![Página de Procesamiento](https://i.imgur.com/2sY2T0G.png)

**3. Dashboard de Resultados**
![Dashboard de Resultados](https://imgur.com/pwzd3r9)


## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo `LICENSE` para más detalles.
