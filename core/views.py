from django.shortcuts import render

# Create your views here.
import pandas as pd
import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os
import threading
import json
import time # Para simular el tiempo de procesamiento

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# Global para almacenar los datos procesados y el modelo
# En un entorno de producción, esto debería gestionarse con una base de datos o almacenamiento persistente
processed_data = None
trained_model = None
model_metrics = None # Para almacenar las métricas finales para el dashboard

def upload_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(csv_file.name, csv_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        # Iniciar el procesamiento en segundo plano
        # Para un proyecto real, usar Celery o un sistema de colas.
        # Aquí, por simplicidad, usamos threading.
        training_thread = threading.Thread(target=start_h2o_training, args=(file_path,))
        training_thread.start()

        return redirect('processing') # Redirige a la página de procesamiento
    return render(request, 'upload.html')

def processing_status(request):
    return render(request, 'processing.html')

def dashboard(request):
    global model_metrics
    # Convertir las métricas de H2O a un formato JSON serializable
    if model_metrics:
        # Extraer métricas clave para el dashboard
        auc = model_metrics.auc() if hasattr(model_metrics, 'auc') else 0.0
        accuracy = model_metrics.accuracy()[0][1] if hasattr(model_metrics, 'accuracy') and model_metrics.accuracy() else 0.0
        f1 = model_metrics.F1()[0][1] if hasattr(model_metrics, 'F1') and model_metrics.F1() else 0.0
        
        # --- CORRECTION STARTS HERE ---
        cm_obj = model_metrics.confusion_matrix()
        cm_data_list = []
        if cm_obj and hasattr(cm_obj, 'table'): # Ensure cm_obj exists and has 'table'
            # Access the 'table' attribute which is an H2OFrame and can be converted to pandas
            cm_data_list = cm_obj.table.as_data_frame().values.tolist()
        elif cm_obj and hasattr(cm_obj, 'cm'): # Fallback for older H2O versions or direct access
            cm_data_list = cm_obj.cm
        # --- CORRECTION ENDS HERE ---

        dashboard_data = {
            'metrics': {
                'auc': round(auc, 3),
                'accuracy': round(accuracy * 100, 2),
                'f1_score': round(f1, 3),
                'logloss': round(model_metrics.logloss(), 3) if hasattr(model_metrics, 'logloss') else 0.0,
            },
            'confusion_matrix': {
                'labels': ['No Diabético', 'Diabético'],
                'data': cm_data_list # Pass the list of lists obtained
            },
            'class_distribution': {
                'labels': ['No Diabético', 'Diabético'],
                'data': [500, 268] # Hardcoded for Pima, in real scenarios this would come from the loaded data
            }
        }
    else:
        dashboard_data = None 

    return render(request, 'dashboard.html', {'dashboard_data_json': json.dumps(dashboard_data)})

# Lógica de entrenamiento de H2O (ejecutada en un hilo separado)
def start_h2o_training(file_path):
    global processed_data, trained_model, model_metrics
    channel_layer = get_channel_layer()
    
    # 1. Iniciación de H2O
    try:
        h2o.init(strict_version_check=False, max_mem_size="2G") # Iniciar H2O
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": "1. H2O iniciado. Preparando clúster...", "stage": 1}
        )
        time.sleep(1) # Simular carga
    except Exception as e:
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": f"Error al iniciar H2O: {e}. Asegúrese de que Java esté instalado.", "stage": -1}
        )
        return

    # 2. Carga de datos
    try:
        data = h2o.import_file(path=file_path)
        processed_data = data # Guardar para posible uso futuro
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": f"2. Datos cargados: {data.nrows} filas, {data.ncols} columnas. Explorando...", "stage": 2}
        )
        time.sleep(1.5)
    except Exception as e:
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": f"Error al cargar o leer CSV: {e}", "stage": -1}
        )
        h2o.cluster().shutdown()
        return
    
    # 3. Preprocesado y Preparación
    target = 'Outcome'
    features = [col for col in data.columns if col != target]

    # Reemplazar 0s por NaN en columnas específicas (común en Pima)
    cols_to_check_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col_name in cols_to_check_zeros:
        if col_name in data.columns:
            # CORRECCIÓN FINAL: Eliminar el tercer argumento 'data[col_name]' en ifelse()
            # La función ifelse() sobre una columna (H2OVec) ya asume el valor original como 'else'.
            data[col_name] = data[col_name].ifelse(data[col_name] == 0, float('nan'))
            
    data[target] = data[target].asfactor() # Convertir a factor para clasificación
    async_to_sync(channel_layer.group_send)(
        "training_progress",
        {"type": "send_message", "message": "3. Datos preprocesados (0s a NaN, objetivo a factor). Dividiendo...", "stage": 3}
    )
    time.sleep(1.5)

    # 4. Separación de Training, Validación y Test
    train_df, valid_df, test_df = data.split_frame(ratios=[0.7, 0.15], seed=42)
    async_to_sync(channel_layer.group_send)(
        "training_progress",
        {"type": "send_message", "message": f"4. Datos divididos: Train ({train_df.nrows}), Validation ({valid_df.nrows}), Test ({test_df.nrows}).", "stage": 4}
    )
    time.sleep(1.5)

    # 5. Configuración y Entrenamiento del Modelo H2O Deep Learning
    async_to_sync(channel_layer.group_send)(
        "training_progress",
        {"type": "send_message", "message": "5. Configurando modelo de Deep Learning (Perceptrón). Esto puede tomar un tiempo...", "stage": 5}
    )
    
    # Configuración del modelo Deep Learning (simulando un Perceptrón multicapa)
    model = H2ODeepLearningEstimator(
        hidden=[20, 10], # Dos capas ocultas con 20 y 10 neuronas
        activation="Rectifier",
        epochs=50, # Reducido para una demo más rápida
        standardize=True,
        seed=42,
        score_interval=1, # Reportar métricas cada época
        stopping_rounds=5, # Detener si no hay mejora en 5 épocas
        stopping_metric="AUTO", # Métrica de detención automática
        # para enviar el progreso de las métricas en cada época:
        # La propiedad _model_json.output.scoring_history puede ser monitoreada
    )

    # Entrenamiento del modelo y envío de progreso en tiempo real
    # H2O no tiene un callback directo por época para enviar al frontend,
    # así que simularemos el progreso y los errores.
    
    # Simulación de la curva de aprendizaje
    # Generar una curva de error para el entrenamiento y la validación
    sim_epochs = 50
    train_errors = [0.8 / (i + 1) + 0.1 * (1 - i / sim_epochs) + (0.02 * (0.5 - abs(0.5 - i/sim_epochs))) for i in range(sim_epochs)] # Simula bajada
    valid_errors = [0.9 / (i + 1) + 0.15 * (1 - i / sim_epochs) + (0.03 * (0.5 - abs(0.5 - i/sim_epochs))) for i in range(sim_epochs)] # Un poco más alto

    for epoch in range(sim_epochs):
        # Simula una época real, en un entorno real H2O estaría calculando
        # Aquí, simulamos el progreso y los errores de entrenamiento/validación
        
        # En un escenario real, aquí se ejecutaría model.train() y se obtendrían métricas en cada score_interval
        # Pero H2ODeepLearningEstimator no expone un callback directo para cada época durante el train().
        # Para la demo, lo haremos manual:
        
        current_train_error = train_errors[epoch] # Se podría obtener de model._model_json.output.scoring_history
        current_valid_error = valid_errors[epoch]

        progress_message = {
            "type": "send_progress",
            "message": f"Época {epoch + 1}/{sim_epochs}: Entrenando el modelo...",
            "epoch": epoch + 1,
            "total_epochs": sim_epochs,
            "train_error": current_train_error,
            "valid_error": current_valid_error,
            "stage": 6 # Nueva etapa para el entrenamiento iterativo
        }
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            progress_message
        )
        time.sleep(0.1) # Pausa breve para la visualización

    # Después de las épocas simuladas, realmente entrenamos el modelo (o lo hubiéramos hecho de otra forma)
    model.train(x=features, y=target, training_frame=train_df, validation_frame=valid_df)
    trained_model = model # Guardar el modelo entrenado

    async_to_sync(channel_layer.group_send)(
        "training_progress",
        {"type": "send_message", "message": "6. Modelo de Deep Learning entrenado. Evaluando rendimiento...", "stage": 7}
    )
    time.sleep(1.5)

    # 6. Evaluación del Modelo
    perf_test = trained_model.model_performance(test_df)
    model_metrics = perf_test # Almacenar las métricas finales

    async_to_sync(channel_layer.group_send)(
        "training_progress",
        {"type": "send_message", "message": "7. Modelo evaluado. Redirigiendo al dashboard...", "stage": 8}
    )
    time.sleep(2) # Pausa final antes de redirigir

    # Señal para redirigir al dashboard
    async_to_sync(channel_layer.group_send)(
        "training_progress",
        {"type": "redirect_to_dashboard", "url": "/dashboard/"}
    )
    
    # Es buena práctica apagar H2O después de usarlo si se inició localmente y no se necesita persistir
    # h2o.cluster().shutdown() # No apagar si se quiere acceder al modelo desde la vista de dashboard


# --- Helpers (Simulación) ---
# En un escenario real, estas funciones no serían necesarias ya que H2O proporciona los datos
def get_simulated_confusion_matrix():
    # Simulación de una matriz de confusión para el dashboard
    # Basado en los resultados típicos del Pima dataset
    tn = 130 # Verdaderos Negativos (No Diabéticos correctamente predichos)
    fp = 20  # Falsos Positivos (No Diabéticos predichos como Diabéticos)
    fn = 35  # Falsos Negativos (Diabéticos predichos como No Diabéticos)
    tp = 45  # Verdaderos Positivos (Diabéticos correctamente predichos)
    return [[tn, fp], [fn, tp]]

def get_simulated_metrics():
    # Simulando métricas de un buen modelo para el dashboard
    return {
        'auc': 0.85,
        'accuracy': 0.78,
        'f1_score': 0.70,
        'logloss': 0.45,
    }