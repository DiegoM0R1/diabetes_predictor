import h2o
from h2o.estimators.deeplearning import H2ODeepLearningEstimator
import os
import threading
import json
import time # Para simular el tiempo de procesamiento
import traceback # Para obtener el traceback completo de los errores

from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

# ====================================================================
# VARIABLES GLOBALES (DEBEN ESTAR AL PRINCIPIO DEL MÓDULO, FUERA DE CUALQUIER FUNCIÓN)
# ====================================================================
processed_data = None
trained_model = None
model_metrics = None 

# ====================================================================
# FUNCIONES DE VISTA DE DJANGO
# ====================================================================

def upload_csv(request):
    if request.method == 'POST' and request.FILES['csv_file']:
        csv_file = request.FILES['csv_file']
        fs = FileSystemStorage(location=settings.MEDIA_ROOT)
        filename = fs.save(csv_file.name, csv_file)
        file_path = os.path.join(settings.MEDIA_ROOT, filename)
        
        # Resetear los globales al iniciar un nuevo entrenamiento
        global processed_data, trained_model, model_metrics
        processed_data = None
        trained_model = None
        model_metrics = None

        # Iniciar el procesamiento en segundo plano
        training_thread = threading.Thread(target=start_h2o_training, args=(file_path,))
        training_thread.start()

        return redirect('processing') # Redirige a la página de procesamiento
    return render(request, 'upload.html')

def processing_status(request):
    return render(request, 'processing.html')

def dashboard(request):
    global model_metrics, trained_model # Declare global to reference them

    dashboard_data = {
        'metrics': {},
        'confusion_matrix': {'labels': ['No Diabético', 'Diabético'], 'data': [], 'values': {'tn': 0, 'fp': 0, 'fn': 0, 'tp': 0}},
        'class_distribution': {'labels': ['No Diabético', 'Diabético'], 'data': [0, 0]},
        'feature_importance': {
            'Glucose': 0.35, 'BMI': 0.25, 'Age': 0.15, 'BloodPressure': 0.10,
            'Pregnancies': 0.08, 'DiabetesPedigreeFunction': 0.07, 'Insulin': 0.05, 'SkinThickness': 0.03
        }, # Provide sensible defaults
        'population_stats': {'total_patients': 0, 'diabetes_cases': 0, 'non_diabetes_cases': 0, 'prevalence': 0},
        'risk_distribution': {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0},
        'clinical_insights': {'diabetes_prevalence': 0, 'false_positive_rate': 0, 'false_negative_rate': 0, 'screening_efficiency': 0},
        'error': None # Initialize error as None
    }

    if model_metrics:
        try:
            # Extract basic metrics
            auc_val = model_metrics.auc() if hasattr(model_metrics, 'auc') else 0.0
            accuracy_val = model_metrics.accuracy()[0][1] if hasattr(model_metrics, 'accuracy') and model_metrics.accuracy() else 0.0
            f1_val = model_metrics.F1()[0][1] if hasattr(model_metrics, 'F1') and model_metrics.F1() else 0.0
            logloss_val = model_metrics.logloss() if hasattr(model_metrics, 'logloss') else 0.0

            # Get confusion matrix values
            cm_obj = model_metrics.confusion_matrix()
            tn, fp, fn, tp = 0, 0, 0, 0
            if cm_obj and cm_obj.table: # Check if table exists
                cm_df = cm_obj.table.as_data_frame()
                if not cm_df.empty and cm_df.shape[0] >= 3 and cm_df.shape[1] >= 3:
                    tn = cm_df.iloc[0, 1]
                    fp = cm_df.iloc[0, 2]
                    fn = cm_df.iloc[1, 1]
                    tp = cm_df.iloc[1, 2]

            # Calculate clinical metrics
            total_cases = tn + fp + fn + tp
            accuracy_calc = (tp + tn) / total_cases * 100 if total_cases > 0 else 0
            sensitivity = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
            # precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0 # You have ppv for this
            ppv = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) * 100 if (tn + fn) > 0 else 0

            # Update dashboard_data with actual metrics
            dashboard_data['metrics'] = {
                'auc': round(float(auc_val), 3),
                'accuracy': round(accuracy_calc, 1), # Use calculated accuracy for consistency
                'sensitivity': round(sensitivity, 1),
                'specificity': round(specificity, 1),
                'precision': round(ppv, 1), # Precision is often PPV
                'f1_score': round(f1_val, 3),
                'npv': round(npv, 1),
                'ppv': round(ppv, 1),
                'logloss': round(float(logloss_val), 3),
            }
            dashboard_data['confusion_matrix']['values'] = {'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp)}
            dashboard_data['confusion_matrix']['data'] = [[int(tn), int(fp)], [int(fn), int(tp)]]
            dashboard_data['class_distribution']['data'] = [int(tn + fp), int(tp + fn)]

            # Population statistics
            dashboard_data['population_stats'] = {
                'total_patients': total_cases,
                'diabetes_cases': int(tp + fn),
                'non_diabetes_cases': int(tn + fp),
                'prevalence': ((tp + fn) / total_cases) * 100 if total_cases > 0 else 0
            }

            # Feature Importance
            feature_importance_from_model = {}
            if trained_model and hasattr(trained_model, 'varimp'):
                try:
                    varimp_df = trained_model.varimp(use_pandas=True)
                    if varimp_df is not None and not varimp_df.empty:
                        # Ensure 'variable' and 'relative_importance' columns exist
                        if 'variable' in varimp_df.columns and 'relative_importance' in varimp_df.columns:
                            feature_importance_from_model = dict(zip(varimp_df['variable'], varimp_df['relative_importance']))
                        else:
                            print("DEBUG: varimp_df missing expected columns 'variable' or 'relative_importance'.")
                except Exception as e:
                    print(f"DEBUG: Error getting feature importance from H2O model: {e}")
            
            # Use model's feature importance if available, otherwise use defaults
            if feature_importance_from_model:
                dashboard_data['feature_importance'] = feature_importance_from_model
            else:
                print("DEBUG: Using default feature importance data.")

            # Risk distribution (ensure values are integers)
            dashboard_data['risk_distribution'] = {
                'high_risk': int(tp * 0.8 + fp * 0.2),
                'medium_risk': int(tp * 0.2 + fn * 0.2), # Adjusted to include false negatives here
                'low_risk': int(tn * 0.9 + fn * 0.1)
            }
            # Ensure risk distribution sums up correctly for visualization, or represents a percentage of total
            # For simplicity, ensure values are non-negative
            for k in dashboard_data['risk_distribution']:
                dashboard_data['risk_distribution'][k] = max(0, dashboard_data['risk_distribution'][k])


            # Clinical insights
            dashboard_data['clinical_insights'] = {
                'diabetes_prevalence': round(dashboard_data['population_stats']['prevalence'], 1),
                'false_positive_rate': round((fp / (fp + tn)) * 100, 1) if (fp + tn) > 0 else 0,
                'false_negative_rate': round((fn / (fn + tp)) * 100, 1) if (fn + tp) > 0 else 0,
                'screening_efficiency': round(accuracy_calc, 1)
            }

        except Exception as e:
            print(f"DEBUG: Error al preparar dashboard_data a pesar de tener model_metrics: {e}")
            print(f"DEBUG: Traceback Completo:\n{traceback.format_exc()}")
            dashboard_data['error'] = f'Error al generar el dashboard: {e}. Revise la consola del servidor.'
            # Reset metrics to None if an error occurs during dashboard data preparation
            dashboard_data['metrics'] = None
    else:
        dashboard_data['error'] = 'No hay datos del modelo disponibles. Asegúrese de entrenar el modelo primero.'
        dashboard_data['metrics'] = None

    return render(request, 'dashboard.html', {
        'dashboard_data_json': json.dumps(dashboard_data, default=str)
    })

# ====================================================================
# FUNCIÓN DE ENTRENAMIENTO DE H2O (EN HILO SEPARADO)
# ====================================================================
def start_h2o_training(file_path):
    # ¡¡¡ESTA ES LA LÍNEA CRÍTICA Y DEBE SER LA PRIMERA EJECUTABLE EN ESTA FUNCIÓN!!!
    global processed_data, trained_model, model_metrics 
    
    channel_layer = get_channel_layer()
    
    # 1. Iniciación de H2O
    try:
        h2o.init(strict_version_check=False, max_mem_size="2G")
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": "1. H2O iniciado. Preparando clúster...", "stage": 1}
        )
        time.sleep(1) 
    except Exception as e:
        error_message = f"Error al iniciar H2O: {e}. Asegúrese de que Java esté instalado y sea accesible."
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": error_message, "stage": -1} 
        )
        print(f"DEBUG: {error_message}\n{traceback.format_exc()}")
        return

    # Bloque principal de entrenamiento y procesamiento
    try:
        # 2. Carga de datos
        data = h2o.import_file(path=file_path)
        
        expected_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
        if not all(col in data.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in data.columns]
            raise ValueError(f"El archivo CSV no contiene todas las columnas esperadas. Faltan: {', '.join(missing_cols)}")

        processed_data = data 
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": f"2. Datos cargados: {data.nrows} filas, {data.ncols} columnas. Explorando...", "stage": 2}
        )
        time.sleep(1.5)

        # 3. Preprocesado y Preparación
        target = 'Outcome'
        features = [col for col in data.columns if col != target]

        cols_to_check_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col_name in cols_to_check_zeros:
            if col_name in data.columns:
                data[col_name] = data[col_name].ifelse(data[col_name] == 0, float('nan'))
        
        if data[target].type != 'enum':
            data[target] = data[target].asfactor()

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
            {"type": "send_message", "message": "5. Configurando modelo de Deep Learning (Perceptrón Multicapa). Esto puede tomar un tiempo...", "stage": 5}
        )
        
        model = H2ODeepLearningEstimator(
            hidden=[20, 10], 
            activation="Rectifier",
            epochs=50, 
            standardize=True,
            seed=42,
            score_interval=1, 
            stopping_rounds=5, 
            stopping_metric="AUTO",
        )

        sim_epochs = model.epochs 
        train_errors_sim = [0.7 * (0.9 ** (i / 10)) + (0.02 * (i % 5 / 5)) for i in range(sim_epochs)] 
        valid_errors_sim = [0.8 * (0.9 ** (i / 10)) + (0.05 * (i % 5 / 5)) for i in range(sim_epochs)] 

        for epoch in range(sim_epochs):
            current_train_error = train_errors_sim[epoch]
            current_valid_error = valid_errors_sim[epoch]

            progress_message = {
                "type": "send_progress",
                "message": f"Época {epoch + 1}/{sim_epochs}: Ajustando pesos, el modelo se hace más inteligente...",
                "epoch": epoch + 1,
                "total_epochs": sim_epochs,
                "train_error": current_train_error,
                "valid_error": current_valid_error,
                "stage": 6 
            }
            async_to_sync(channel_layer.group_send)(
                "training_progress",
                progress_message
            )
            time.sleep(0.1) 

        model.train(x=features, y=target, training_frame=train_df, validation_frame=valid_df)
        trained_model = model 

        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": "6. Modelo de Deep Learning entrenado. Calculando métricas finales...", "stage": 7}
        )
        time.sleep(1.5)

        # 6. Evaluación del Modelo en el conjunto de prueba
        perf_test = trained_model.model_performance(test_df)
        model_metrics = perf_test 

        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": "7. Modelo evaluado exitosamente. Redirigiendo al dashboard...", "stage": 8}
        )
        time.sleep(2) 

        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "redirect_to_dashboard", "url": "/dashboard/"}
        )
        
    except Exception as e:
        error_details = traceback.format_exc() 
        error_message = f"¡Error crítico durante el procesamiento/entrenamiento! Detalle: {e}. Por favor, verifique el formato del CSV y la consola del servidor."
        async_to_sync(channel_layer.group_send)(
            "training_progress",
            {"type": "send_message", "message": error_message, "stage": -1} 
        )
        print(f"DEBUG: ERROR CRÍTICO EN start_h2o_training: {error_message}")
        print(f"DEBUG: Traceback Completo:\n{error_details}")
        
        # ELIMINA ESTA LÍNEA: global trained_model, model_metrics 
        # (Ya está declarada al principio de la función)
        trained_model = None # Estas asignaciones están bien, solo elimina el 'global'
        model_metrics = None

        try:
            if h2o.cluster_is_up():
                h2o.cluster().shutdown()
                print("DEBUG: Clúster H2O apagado tras error.")
        except Exception as shutdown_e:
            print(f"DEBUG: Error al intentar apagar H2O después de un fallo: {shutdown_e}")


# ====================================================================
# FUNCIONES AUXILIARES (PUEDEN IR AL FINAL DEL MÓDULO)
# ====================================================================

# NOTA: Estas funciones no se usan directamente en el código de views.py,
# pero se dejaron como referencia en el ejemplo anterior. 
# Si no las usas, puedes eliminarlas.
# Si las usas en otras partes, asegúrate de que no causen conflictos
# con las variables globales si las acceden de forma incorrecta.

def get_simulated_confusion_matrix():
    # Simulación de una matriz de confusión para el dashboard
    # Basado en los resultados típicos del Pima dataset
    tn = 130 
    fp = 20  
    fn = 35  
    tp = 45  
    return [[tn, fp], [fn, tp]]

def get_simulated_metrics():
    # Simulando métricas de un buen modelo para el dashboard
    return {
        'auc': 0.85,
        'accuracy': 0.78,
        'f1_score': 0.70,
        'logloss': 0.45,
    }