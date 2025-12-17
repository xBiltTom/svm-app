from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import numpy as np
import joblib
import pandas as pd

def train_svm(X_train, y_train, kernel='rbf', C=1.0, gamma='scale', degree=3, random_state=42):
    """
    Entrena un modelo SVM con los parámetros especificados
    """
    # Escalar datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Crear y entrenar modelo
    model = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        degree=degree,
        random_state=random_state,
        probability=True  # Para poder obtener probabilidades
    )
    
    model.fit(X_train_scaled, y_train)
    
    return model, scaler

def predict_svm(model, X, scaler):
    """
    Realiza predicciones con el modelo entrenado
    """
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)
    return predictions

def predict_proba_svm(model, X, scaler):
    """
    Obtiene las probabilidades de predicción
    """
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)
    return probabilities

def get_model_metrics(y_true, y_pred):
    """
    Calcula métricas de evaluación del modelo
    """
    # Determinar si es clasificación binaria o multiclase
    n_classes = len(np.unique(y_true))
    average = 'binary' if n_classes == 2 else 'weighted'
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }
    
    return metrics

def get_classification_report(y_true, y_pred, target_names=None):
    """
    Genera un reporte de clasificación detallado
    """
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        zero_division=0
    )
    return report

def get_confusion_matrix(y_true, y_pred):
    """
    Calcula la matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    return cm

def save_model(model, scaler, filepath):
    """
    Guarda el modelo y el scaler en disco
    """
    try:
        joblib.dump({
            'model': model,
            'scaler': scaler
        }, filepath)
        return True
    except Exception as e:
        print(f"Error al guardar el modelo: {str(e)}")
        return False

def load_model(filepath):
    """
    Carga un modelo previamente guardado
    """
    try:
        data = joblib.load(filepath)
        return data['model'], data['scaler']
    except Exception as e:
        print(f"Error al cargar el modelo: {str(e)}")
        return None, None

def cross_validate_svm(X, y, kernel='rbf', C=1.0, gamma='scale', degree=3, cv=5, random_state=42):
    """
    Realiza validación cruzada del modelo con múltiples métricas
    SIN data leakage: escala dentro de cada fold
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import Pipeline
    
    # Crear pipeline que escala dentro de cada fold
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=random_state, probability=True))
    ])
    
    # Crear StratifiedKFold para mantener proporción de clases
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Calcular métricas manualmente para cada fold
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Entrenar pipeline (escala y entrena)
        pipeline.fit(X_train_fold, y_train_fold)
        
        # Predecir (escala con el mismo scaler)
        y_pred = pipeline.predict(X_val_fold)
        
        # Calcular métricas
        metrics = get_model_metrics(y_val_fold, y_pred)
        accuracy_scores.append(metrics['accuracy'])
        precision_scores.append(metrics['precision'])
        recall_scores.append(metrics['recall'])
        f1_scores.append(metrics['f1'])
    
    accuracy_scores = np.array(accuracy_scores)
    precision_scores = np.array(precision_scores)
    recall_scores = np.array(recall_scores)
    f1_scores = np.array(f1_scores)
    
    return {
        'accuracy': {
            'scores': accuracy_scores,
            'mean': accuracy_scores.mean(),
            'std': accuracy_scores.std()
        },
        'precision': {
            'scores': precision_scores,
            'mean': precision_scores.mean(),
            'std': precision_scores.std()
        },
        'recall': {
            'scores': recall_scores,
            'mean': recall_scores.mean(),
            'std': recall_scores.std()
        },
        'f1': {
            'scores': f1_scores,
            'mean': f1_scores.mean(),
            'std': f1_scores.std()
        },
        'cv_folds': cv
    }

def perform_stratified_kfold(X, y, kernel='rbf', C=1.0, gamma='scale', degree=3, n_splits=5, random_state=42):
    """
    Realiza validación cruzada con StratifiedKFold y retorna resultados detallados por fold
    SIN data leakage: escala dentro de cada fold
    """
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Escalar datos DENTRO del fold
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        
        # Entrenar modelo en este fold
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=random_state, probability=True)
        model.fit(X_train_fold_scaled, y_train_fold)
        
        # Predecir en validación
        y_pred_fold = model.predict(X_val_fold_scaled)
        
        # Calcular métricas
        metrics = get_model_metrics(y_val_fold, y_pred_fold)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'metrics': metrics
        })
    
    return fold_results
