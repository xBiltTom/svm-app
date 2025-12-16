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

def grid_search_svm(X, y, param_grid=None, cv=5, scoring='accuracy', random_state=42):
    """
    Búsqueda de hiperparámetros óptimos usando GridSearchCV con Pipeline
    SIN data leakage: escala dentro de cada fold
    """
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    
    if param_grid is None:
        # Grid completo de parámetros
        param_grid = {
            'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'svm__C': [0.01, 0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto'],
            'svm__degree': [2, 3, 4, 5]  # Solo para poly
        }
    
    # Crear pipeline para evitar data leakage
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=random_state))
    ])
    
    # StratifiedKFold para mantener proporción de clases
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # GridSearchCV con pipeline
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=skf, 
        scoring=scoring, 
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X, y)
    
    # Extraer el mejor modelo y scaler del pipeline
    best_pipeline = grid_search.best_estimator_
    best_scaler = best_pipeline.named_steps['scaler']
    best_model = best_pipeline.named_steps['svm']
    
    # Organizar resultados
    results = {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': best_model,
        'best_scaler': best_scaler,
        'best_pipeline': best_pipeline,
        'cv_results': grid_search.cv_results_,
        'n_combinations': len(grid_search.cv_results_['params'])
    }
    
    return results

def create_param_grid(search_mode='quick'):
    """
    Crea grids de parámetros predefinidos según el modo de búsqueda
    
    Modes:
    - 'quick': Búsqueda rápida con pocos parámetros
    - 'balanced': Búsqueda balanceada (recomendado)
    - 'exhaustive': Búsqueda exhaustiva (puede tardar mucho)
    """
    if search_mode == 'quick':
        return {
            'svm__kernel': ['linear', 'rbf'],
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto']
        }
    elif search_mode == 'balanced':
        return {
            'svm__kernel': ['linear', 'rbf', 'poly'],
            'svm__C': [0.01, 0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'svm__degree': [2, 3, 4]
        }
    elif search_mode == 'exhaustive':
        return {
            'svm__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
            'svm__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'svm__gamma': ['scale', 'auto', 0.0001, 0.001, 0.01, 0.1, 1],
            'svm__degree': [2, 3, 4, 5, 6]
        }
    else:
        raise ValueError(f"Modo '{search_mode}' no reconocido. Use 'quick', 'balanced' o 'exhaustive'")

def get_grid_search_results_df(grid_search_results):
    """
    Convierte los resultados de GridSearchCV en un DataFrame ordenado
    """
    cv_results = grid_search_results['cv_results']
    
    # Crear DataFrame con resultados
    results_df = pd.DataFrame({
        'params': cv_results['params'],
        'mean_test_score': cv_results['mean_test_score'],
        'std_test_score': cv_results['std_test_score'],
        'mean_train_score': cv_results['mean_train_score'],
        'std_train_score': cv_results['std_train_score'],
        'rank': cv_results['rank_test_score']
    })
    
    # Expandir parámetros en columnas separadas
    params_df = pd.json_normalize(results_df['params'])
    results_df = pd.concat([params_df, results_df.drop('params', axis=1)], axis=1)
    
    # Ordenar por score
    results_df = results_df.sort_values('mean_test_score', ascending=False)
    
    return results_df
