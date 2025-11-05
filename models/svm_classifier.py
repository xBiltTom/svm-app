from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)
import numpy as np
import joblib

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
    """
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=random_state, probability=True)
    
    # Crear StratifiedKFold para mantener proporción de clases
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    # Calcular múltiples métricas
    accuracy_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='accuracy')
    precision_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='precision_weighted')
    recall_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='recall_weighted')
    f1_scores = cross_val_score(model, X_scaled, y, cv=skf, scoring='f1_weighted')
    
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
    """
    from sklearn.model_selection import StratifiedKFold
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_scaled, y)):
        X_train_fold, X_val_fold = X_scaled[train_idx], X_scaled[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Entrenar modelo en este fold
        model = SVC(kernel=kernel, C=C, gamma=gamma, degree=degree, random_state=random_state, probability=True)
        model.fit(X_train_fold, y_train_fold)
        
        # Predecir en validación
        y_pred_fold = model.predict(X_val_fold)
        
        # Calcular métricas
        metrics = get_model_metrics(y_val_fold, y_pred_fold)
        
        fold_results.append({
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'metrics': metrics
        })
    
    return fold_results

def grid_search_svm(X_train, y_train, param_grid=None):
    """
    Búsqueda de hiperparámetros óptimos usando GridSearchCV
    """
    from sklearn.model_selection import GridSearchCV
    
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    svm = SVC(probability=True)
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'best_model': grid_search.best_estimator_,
        'scaler': scaler
    }
