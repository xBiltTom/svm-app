import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc, RocCurveDisplay
from sklearn.preprocessing import label_binarize
import pandas as pd

# Configurar estilo de visualizaciones
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def plot_confusion_matrix(y_true, y_pred, class_names, title="Matriz de Confusión"):
    """
    Crea un heatmap de la matriz de confusión
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cantidad'})
    
    ax.set_xlabel('Predicción', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor Real', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def plot_decision_boundary(model, scaler, X, y, feature_x_idx, feature_y_idx, 
                          class_names, feature_x_name, feature_y_name):
    """
    Visualiza la frontera de decisión del modelo SVM en 2D
    """
    # Seleccionar solo las dos features para visualización
    X_2d = X[:, [feature_x_idx, feature_y_idx]]
    X_2d_scaled = scaler.transform(X)[:, [feature_x_idx, feature_y_idx]]
    
    # Crear un modelo temporal con solo 2 features
    from sklearn.svm import SVC
    temp_model = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma)
    temp_model.fit(X_2d_scaled, y)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear malla de puntos
    x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
    y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predecir en cada punto de la malla
    Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plotear frontera de decisión
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    
    # Plotear puntos de datos
    scatter = ax.scatter(X_2d_scaled[:, 0], X_2d_scaled[:, 1], 
                        c=y, cmap='viridis', edgecolors='black', 
                        s=100, alpha=0.7)
    
    # Plotear vectores de soporte
    support_vectors = temp_model.support_vectors_
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], 
              s=200, linewidth=2, facecolors='none', 
              edgecolors='red', label='Vectores de Soporte')
    
    ax.set_xlabel(feature_x_name, fontsize=12, fontweight='bold')
    ax.set_ylabel(feature_y_name, fontsize=12, fontweight='bold')
    ax.set_title('Frontera de Decisión del SVM', fontsize=14, fontweight='bold', pad=20)
    ax.legend()
    
    # Añadir colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Clase', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_roc_curve(model, X_test, y_test, scaler):
    """
    Genera la curva ROC para clasificación binaria
    """
    X_test_scaled = scaler.transform(X_test)
    
    # Obtener probabilidades
    y_score = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'Curva ROC (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Clasificador Aleatorio')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tasa de Verdaderos Positivos', fontsize=12, fontweight='bold')
    ax.set_title('Curva ROC (Receiver Operating Characteristic)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_feature_importance(feature_names, importances, title="Importancia de Features"):
    """
    Visualiza la importancia de las características
    Nota: SVM lineal tiene coeficientes que pueden interpretarse como importancia
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    indices = np.argsort(importances)[::-1]
    
    ax.bar(range(len(importances)), importances[indices], color='steelblue', alpha=0.8)
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
    
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importancia', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_class_distribution(y, class_names, title="Distribución de Clases"):
    """
    Visualiza la distribución de clases en el dataset
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    unique, counts = np.unique(y, return_counts=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique)))
    bars = ax.bar([class_names[i] for i in unique], counts, color=colors, alpha=0.8)
    
    # Añadir valores en las barras
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Clases', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cantidad', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_training_history(train_scores, test_scores, param_name, param_values):
    """
    Visualiza el rendimiento del modelo para diferentes valores de hiperparámetros
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(param_values, train_scores, 'o-', label='Entrenamiento', 
            linewidth=2, markersize=8)
    ax.plot(param_values, test_scores, 's-', label='Prueba', 
            linewidth=2, markersize=8)
    
    ax.set_xlabel(param_name, fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Rendimiento vs {param_name}', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(df, feature_columns, title="Matriz de Correlación"):
    """
    Visualiza la matriz de correlación entre features
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = df[feature_columns].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1, 
                cbar_kws={"shrink": 0.8})
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def plot_metrics_comparison(metrics_dict, title="Comparación de Métricas"):
    """
    Compara múltiples métricas en un gráfico de barras
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics_dict))
    width = 0.35
    
    train_metrics = [v[0] for v in metrics_dict.values()]
    test_metrics = [v[1] for v in metrics_dict.values()]
    
    ax.bar(x - width/2, train_metrics, width, label='Entrenamiento', alpha=0.8)
    ax.bar(x + width/2, test_metrics, width, label='Prueba', alpha=0.8)
    
    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_dict.keys())
    ax.legend()
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_cv_results(cv_results, title="Resultados de Validación Cruzada"):
    """
    Visualiza los resultados de validación cruzada con múltiples métricas
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_display = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    means = [cv_results[m]['mean'] for m in metrics_names]
    stds = [cv_results[m]['std'] for m in metrics_names]
    
    x = np.arange(len(metrics_display))
    width = 0.6
    
    bars = ax.bar(x, means, width, alpha=0.8, color='steelblue', 
                  yerr=stds, capsize=10, error_kw={'linewidth': 2, 'ecolor': 'darkred'})
    
    # Añadir valores en las barras
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.3f}\n±{std:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax.set_xlabel('Métricas', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_display)
    ax.set_ylim([0, 1.15])
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    plt.tight_layout()
    return fig

def plot_cv_folds_comparison(fold_results, title="Comparación de Métricas por Fold"):
    """
    Compara las métricas de cada fold en la validación cruzada
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    folds = [f['fold'] for f in fold_results]
    accuracy = [f['metrics']['accuracy'] for f in fold_results]
    precision = [f['metrics']['precision'] for f in fold_results]
    recall = [f['metrics']['recall'] for f in fold_results]
    f1 = [f['metrics']['f1'] for f in fold_results]
    
    x = np.arange(len(folds))
    width = 0.2
    
    ax.bar(x - 1.5*width, accuracy, width, label='Accuracy', alpha=0.8)
    ax.bar(x - 0.5*width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x + 0.5*width, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + 1.5*width, f1, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Fold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels([f'Fold {f}' for f in folds])
    ax.legend(loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_cv_scores_distribution(cv_results, title="Distribución de Scores por Métrica"):
    """
    Visualiza la distribución de scores en cada fold usando boxplot
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_display = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    data = [cv_results[m]['scores'] for m in metrics_names]
    
    bp = ax.boxplot(data, labels=metrics_display, patch_artist=True,
                    notch=True, showmeans=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    meanprops=dict(marker='D', markerfacecolor='green', markersize=8))
    
    ax.set_ylabel('Valor', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0, 1.1])
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_roc_with_auc(model, X_test, y_test, scaler, class_names=None):
    """
    Genera la curva ROC con AUC destacado para clasificación binaria o multiclase
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize
    
    X_test_scaled = scaler.transform(X_test)
    n_classes = len(np.unique(y_test))
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    if n_classes == 2:
        # Clasificación binaria
        y_score = model.predict_proba(X_test_scaled)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=3, 
                label=f'Curva ROC (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Clasificador Aleatorio (AUC = 0.500)')
        
        # Añadir punto óptimo
        optimal_idx = np.argmax(tpr - fpr)
        ax.plot(fpr[optimal_idx], tpr[optimal_idx], 'ro', markersize=10, 
                label=f'Punto Óptimo')
        
        # Añadir texto con AUC grande
        ax.text(0.6, 0.2, f'AUC = {roc_auc:.3f}', 
                fontsize=20, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    else:
        # Clasificación multiclase
        y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
        y_score = model.predict_proba(X_test_scaled)
        
        # Calcular ROC para cada clase
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            
            class_label = class_names[i] if class_names else f'Clase {i}'
            ax.plot(fpr[i], tpr[i], lw=2, 
                   label=f'{class_label} (AUC = {roc_auc[i]:.3f})')
        
        # ROC micro-average
        fpr_micro, tpr_micro, _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)
        
        ax.plot(fpr_micro, tpr_micro, 
                color='deeppink', linestyle=':', lw=3,
                label=f'Micro-promedio (AUC = {roc_auc_micro:.3f})')
        
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--',
                label='Clasificador Aleatorio')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12, fontweight='bold')
    ax.set_title('Curva ROC (Receiver Operating Characteristic)', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
