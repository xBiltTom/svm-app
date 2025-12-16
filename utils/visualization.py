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
    Crea un heatmap de la matriz de confusión con anotaciones claras
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Calcular porcentajes
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Crear anotaciones combinadas (cantidad y porcentaje)
    annot = np.array([[f'{value}\n({percent:.1f}%)' 
                       for value, percent in zip(row_cm, row_percent)]
                      for row_cm, row_percent in zip(cm, cm_percent)])
    
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Cantidad de predicciones'},
                linewidths=2, linecolor='white',
                ax=ax)
    
    ax.set_xlabel('Predicción del Modelo', fontsize=14, fontweight='bold')
    ax.set_ylabel('Clase Real', fontsize=14, fontweight='bold')
    ax.set_title(f'{title}\n(Valores: cantidad y % por fila)', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Agregar texto explicativo
    total_correct = np.trace(cm)
    total = cm.sum()
    accuracy = total_correct / total * 100
    
    ax.text(0.5, -0.15, f'Predicciones correctas: {total_correct}/{total} ({accuracy:.1f}%)',
            transform=ax.transAxes, ha='center', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
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
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Crear malla de puntos
    x_min, x_max = X_2d_scaled[:, 0].min() - 0.5, X_2d_scaled[:, 0].max() + 0.5
    y_min, y_max = X_2d_scaled[:, 1].min() - 0.5, X_2d_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Predecir en cada punto de la malla
    Z = temp_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plotear frontera de decisión con contornos más claros
    contour = ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu', levels=len(np.unique(y)))
    
    # Plotear puntos de datos con mejor contraste
    unique_classes = np.unique(y)
    colors = plt.cm.RdYlBu(np.linspace(0, 1, len(unique_classes)))
    
    for i, cls in enumerate(unique_classes):
        mask = y == cls
        class_name = class_names[i] if i < len(class_names) else f'Clase {cls}'
        ax.scatter(X_2d_scaled[mask, 0], X_2d_scaled[mask, 1], 
                  c=[colors[i]], edgecolors='black', linewidth=1.5,
                  s=120, alpha=0.8, label=class_name)
    
    # Plotear vectores de soporte con mayor énfasis
    support_vectors = temp_model.support_vectors_
    ax.scatter(support_vectors[:, 0], support_vectors[:, 1], 
              s=250, linewidth=3, facecolors='none', 
              edgecolors='red', label=f'Vectores de Soporte ({len(support_vectors)})',
              marker='o')
    
    ax.set_xlabel(f'{feature_x_name}\n(valores escalados)', fontsize=13, fontweight='bold')
    ax.set_ylabel(f'{feature_y_name}\n(valores escalados)', fontsize=13, fontweight='bold')
    ax.set_title(f'Frontera de Decisión del SVM\nKernel: {model.kernel} | C: {model.C}', 
                fontsize=15, fontweight='bold', pad=20)
    
    # Leyenda más clara
    ax.legend(loc='best', fontsize=11, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir explicación
    ax.text(0.02, 0.98, 'Regiones de color = Áreas de decisión\nPuntos rojos = Vectores críticos', 
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    return fig

def plot_roc_curve(model, X_test, y_test, scaler):
    """
    Genera la curva ROC para clasificación binaria de forma más descriptiva
    """
    X_test_scaled = scaler.transform(X_test)
    
    # Obtener probabilidades
    y_score = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calcular curva ROC
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Curva ROC principal con mejor grosor
    ax.plot(fpr, tpr, color='darkorange', lw=3, 
            label=f'Curva ROC (AUC = {roc_auc:.3f})')
    
    # Línea diagonal de referencia
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Clasificador Aleatorio (AUC = 0.500)')
    
    # Añadir texto interpretativo
    if roc_auc >= 0.9:
        interpretation = "Excelente"
        color = 'green'
    elif roc_auc >= 0.8:
        interpretation = "Muy Bueno"
        color = 'yellowgreen'
    elif roc_auc >= 0.7:
        interpretation = "Bueno"
        color = 'orange'
    else:
        interpretation = "Regular"
        color = 'red'
    
    ax.text(0.6, 0.2, f'Desempeño: {interpretation}', 
           fontsize=14, fontweight='bold', color=color,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor=color, linewidth=2))
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)\n(Proporción de negativos incorrectamente clasificados)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)\n(Proporción de positivos correctamente clasificados)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Curva ROC - Característica Operativa del Receptor\nMayor área = Mejor discriminación', 
                fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir guía de interpretación
    guide_text = ('Interpretación AUC:\n'
                 '0.90-1.00 = Excelente\n'
                 '0.80-0.90 = Muy Bueno\n'
                 '0.70-0.80 = Bueno\n'
                 '0.60-0.70 = Regular\n'
                 '< 0.60 = Pobre')
    ax.text(0.02, 0.98, guide_text, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
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
    Visualiza los resultados de validación cruzada de forma clara y descriptiva
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_display = ['Accuracy\n(Exactitud Global)', 
                      'Precision\n(Acierto en Positivos)', 
                      'Recall\n(Sensibilidad)', 
                      'F1-Score\n(Balance P&R)']
    means = [cv_results[m]['mean'] for m in metrics_names]
    stds = [cv_results[m]['std'] for m in metrics_names]
    
    x = np.arange(len(metrics_display))
    width = 0.6
    
    # Colorear barras según su rendimiento
    colors = []
    for mean in means:
        if mean >= 0.9:
            colors.append('darkgreen')
        elif mean >= 0.8:
            colors.append('yellowgreen')
        elif mean >= 0.7:
            colors.append('orange')
        else:
            colors.append('red')
    
    bars = ax.bar(x, means, width, alpha=0.7, color=colors, 
                  yerr=stds, capsize=12, error_kw={'linewidth': 2.5, 'ecolor': 'black'})
    
    # Añadir valores en las barras
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        # Valor principal
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.03,
                f'{mean:.2%}',
                ha='center', va='bottom', fontweight='bold', fontsize=13)
        # Desviación estándar
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                f'±{std:.3f}',
                ha='center', va='center', fontweight='bold', fontsize=10,
                color='white')
    
    ax.set_xlabel('Métricas de Evaluación', fontsize=13, fontweight='bold')
    ax.set_ylabel('Valor de la Métrica', fontsize=13, fontweight='bold')
    ax.set_title(f'{title}\nPromedio de {len(cv_results["accuracy"]["scores"])} Folds de Validación Cruzada', 
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_display, fontsize=11)
    ax.set_ylim([0, 1.2])
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=2, alpha=0.5, 
              label='Línea base (50%)')
    
    # Leyenda de interpretación
    legend_text = ('Interpretación:\n'
                  '≥90% = Excelente\n'
                  '80-90% = Muy Bueno\n'
                  '70-80% = Bueno\n'
                  '<70% = Requiere mejora')
    ax.text(0.98, 0.98, legend_text, transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, edgecolor='black'))
    
    ax.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    return fig

def plot_cv_folds_comparison(fold_results, title="Rendimiento Individual de Cada Fold"):
    """
    Muestra el rendimiento de cada fold con una línea de tendencia clara
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    folds = [f['fold'] for f in fold_results]
    accuracy = [f['metrics']['accuracy'] for f in fold_results]
    precision = [f['metrics']['precision'] for f in fold_results]
    recall = [f['metrics']['recall'] for f in fold_results]
    f1 = [f['metrics']['f1'] for f in fold_results]
    
    # Gráfico 1: Líneas de tendencia
    ax1.plot(folds, accuracy, 'o-', label='Accuracy', linewidth=3, markersize=10, color='blue')
    ax1.plot(folds, precision, 's-', label='Precision', linewidth=3, markersize=10, color='green')
    ax1.plot(folds, recall, '^-', label='Recall', linewidth=3, markersize=10, color='orange')
    ax1.plot(folds, f1, 'd-', label='F1-Score', linewidth=3, markersize=10, color='red')
    
    # Añadir líneas de promedio
    ax1.axhline(y=np.mean(accuracy), color='blue', linestyle='--', alpha=0.3)
    ax1.axhline(y=np.mean(f1), color='red', linestyle='--', alpha=0.3)
    
    ax1.set_xlabel('Número de Fold', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Valor de la Métrica', fontsize=13, fontweight='bold')
    ax1.set_title('Evolución de Métricas por Fold\n(Busca consistencia entre folds)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(folds)
    
    # Gráfico 2: Tabla resumen con colores
    metrics_summary = {
        'Accuracy': [f'{np.mean(accuracy):.2%}', f'{np.std(accuracy):.3f}', f'{min(accuracy):.2%}', f'{max(accuracy):.2%}'],
        'Precision': [f'{np.mean(precision):.2%}', f'{np.std(precision):.3f}', f'{min(precision):.2%}', f'{max(precision):.2%}'],
        'Recall': [f'{np.mean(recall):.2%}', f'{np.std(recall):.3f}', f'{min(recall):.2%}', f'{max(recall):.2%}'],
        'F1-Score': [f'{np.mean(f1):.2%}', f'{np.std(f1):.3f}', f'{min(f1):.2%}', f'{max(f1):.2%}']
    }
    
    ax2.axis('tight')
    ax2.axis('off')
    
    table_data = [['Métrica', 'Promedio', 'Desv. Est.', 'Mínimo', 'Máximo']]
    for metric, values in metrics_summary.items():
        table_data.append([metric] + values)
    
    table = ax2.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.22, 0.19, 0.19, 0.19, 0.19])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 3)
    
    # Colorear header
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Colorear filas alternadas
    for i in range(1, 5):
        color = '#E7E6E6' if i % 2 == 0 else 'white'
        for j in range(5):
            table[(i, j)].set_facecolor(color)
            if j == 0:
                table[(i, j)].set_text_props(weight='bold')
    
    ax2.set_title('Resumen Estadístico de Validación Cruzada\n(Baja desviación = Modelo estable)', 
                 fontsize=14, fontweight='bold', pad=15)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig

def plot_cv_scores_distribution(cv_results, title="Consistencia del Modelo en Validación Cruzada"):
    """
    Visualiza la consistencia del modelo mostrando la variabilidad entre folds
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    metrics_names = ['accuracy', 'precision', 'recall', 'f1']
    metrics_display = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Gráfico 1: Violín plot más informativo que boxplot
    data = [cv_results[m]['scores'] for m in metrics_names]
    
    parts = ax1.violinplot(data, positions=range(len(metrics_display)), 
                           showmeans=True, showmedians=True, widths=0.7)
    
    # Colorear violines
    colors = ['blue', 'green', 'orange', 'red']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    
    # Añadir puntos individuales para ver todos los folds
    for i, (metric_data, color) in enumerate(zip(data, colors)):
        y = metric_data
        x = np.random.normal(i, 0.04, size=len(y))  # Jitter
        ax1.scatter(x, y, alpha=0.6, s=80, color=color, edgecolors='black', linewidth=1)
    
    ax1.set_xticks(range(len(metrics_display)))
    ax1.set_xticklabels(metrics_display, fontsize=12, fontweight='bold')
    ax1.set_ylabel('Valor de la Métrica', fontsize=13, fontweight='bold')
    ax1.set_title('Distribución de Scores por Fold\n(Distribuciones estrechas = Mayor estabilidad)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax1.set_ylim([0, 1.1])
    ax1.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax1.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    
    # Añadir anotaciones de variabilidad
    for i, m in enumerate(metrics_names):
        std = cv_results[m]['std']
        mean = cv_results[m]['mean']
        variability = "Baja" if std < 0.02 else "Media" if std < 0.05 else "Alta"
        color_var = 'green' if std < 0.02 else 'orange' if std < 0.05 else 'red'
        ax1.text(i, 0.05, f'Var: {variability}', ha='center', fontsize=9, 
                color=color_var, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Gráfico 2: Coeficiente de variación (CV) para evaluar estabilidad
    cvs = [(cv_results[m]['std'] / cv_results[m]['mean']) * 100 if cv_results[m]['mean'] > 0 else 0 
           for m in metrics_names]
    
    colors_cv = []
    for cv_val in cvs:
        if cv_val < 3:
            colors_cv.append('darkgreen')
        elif cv_val < 7:
            colors_cv.append('yellowgreen')
        elif cv_val < 12:
            colors_cv.append('orange')
        else:
            colors_cv.append('red')
    
    bars = ax2.barh(metrics_display, cvs, color=colors_cv, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Añadir valores
    for i, (bar, cv_val) in enumerate(zip(bars, cvs)):
        width = bar.get_width()
        ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{cv_val:.1f}%',
                ha='left', va='center', fontweight='bold', fontsize=12)
    
    ax2.set_xlabel('Coeficiente de Variación (%)', fontsize=13, fontweight='bold')
    ax2.set_title('Estabilidad del Modelo\n(Menor % = Más consistente entre folds)', 
                 fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlim([0, max(cvs) * 1.3 if max(cvs) > 0 else 15])
    ax2.grid(True, axis='x', alpha=0.3, linestyle='--')
    
    # Líneas de referencia
    ax2.axvline(x=3, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Excelente (<3%)')
    ax2.axvline(x=7, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Aceptable (<7%)')
    ax2.legend(loc='lower right', fontsize=10)
    
    # Texto interpretativo
    avg_cv = np.mean(cvs)
    if avg_cv < 3:
        interpretation = "El modelo es MUY ESTABLE"
        color_interp = 'green'
    elif avg_cv < 7:
        interpretation = "El modelo es ESTABLE"
        color_interp = 'yellowgreen'
    else:
        interpretation = "El modelo tiene VARIABILIDAD"
        color_interp = 'red'
    
    ax2.text(0.98, 0.02, f'{interpretation}\nCV promedio: {avg_cv:.1f}%', 
            transform=ax2.transAxes, fontsize=11, fontweight='bold',
            ha='right', va='bottom', color=color_interp,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                     edgecolor=color_interp, linewidth=2))
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    return fig

def plot_roc_with_auc(model, X_test, y_test, scaler, class_names=None):
    """
    Genera la curva ROC con AUC destacado para clasificación binaria o multiclase
    """
    from sklearn.metrics import roc_curve, auc, roc_auc_score
    from sklearn.preprocessing import label_binarize
    
    X_test_scaled = scaler.transform(X_test)
    
    # Obtener clases únicas en y_test (ordenadas)
    unique_classes = np.unique(y_test)
    n_classes = len(unique_classes)
    
    # Convertir class_names a lista si es necesario y validar
    if class_names is not None:
        if hasattr(class_names, '__iter__') and not isinstance(class_names, str):
            class_names = list(class_names)
        else:
            class_names = None
    
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
        # Usar model.classes_ para asegurar que las columnas coincidan con predict_proba
        model_classes = model.classes_
        y_test_bin = label_binarize(y_test, classes=model_classes)
        y_score = model.predict_proba(X_test_scaled)
        
        # Calcular ROC para cada clase
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for idx, class_idx in enumerate(model_classes):
            fpr[idx], tpr[idx], _ = roc_curve(y_test_bin[:, idx], y_score[:, idx])
            roc_auc[idx] = auc(fpr[idx], tpr[idx])
            
            # Mapear correctamente el índice de la clase al nombre
            if class_names is not None and len(class_names) > class_idx:
                try:
                    class_label = class_names[class_idx]
                except (IndexError, TypeError):
                    class_label = f'Clase {class_idx}'
            else:
                class_label = f'Clase {class_idx}'
            
            ax.plot(fpr[idx], tpr[idx], lw=2, 
                   label=f'{class_label} (AUC = {roc_auc[idx]:.3f})')
        
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

def plot_grid_search_results(results_df, top_n=10):
    """
    Visualiza los mejores resultados del Grid Search
    """
    top_results = results_df.head(top_n)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Gráfico 1: Top N configuraciones
    y_pos = np.arange(len(top_results))
    scores = top_results['mean_test_score'].values
    stds = top_results['std_test_score'].values
    
    # Crear etiquetas para cada configuración
    labels = []
    for idx, row in top_results.iterrows():
        kernel = row.get('svm__kernel', 'N/A')
        C = row.get('svm__C', 'N/A')
        gamma = row.get('svm__gamma', 'N/A')
        label = f"K:{kernel}, C:{C}, γ:{gamma}"
        labels.append(label)
    
    ax1.barh(y_pos, scores, xerr=stds, align='center', alpha=0.7, 
             color=plt.cm.viridis(scores / scores.max()))
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.invert_yaxis()
    ax1.set_xlabel('Score (mean ± std)', fontsize=11, fontweight='bold')
    ax1.set_title(f'Top {top_n} Configuraciones', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Gráfico 2: Comparación Train vs Test Score
    train_scores = top_results['mean_train_score'].values
    test_scores = top_results['mean_test_score'].values
    
    x = np.arange(len(top_results))
    width = 0.35
    
    ax2.bar(x - width/2, train_scores, width, label='Train Score', alpha=0.8, color='skyblue')
    ax2.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8, color='coral')
    
    ax2.set_xlabel('Configuración (por ranking)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score', fontsize=11, fontweight='bold')
    ax2.set_title('Train vs Test Score', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'#{i+1}' for i in range(len(top_results))])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([min(test_scores.min(), train_scores.min()) - 0.05, 1.05])
    
    plt.tight_layout()
    return fig

def plot_param_importance(results_df):
    """
    Analiza la importancia de cada parámetro en el Grid Search
    """
    # Identificar columnas de parámetros
    param_cols = [col for col in results_df.columns if col.startswith('svm__')]
    
    if len(param_cols) == 0:
        return None
    
    n_params = len(param_cols)
    fig, axes = plt.subplots(1, n_params, figsize=(6*n_params, 5))
    
    if n_params == 1:
        axes = [axes]
    
    for idx, param_col in enumerate(param_cols):
        ax = axes[idx]
        
        # Agrupar por parámetro y calcular score promedio
        grouped = results_df.groupby(param_col)['mean_test_score'].agg(['mean', 'std']).reset_index()
        grouped = grouped.sort_values('mean', ascending=False)
        
        # Gráfico de barras
        x_pos = np.arange(len(grouped))
        ax.bar(x_pos, grouped['mean'].values, yerr=grouped['std'].values, 
               alpha=0.7, color=plt.cm.Set2(np.linspace(0, 1, len(grouped))))
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped[param_col].values, rotation=45, ha='right')
        ax.set_ylabel('Score Promedio', fontsize=10, fontweight='bold')
        ax.set_title(param_col.replace('svm__', '').upper(), fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    return fig

