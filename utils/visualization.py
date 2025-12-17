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

def plot_roc_multiclass(model, X_test, y_test, scaler, class_names):
    """
    Visualiza la curva ROC para clasificación multiclase de forma clara e intuitiva
    Muestra ROC por clase y promedio micro/macro
    """
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize
    from itertools import cycle
    
    X_test_scaled = scaler.transform(X_test)
    
    # Obtener clases únicas
    classes = np.unique(y_test)
    n_classes = len(classes)
    
    # Binarizar las etiquetas
    y_test_bin = label_binarize(y_test, classes=classes)
    
    # Obtener probabilidades predichas
    y_score = model.predict_proba(X_test_scaled)
    
    # Calcular ROC y AUC para cada clase
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Calcular micro-average ROC curve y AUC
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    # Calcular macro-average ROC curve y AUC
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Colores para cada clase
    colors = cycle(['navy', 'darkorange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan'])
    
    # Plotear ROC para cada clase
    for i, color in zip(range(n_classes), colors):
        class_name = class_names[i] if i < len(class_names) else f'Clase {i}'
        ax.plot(fpr[i], tpr[i], color=color, lw=2.5, alpha=0.8,
               label=f'{class_name} (AUC = {roc_auc[i]:.3f})')
    
    # Plotear micro-average
    ax.plot(fpr["micro"], tpr["micro"],
           label=f'Promedio Micro (AUC = {roc_auc["micro"]:.3f})',
           color='deeppink', linestyle=':', lw=4, alpha=0.9)
    
    # Plotear macro-average
    ax.plot(fpr["macro"], tpr["macro"],
           label=f'Promedio Macro (AUC = {roc_auc["macro"]:.3f})',
           color='darkblue', linestyle='--', lw=4, alpha=0.9)
    
    # Línea diagonal de referencia
    ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Clasificador Aleatorio (AUC = 0.500)')
    
    # Configuración de ejes
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Tasa de Falsos Positivos (FPR)\n(Proporción de negativos mal clasificados)', 
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)\n(Proporción de positivos bien clasificados)', 
                 fontsize=12, fontweight='bold')
    ax.set_title('Curvas ROC - Clasificación Multiclase\nMayor área bajo la curva = Mejor discriminación por clase', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Leyenda fuera del gráfico para no obstruir
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Añadir cuadro explicativo
    explanation = (
        'Interpretación:\n'
        '• Micro-promedio: Agregación global\n'
        '• Macro-promedio: Promedio simple\n'
        '• AUC cercano a 1.0 = Excelente\n'
        '• AUC cercano a 0.5 = Aleatorio'
    )
    ax.text(0.02, 0.98, explanation, transform=ax.transAxes, 
           fontsize=9, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85, edgecolor='black'))
    
    # Evaluación general del modelo
    avg_auc = np.mean([roc_auc[i] for i in range(n_classes)])
    if avg_auc >= 0.9:
        evaluation = "EXCELENTE discriminación"
        eval_color = 'darkgreen'
    elif avg_auc >= 0.8:
        evaluation = "MUY BUENA discriminación"
        eval_color = 'green'
    elif avg_auc >= 0.7:
        evaluation = "BUENA discriminación"
        eval_color = 'orange'
    else:
        evaluation = "Discriminación REGULAR"
        eval_color = 'red'
    
    ax.text(0.98, 0.02, f'Promedio AUC: {avg_auc:.3f}\n{evaluation}', 
           transform=ax.transAxes, fontsize=11, fontweight='bold',
           ha='right', va='bottom', color=eval_color,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, 
                    edgecolor=eval_color, linewidth=2))
    
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

