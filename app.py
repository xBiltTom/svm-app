import streamlit as st
import pandas as pd
import numpy as np
from utils.data_processing import load_data, preprocess_data, split_data
from models.svm_classifier import train_svm, predict_svm, get_model_metrics, cross_validate_svm, perform_stratified_kfold
from utils.visualization import (
    plot_confusion_matrix, plot_decision_boundary, plot_feature_importance, 
    plot_roc_curve, plot_roc_with_auc, plot_cv_results, 
    plot_cv_folds_comparison, plot_cv_scores_distribution
)

st.set_page_config(
    page_title="SVM Classifier App",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Máquinas de Vectores de Soporte (SVM)")
st.markdown("### Aplicación interactiva para clasificación con SVM")

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración del Modelo")

# Subida de archivo
uploaded_file = st.sidebar.file_uploader(
    "📁 Cargar dataset (CSV)",
    type=['csv'],
    help="Sube un archivo CSV con tus datos de clasificación"
)

if uploaded_file is not None:
    # Cargar datos
    df = load_data(uploaded_file)
    
    st.sidebar.success(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # Mostrar vista previa de los datos
    with st.expander("📊 Vista previa del dataset", expanded=True):
        st.dataframe(df.head(10))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Filas", df.shape[0])
        with col2:
            st.metric("Columnas", df.shape[1])
        with col3:
            st.metric("Valores nulos", df.isnull().sum().sum())
    
    # Selección de variables
    st.sidebar.subheader("🎯 Selección de Variables")
    
    columns = df.columns.tolist()
    target_column = st.sidebar.selectbox(
        "Variable objetivo (target)",
        options=columns,
        index=len(columns)-1,
        help="Selecciona la columna que contiene las clases a predecir"
    )
    
    feature_columns = st.sidebar.multiselect(
        "Variables predictoras (features)",
        options=[col for col in columns if col != target_column],
        default=[col for col in columns if col != target_column][:min(4, len(columns)-1)],
        help="Selecciona las columnas que se usarán para entrenar el modelo"
    )
    
    if len(feature_columns) > 0:
        # Selector de modo de entrenamiento
        st.sidebar.markdown("---")
        st.sidebar.subheader("🎯 Modo de Entrenamiento")
        
        training_mode = st.sidebar.radio(
            "Seleccionar modo:",
            options=["Manual", "Búsqueda Automática (Grid Search)"],
            index=0,
            help="Manual: Configuras los parámetros manualmente\nBúsqueda Automática: El sistema encuentra los mejores parámetros"
        )
        
        if training_mode == "Manual":
            # Parámetros del modelo SVM (configuración manual)
            st.sidebar.subheader("🔧 Parámetros del SVM")
            
            kernel = st.sidebar.selectbox(
                "Kernel",
                options=['linear', 'poly', 'rbf', 'sigmoid'],
                index=2,
                help="Función kernel para el SVM"
            )
            
            C = st.sidebar.slider(
                "Parámetro C (regularización)",
                min_value=0.01,
                max_value=100.0,
                value=1.0,
                step=0.01,
                help="Controla el trade-off entre error de entrenamiento y margen"
            )
            
            if kernel in ['rbf', 'poly', 'sigmoid']:
                gamma = st.sidebar.selectbox(
                    "Gamma",
                    options=['scale', 'auto'],
                    index=0,
                    help="Coeficiente del kernel"
                )
            else:
                gamma = 'scale'
            
            if kernel == 'poly':
                degree = st.sidebar.slider(
                    "Grado del polinomio",
                    min_value=2,
                    max_value=5,
                    value=3,
                    help="Grado para el kernel polinomial"
                )
            else:
                degree = 3
        
        else:  # Búsqueda Automática
            st.sidebar.subheader("🔍 Configuración de Grid Search")
            
            search_mode = st.sidebar.selectbox(
                "Modo de búsqueda",
                options=['quick', 'balanced', 'exhaustive'],
                index=1,
                format_func=lambda x: {
                    'quick': '⚡ Rápida (~50 combinaciones)',
                    'balanced': '⚖️ Balanceada (~300 combinaciones)',
                    'exhaustive': '🔬 Exhaustiva (~1000+ combinaciones)'
                }[x],
                help="Rápida: Prueba pocos parámetros\nBalanceada: Equilibrio entre tiempo y cobertura\nExhaustiva: Prueba todas las combinaciones (puede tardar mucho)"
            )
            
            cv_folds_grid = st.sidebar.slider(
                "Folds para Cross-Validation",
                min_value=3,
                max_value=10,
                value=5,
                help="Número de particiones para validación cruzada"
            )
        
        # Parámetros comunes para ambos modos
        test_size = st.sidebar.slider(
            "Tamaño del conjunto de prueba (%)",
            min_value=10,
            max_value=50,
            value=30,
            step=5,
            help="Porcentaje de datos para testing (70% entrenamiento / 30% prueba)"
        ) / 100
        
        random_state = st.sidebar.number_input(
            "Semilla aleatoria",
            min_value=0,
            max_value=999,
            value=42,
            help="Para reproducibilidad de resultados"
        )
        
        # Botón de entrenamiento
        if training_mode == "Manual":
            button_label = "🚀 Entrenar Modelo"
        else:
            button_label = "🔍 Buscar Mejores Parámetros"
        
        if st.sidebar.button(button_label, type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo SVM..." if training_mode == "Manual" else "Buscando mejores parámetros..."):
                try:
                    # Preparar datos
                    X, y, label_encoder = preprocess_data(df, feature_columns, target_column)
                    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
                    
                    if training_mode == "Manual":
                        # Entrenamiento manual con parámetros especificados
                        model, scaler = train_svm(
                            X_train, y_train,
                            kernel=kernel,
                            C=C,
                            gamma=gamma,
                            degree=degree,
                            random_state=random_state
                        )
                        
                        # Guardar en session state (datos SIN escalar para CV correcta)
                        st.session_state['model'] = model
                        st.session_state['scaler'] = scaler
                        st.session_state['X_train'] = X_train  # SIN escalar
                        st.session_state['X_test'] = X_test    # SIN escalar
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test
                        st.session_state['label_encoder'] = label_encoder
                        st.session_state['feature_columns'] = feature_columns
                        st.session_state['target_column'] = target_column
                        st.session_state['training_mode'] = 'manual'
                        # Guardar parámetros del modelo
                        st.session_state['model_params'] = {
                            'kernel': kernel,
                            'C': C,
                            'gamma': gamma,
                            'degree': degree
                        }
                        
                        st.sidebar.success("✅ Modelo entrenado exitosamente!")
                    
                    else:  # Búsqueda Automática (Grid Search)
                        from models.svm_classifier import grid_search_svm, create_param_grid, get_grid_search_results_df
                        
                        # Crear grid de parámetros según modo seleccionado
                        param_grid = create_param_grid(search_mode)
                        
                        # Ejecutar Grid Search
                        grid_results = grid_search_svm(
                            X_train, y_train,
                            param_grid=param_grid,
                            cv=cv_folds_grid,
                            scoring='accuracy',
                            random_state=random_state
                        )
                        
                        # Extraer mejor modelo y scaler
                        model = grid_results['best_model']
                        scaler = grid_results['best_scaler']
                        
                        # Extraer parámetros del mejor modelo
                        best_params = grid_results['best_params']
                        best_kernel = best_params.get('svm__kernel', 'rbf')
                        best_C = best_params.get('svm__C', 1.0)
                        best_gamma = best_params.get('svm__gamma', 'scale')
                        best_degree = best_params.get('svm__degree', 3)
                        
                        # Guardar en session state
                        st.session_state['model'] = model
                        st.session_state['scaler'] = scaler
                        st.session_state['X_train'] = X_train  # SIN escalar
                        st.session_state['X_test'] = X_test    # SIN escalar
                        st.session_state['y_train'] = y_train
                        st.session_state['y_test'] = y_test
                        st.session_state['label_encoder'] = label_encoder
                        st.session_state['feature_columns'] = feature_columns
                        st.session_state['target_column'] = target_column
                        st.session_state['training_mode'] = 'grid_search'
                        st.session_state['grid_results'] = grid_results
                        st.session_state['grid_results_df'] = get_grid_search_results_df(grid_results)
                        # Guardar parámetros del mejor modelo
                        st.session_state['model_params'] = {
                            'kernel': best_kernel,
                            'C': best_C,
                            'gamma': best_gamma if best_kernel != 'linear' else None,
                            'degree': best_degree if best_kernel == 'poly' else None
                        }
                        
                        # Mostrar mejores parámetros encontrados
                        st.sidebar.success(f"✅ Mejor configuración encontrada!")
                        st.sidebar.markdown(f"**Score CV:** {grid_results['best_score']:.4f}")
                        st.sidebar.markdown(f"**Kernel:** {best_kernel}")
                        st.sidebar.markdown(f"**C:** {best_C}")
                        
                        # Solo mostrar gamma si el kernel lo requiere
                        if best_kernel in ['rbf', 'poly', 'sigmoid']:
                            st.sidebar.markdown(f"**Gamma:** {best_gamma}")
                        
                        # Solo mostrar degree si es kernel poly
                        if best_kernel == 'poly':
                            st.sidebar.markdown(f"**Degree:** {best_degree}")
                        
                        st.sidebar.info(f"Se probaron {grid_results['n_combinations']} combinaciones")
                    

                except Exception as e:
                    st.sidebar.error(f"❌ Error al entrenar el modelo: {str(e)}")
        
        # Mostrar resultados si el modelo está entrenado
        if 'model' in st.session_state:
            
            # Si es Grid Search, mostrar vista simplificada
            if st.session_state.get('training_mode') == 'grid_search':
                st.markdown("---")
                st.header("🏆 Configuración Óptima Encontrada")
                st.markdown("*Se probaron múltiples combinaciones de parámetros mediante validación cruzada*")
                
                model_params = st.session_state.get('model_params', {})
                grid_results = st.session_state['grid_results']
                
                # Obtener parámetros
                best_kernel = model_params.get('kernel', 'N/A')
                best_C = model_params.get('C', 'N/A')
                best_gamma_raw = model_params.get('gamma', None)
                best_degree = model_params.get('degree', None)
                best_score = grid_results['best_score']
                n_combinations = grid_results['n_combinations']
                
                # Convertir gamma numérico a formato compatible con modo manual
                # Si gamma es un número, mostrar advertencia de que no se puede usar directamente
                if best_gamma_raw is not None:
                    try:
                        gamma_float = float(best_gamma_raw)
                        # Es un valor numérico, no 'scale' o 'auto'
                        best_gamma = f"{gamma_float} (usar 'scale' en manual)"
                        gamma_is_numeric = True
                    except (ValueError, TypeError):
                        # Es 'scale' o 'auto'
                        best_gamma = best_gamma_raw
                        gamma_is_numeric = False
                else:
                    best_gamma = None
                    gamma_is_numeric = False
                
                # Crear contenido según el kernel
                st.markdown("### 📋 Parámetros para ingresar en Modo Manual:")
                
                # Panel principal con la configuración
                st.success(f"✅ **Se probaron {n_combinations} combinaciones** y la mejor obtuvo **{best_score*100:.2f}% de precisión** en validación cruzada.")
                
                # Mostrar configuración de forma clara
                config_col1, config_col2 = st.columns(2)
                
                with config_col1:
                    st.markdown("#### 🔧 Configuración del Kernel")
                    st.markdown(f"**1. Kernel:** `{best_kernel}`")
                    st.markdown(f"**2. Parámetro C:** `{best_C}`")
                
                with config_col2:
                    st.markdown("#### ⚙️ Parámetros específicos")
                    
                    if best_kernel == 'linear':
                        st.markdown("*El kernel linear no requiere parámetros adicionales*")
                        st.markdown(f"- Gamma: No aplica")
                        st.markdown(f"- Degree: No aplica")
                    
                    elif best_kernel == 'rbf':
                        if best_gamma is not None:
                            st.markdown(f"**3. Gamma:** `{best_gamma}`")
                            if gamma_is_numeric:
                                st.warning("⚠️ El Grid Search encontró un valor numérico para gamma. En modo manual, usa 'scale' o 'auto'.")
                        st.markdown(f"- Degree: No aplica (solo para poly)")
                    
                    elif best_kernel == 'poly':
                        if best_gamma is not None:
                            st.markdown(f"**3. Gamma:** `{best_gamma}`")
                            if gamma_is_numeric:
                                st.warning("⚠️ El Grid Search encontró un valor numérico para gamma. En modo manual, usa 'scale' o 'auto'.")
                        if best_degree is not None:
                            st.markdown(f"**4. Degree:** `{best_degree}`")
                    
                    elif best_kernel == 'sigmoid':
                        if best_gamma is not None:
                            st.markdown(f"**3. Gamma:** `{best_gamma}`")
                            if gamma_is_numeric:
                                st.warning("⚠️ El Grid Search encontró un valor numérico para gamma. En modo manual, usa 'scale' o 'auto'.")
                        st.markdown(f"- Degree: No aplica (solo para poly)")
                
                # Resumen visual
                st.markdown("---")
                st.markdown("### 📝 Resumen para copiar:")
                
                # Crear texto de resumen según kernel
                if best_kernel == 'linear':
                    resumen = f"""
| Parámetro | Valor |
|-----------|-------|
| **Kernel** | {best_kernel} |
| **C** | {best_C} |
"""
                elif best_kernel == 'rbf' or best_kernel == 'sigmoid':
                    # Mostrar gamma solo si no es None
                    gamma_display = best_gamma if best_gamma is not None else "N/A"
                    resumen = f"""
| Parámetro | Valor |
|-----------|-------|
| **Kernel** | {best_kernel} |
| **C** | {best_C} |
| **Gamma** | {gamma_display} |
"""
                    if gamma_is_numeric:
                        resumen += "\n⚠️ **Nota:** El gamma es numérico. En modo manual, selecciona 'scale' o 'auto'.\n"
                
                elif best_kernel == 'poly':
                    gamma_display = best_gamma if best_gamma is not None else "N/A"
                    degree_display = best_degree if best_degree is not None else "N/A"
                    resumen = f"""
| Parámetro | Valor |
|-----------|-------|
| **Kernel** | {best_kernel} |
| **C** | {best_C} |
| **Gamma** | {gamma_display} |
| **Degree** | {degree_display} |
"""
                    if gamma_is_numeric:
                        resumen += "\n⚠️ **Nota:** El gamma es numérico. En modo manual, selecciona 'scale' o 'auto'.\n"
                else:
                    resumen = f"""
| Parámetro | Valor |
|-----------|-------|
| **Kernel** | {best_kernel} |
| **C** | {best_C} |
| **Gamma** | {best_gamma} |
| **Degree** | {best_degree} |
"""
                
                st.markdown(resumen)
                
                st.info(f"💡 **Siguiente paso:** Cambia a modo **Manual** en el panel lateral, ingresa estos parámetros y entrena el modelo para verificar los resultados.")
            
            # Resultados del modelo entrenado (común para ambos modos)
            st.markdown("---")
            st.header("📈 Resultados del Modelo Entrenado")
            
            model = st.session_state['model']
            scaler = st.session_state['scaler']
            X_train = st.session_state['X_train']
            X_test = st.session_state['X_test']
            y_train = st.session_state['y_train']
            y_test = st.session_state['y_test']
            label_encoder = st.session_state['label_encoder']
            
            # Predicciones
            y_train_pred = predict_svm(model, X_train, scaler)
            y_test_pred = predict_svm(model, X_test, scaler)
            
            # Métricas
            train_metrics = get_model_metrics(y_train, y_train_pred)
            test_metrics = get_model_metrics(y_test, y_test_pred)
            
            # Mostrar métricas
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Métricas de Entrenamiento")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Accuracy", f"{train_metrics['accuracy']:.3f}")
                with metric_col2:
                    st.metric("Precision", f"{train_metrics['precision']:.3f}")
                with metric_col3:
                    st.metric("Recall", f"{train_metrics['recall']:.3f}")
                
                st.metric("F1-Score", f"{train_metrics['f1']:.3f}")
            
            with col2:
                st.subheader("📊 Métricas de Prueba")
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("Accuracy", f"{test_metrics['accuracy']:.3f}")
                with metric_col2:
                    st.metric("Precision", f"{test_metrics['precision']:.3f}")
                with metric_col3:
                    st.metric("Recall", f"{test_metrics['recall']:.3f}")
                
                st.metric("F1-Score", f"{test_metrics['f1']:.3f}")
            
            st.markdown("---")
            
            # Visualizaciones
            st.header("📉 Visualizaciones")
            
            tab1, tab2, tab3 = st.tabs(["Matriz de Confusión", "Frontera de Decisión", "Curva ROC"])
            
            with tab1:
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Entrenamiento")
                    fig_cm_train = plot_confusion_matrix(
                        y_train, y_train_pred,
                        label_encoder.classes_,
                        "Matriz de Confusión - Entrenamiento"
                    )
                    st.pyplot(fig_cm_train)
                
                with col2:
                    st.subheader("Prueba")
                    fig_cm_test = plot_confusion_matrix(
                        y_test, y_test_pred,
                        label_encoder.classes_,
                        "Matriz de Confusión - Prueba"
                    )
                    st.pyplot(fig_cm_test)
            
            with tab2:
                if len(feature_columns) >= 2:
                    feature_x = st.selectbox("Feature X", feature_columns, index=0)
                    feature_y = st.selectbox("Feature Y", feature_columns, index=min(1, len(feature_columns)-1))
                    
                    idx_x = feature_columns.index(feature_x)
                    idx_y = feature_columns.index(feature_y)
                    
                    fig_boundary = plot_decision_boundary(
                        model, scaler, X_test, y_test,
                        idx_x, idx_y,
                        label_encoder.classes_,
                        feature_x, feature_y
                    )
                    st.pyplot(fig_boundary)
                else:
                    st.info("Se necesitan al menos 2 features para visualizar la frontera de decisión")
            
            
            with tab3:
                n_classes = len(np.unique(y_test))
                if n_classes == 2:
                    st.subheader("Clasificación Binaria")
                    fig_roc = plot_roc_with_auc(model, X_test, y_test, scaler)
                    st.pyplot(fig_roc)
                else:
                    st.subheader("Clasificación Multiclase")
                    fig_roc = plot_roc_with_auc(model, X_test, y_test, scaler, label_encoder.classes_)
                    st.pyplot(fig_roc)
            
            # NUEVA SECCIÓN: Validación Cruzada
            st.markdown("---")
            st.header("🔄 Validación Cruzada")
            
            with st.expander("ℹ️ Sobre la Validación Cruzada", expanded=False):
                st.markdown("""
                **¿Qué es la Validación Cruzada?**
                
                La validación cruzada es una técnica robusta para evaluar el rendimiento del modelo:
                
                - **Concepto**: Divide el dataset en K folds (particiones) y entrena K veces
                - **StratifiedKFold**: Mantiene la proporción de clases en cada fold
                - **Ventajas**:
                  - Uso eficiente de todos los datos
                  - Reduce el sesgo de una única partición
                  - Proporciona estimaciones más confiables del rendimiento
                  - Detecta overfitting/underfitting
                
                **Métricas promediadas**: Obtenemos la media y desviación estándar de cada métrica
                """)
            
            cv_folds = st.slider(
                "Número de folds para validación cruzada",
                min_value=2,
                max_value=10,
                value=5,
                help="Mayor número de folds = más tiempo de cómputo pero mejor estimación"
            )
            
            if st.button("🔄 Ejecutar Validación Cruzada", type="secondary", use_container_width=True):
                with st.spinner(f"Ejecutando validación cruzada con {cv_folds} folds..."):
                    try:
                        # Obtener parámetros del modelo guardados
                        model_params = st.session_state.get('model_params', {})
                        kernel = model_params.get('kernel', 'rbf')
                        C = model_params.get('C', 1.0)
                        gamma = model_params.get('gamma', 'scale')
                        degree = model_params.get('degree', 3)
                        
                        # Obtener todos los datos (sin split)
                        X_full = st.session_state['X_train']
                        y_full = st.session_state['y_train']
                        
                        # Combinar train y test para CV completa
                        X_full = np.vstack([st.session_state['X_train'], st.session_state['X_test']])
                        y_full = np.concatenate([st.session_state['y_train'], st.session_state['y_test']])
                        
                        # Ejecutar validación cruzada
                        cv_results = cross_validate_svm(
                            X_full, y_full,
                            kernel=kernel,
                            C=C,
                            gamma=gamma,
                            degree=degree,
                            cv=cv_folds,
                            random_state=random_state
                        )
                        
                        # Ejecutar StratifiedKFold detallado
                        fold_results = perform_stratified_kfold(
                            X_full, y_full,
                            kernel=kernel,
                            C=C,
                            gamma=gamma,
                            degree=degree,
                            n_splits=cv_folds,
                            random_state=random_state
                        )
                        
                        st.session_state['cv_results'] = cv_results
                        st.session_state['fold_results'] = fold_results
                        
                        st.success("✅ Validación cruzada completada!")
                        
                    except Exception as e:
                        st.error(f"❌ Error en validación cruzada: {str(e)}")
            
            # Mostrar resultados de CV si existen
            if 'cv_results' in st.session_state:
                cv_results = st.session_state['cv_results']
                fold_results = st.session_state['fold_results']
                
                st.subheader("📊 Resultados de Validación Cruzada")
                
                # Métricas promedio
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric(
                        "Accuracy", 
                        f"{cv_results['accuracy']['mean']:.3f}",
                        delta=f"±{cv_results['accuracy']['std']:.3f}"
                    )
                with col2:
                    st.metric(
                        "Precision", 
                        f"{cv_results['precision']['mean']:.3f}",
                        delta=f"±{cv_results['precision']['std']:.3f}"
                    )
                with col3:
                    st.metric(
                        "Recall", 
                        f"{cv_results['recall']['mean']:.3f}",
                        delta=f"±{cv_results['recall']['std']:.3f}"
                    )
                with col4:
                    st.metric(
                        "F1-Score", 
                        f"{cv_results['f1']['mean']:.3f}",
                        delta=f"±{cv_results['f1']['std']:.3f}"
                    )
                
                # Visualizaciones de CV
                st.markdown("### 📈 Visualizaciones de Validación Cruzada")
                
                tab_cv1, tab_cv2, tab_cv3 = st.tabs([
                    "Promedios con Desviación",
                    "Comparación por Fold", 
                    "Distribución de Scores"
                ])
                
                with tab_cv1:
                    fig_cv = plot_cv_results(cv_results)
                    st.pyplot(fig_cv)
                    st.caption("Métricas promedio con barras de error (desviación estándar)")
                
                with tab_cv2:
                    fig_folds = plot_cv_folds_comparison(fold_results)
                    st.pyplot(fig_folds)
                    st.caption("Comparación de métricas en cada fold individual")
                
                with tab_cv3:
                    fig_dist = plot_cv_scores_distribution(cv_results)
                    st.pyplot(fig_dist)
                    st.caption("Distribución de scores usando boxplots (mediana=línea roja, media=diamante verde)")
                
                # Tabla detallada por fold
                with st.expander("📋 Resultados Detallados por Fold"):
                    fold_df = pd.DataFrame([
                        {
                            'Fold': f['fold'],
                            'Tamaño Train': f['train_size'],
                            'Tamaño Val': f['val_size'],
                            'Accuracy': f'{f["metrics"]["accuracy"]:.4f}',
                            'Precision': f'{f["metrics"]["precision"]:.4f}',
                            'Recall': f'{f["metrics"]["recall"]:.4f}',
                            'F1-Score': f'{f["metrics"]["f1"]:.4f}'
                        }
                        for f in fold_results
                    ])
                    st.dataframe(fold_df, use_container_width=True)
            
            # Información del modelo
            st.markdown("---")
            with st.expander("ℹ️ Información del Modelo"):
                model_params = st.session_state.get('model_params', {})
                kernel_info = model_params.get('kernel', 'N/A')
                C_info = model_params.get('C', 'N/A')
                gamma_info = model_params.get('gamma', 'N/A')
                degree_info = model_params.get('degree', 'N/A')
                
                st.write(f"**Kernel:** {kernel_info}")
                st.write(f"**C:** {C_info}")
                st.write(f"**Gamma:** {gamma_info}")
                if kernel_info == 'poly':
                    st.write(f"**Grado:** {degree_info}")
                st.write(f"**Número de vectores de soporte:** {model.n_support_.sum()}")
                st.write(f"**Clases:** {list(label_encoder.classes_)}")
                
    else:
        st.warning("⚠️ Por favor selecciona al menos una variable predictora")
        
else:
    st.info("👈 Comienza subiendo un archivo CSV desde el panel lateral")
    
    st.markdown("""
    ### 📚 Sobre las Máquinas de Vectores de Soporte (SVM)
    
    Las **SVM** son algoritmos de aprendizaje supervisado utilizados principalmente para **clasificación**.
    
    #### 🎯 Características principales:
    - Encuentran el hiperplano óptimo que maximiza el margen entre clases
    - Funcionan bien en espacios de alta dimensionalidad
    - Efectivos cuando el número de dimensiones es mayor que el número de muestras
    - Utilizan diferentes funciones kernel para manejar datos no linealmente separables
    
    #### 🔧 Kernels disponibles:
    - **Linear:** Para datos linealmente separables
    - **RBF (Radial Basis Function):** El más popular, funciona bien en la mayoría de casos
    - **Polynomial:** Para relaciones polinomiales entre features
    - **Sigmoid:** Similar a redes neuronales
    
    #### 📊 Formato del dataset:
    - Archivo CSV con encabezados
    - Última columna (o la que elijas) como variable objetivo
    - Valores numéricos o categóricos (se convertirán automáticamente)
    """)
