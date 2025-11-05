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
        # Parámetros del modelo SVM
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
        if st.sidebar.button("🚀 Entrenar Modelo", type="primary", use_container_width=True):
            with st.spinner("Entrenando modelo SVM..."):
                try:
                    # Preparar datos
                    X, y, label_encoder = preprocess_data(df, feature_columns, target_column)
                    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
                    
                    # Entrenar modelo
                    model, scaler = train_svm(
                        X_train, y_train,
                        kernel=kernel,
                        C=C,
                        gamma=gamma,
                        degree=degree,
                        random_state=random_state
                    )
                    
                    # Guardar en session state
                    st.session_state['model'] = model
                    st.session_state['scaler'] = scaler
                    st.session_state['X_train'] = X_train
                    st.session_state['X_test'] = X_test
                    st.session_state['y_train'] = y_train
                    st.session_state['y_test'] = y_test
                    st.session_state['label_encoder'] = label_encoder
                    st.session_state['feature_columns'] = feature_columns
                    st.session_state['target_column'] = target_column
                    
                    st.sidebar.success("✅ Modelo entrenado exitosamente!")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ Error al entrenar el modelo: {str(e)}")
        
        # Mostrar resultados si el modelo está entrenado
        if 'model' in st.session_state:
            st.markdown("---")
            st.header("📈 Resultados del Modelo")
            
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
                st.write(f"**Kernel:** {kernel}")
                st.write(f"**C:** {C}")
                st.write(f"**Gamma:** {gamma}")
                if kernel == 'poly':
                    st.write(f"**Grado:** {degree}")
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
