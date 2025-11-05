import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import streamlit as st

# ============================================================================
# PASO 1: CARGA DEL DATASET
# ============================================================================

def load_data(uploaded_file):
    """
    PASO 1: Carga un archivo CSV y retorna un DataFrame
    """
    try:
        df = pd.read_csv(uploaded_file)
        return df
    except Exception as e:
        st.error(f"Error al cargar el archivo: {str(e)}")
        return None

# ============================================================================
# PASO 2: EXPLORACIÓN INICIAL
# ============================================================================

def explore_data(df):
    """
    PASO 2: Exploración inicial del dataset
    Retorna información sobre estructura, tipos, estadísticas y valores nulos
    """
    exploration_report = {
        # Información básica
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        
        # Tipos de datos
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist(),
        
        # Valores nulos
        'null_counts': df.isnull().sum().to_dict(),
        'null_percentages': (df.isnull().sum() / len(df) * 100).to_dict(),
        'total_nulls': df.isnull().sum().sum(),
        
        # Estadísticas descriptivas
        'describe_numeric': df.describe().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {},
        
        # Duplicados
        'duplicated_rows': df.duplicated().sum(),
        
        # Memoria
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
    }
    
    return exploration_report

def print_data_info(df):
    """
    PASO 2: Imprime información detallada del dataset (para debugging)
    """
    print("="*80)
    print("EXPLORACIÓN INICIAL DEL DATASET")
    print("="*80)
    
    print(f"\n1. ESTRUCTURA:")
    print(f"   - Filas: {df.shape[0]}")
    print(f"   - Columnas: {df.shape[1]}")
    print(f"   - Memoria: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\n2. TIPOS DE DATOS:")
    print(df.dtypes)
    
    print(f"\n3. VALORES NULOS:")
    null_counts = df.isnull().sum()
    if null_counts.sum() > 0:
        print(null_counts[null_counts > 0])
    else:
        print("   No hay valores nulos")
    
    print(f"\n4. DUPLICADOS:")
    print(f"   - Filas duplicadas: {df.duplicated().sum()}")
    
    print(f"\n5. ESTADÍSTICAS DESCRIPTIVAS:")
    print(df.describe())
    
    print("="*80 + "\n")

# ============================================================================
# PASO 3: LIMPIEZA DE DATOS
# ============================================================================

def detect_outliers_iqr(df, columns=None):
    """
    PASO 3: Detecta outliers usando el método IQR (Rango Intercuartílico)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    outliers_info = {}
    
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        
        outliers_info[col] = {
            'count': len(outliers),
            'percentage': len(outliers) / len(df) * 100,
            'lower_bound': lower_bound,
            'upper_bound': upper_bound
        }
    
    return outliers_info

def remove_outliers_iqr(df, columns=None, method='remove'):
    """
    PASO 3: Elimina o trata outliers usando IQR
    method: 'remove' (eliminar filas) o 'cap' (limitar valores)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    df_clean = df.copy()
    
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        if method == 'remove':
            # Eliminar filas con outliers
            df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
        elif method == 'cap':
            # Limitar valores a los límites
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def clean_data(df, remove_duplicates=True, handle_nulls=True, null_strategy='mean', 
               handle_outliers=False, outlier_method='remove'):
    """
    PASO 3: Limpieza completa de datos
    - Elimina duplicados
    - Maneja valores nulos
    - Opcionalmente maneja outliers
    """
    df_clean = df.copy()
    
    print(f"Filas originales: {len(df_clean)}")
    
    # 3.1: Eliminar duplicados
    if remove_duplicates:
        duplicados_antes = df_clean.duplicated().sum()
        df_clean = df_clean.drop_duplicates()
        print(f"Duplicados eliminados: {duplicados_antes}")
    
    # 3.2: Manejar valores nulos
    if handle_nulls:
        nulls_antes = df_clean.isnull().sum().sum()
        df_clean = handle_missing_values(df_clean, strategy=null_strategy)
        print(f"Valores nulos tratados: {nulls_antes}")
    
    # 3.3: Manejar outliers (opcional)
    if handle_outliers:
        outliers_info = detect_outliers_iqr(df_clean)
        total_outliers = sum([info['count'] for info in outliers_info.values()])
        df_clean = remove_outliers_iqr(df_clean, method=outlier_method)
        print(f"Outliers tratados: {total_outliers}")
    
    print(f"Filas después de limpieza: {len(df_clean)}")
    
    return df_clean


# ============================================================================
# PASO 4: CODIFICACIÓN DE VARIABLES CATEGÓRICAS
# ============================================================================

def encode_categorical_variables(df, columns=None, method='label'):
    """
    PASO 4: Codifica variables categóricas
    method: 'label' (LabelEncoder) u 'onehot' (One-Hot Encoding)
    """
    df_encoded = df.copy()
    encoders = {}
    
    if columns is None:
        columns = df_encoded.select_dtypes(include=['object']).columns
    
    print(f"\nCodificando {len(columns)} variables categóricas:")
    
    for col in columns:
        print(f"  - {col}: {df_encoded[col].nunique()} valores únicos")
        
        if method == 'label':
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
        elif method == 'onehot':
            # One-Hot Encoding (crea columnas dummy)
            dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
            df_encoded = pd.concat([df_encoded, dummies], axis=1)
            df_encoded = df_encoded.drop(col, axis=1)
            encoders[col] = 'onehot'
    
    return df_encoded, encoders

def preprocess_data(df, feature_columns, target_column):
    """
    PASO 4: Preprocesa los datos siguiendo el pipeline completo
    - Codifica variables categóricas
    - Prepara X e y
    """
    print("\n" + "="*80)
    print("PASO 4: CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
    print("="*80)
    
    # Crear copia del dataframe
    df_processed = df.copy()
    
    # Separar features y target
    X = df_processed[feature_columns].copy()
    y = df_processed[target_column].copy()
    
    # Codificar variables categóricas en X
    label_encoders_X = {}
    for col in X.columns:
        if X[col].dtype == 'object':
            print(f"Codificando feature categórica: {col}")
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders_X[col] = le
    
    # Convertir a numérico si hay valores no numéricos
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Rellenar valores NaN con la media (si quedó alguno)
    if X.isnull().sum().sum() > 0:
        print(f"Rellenando {X.isnull().sum().sum()} valores NaN con la media")
        X = X.fillna(X.mean())
    
    # Codificar variable objetivo si es categórica
    label_encoder_y = LabelEncoder()
    if y.dtype == 'object':
        print(f"Codificando target categórico: {target_column}")
        y = label_encoder_y.fit_transform(y.astype(str))
        print(f"Clases encontradas: {list(label_encoder_y.classes_)}")
    else:
        # Asegurar que y es numérico
        y = pd.to_numeric(y, errors='coerce')
        # Si hay valores únicos categóricos, codificar
        y = label_encoder_y.fit_transform(y)
    
    print(f"\nDatos procesados:")
    print(f"  - Features (X): {X.shape}")
    print(f"  - Target (y): {y.shape}")
    print(f"  - Clases: {len(np.unique(y))}")
    
    return X.values, y, label_encoder_y

# ============================================================================
# PASO 5: NORMALIZACIÓN O ESTANDARIZACIÓN
# ============================================================================

def normalize_or_standardize(X, method='standardize'):
    """
    PASO 5: Normaliza o estandariza las características
    method: 'standardize' (StandardScaler) o 'normalize' (MinMaxScaler)
    """
    print(f"\n" + "="*80)
    print(f"PASO 5: {'ESTANDARIZACIÓN' if method == 'standardize' else 'NORMALIZACIÓN'}")
    print("="*80)
    
    if method == 'standardize':
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print("Aplicando StandardScaler (media=0, std=1)")
    elif method == 'normalize':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        print("Aplicando MinMaxScaler (rango 0-1)")
    
    X_scaled = scaler.fit_transform(X)
    
    print(f"Características escaladas: {X_scaled.shape}")
    print(f"  - Media: {X_scaled.mean(axis=0).mean():.4f}")
    print(f"  - Desviación estándar: {X_scaled.std(axis=0).mean():.4f}")
    
    return X_scaled, scaler

# ============================================================================
# PASO 6: DIVISIÓN EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
# ============================================================================

def split_data(X, y, test_size=0.3, random_state=42):
    """
    PASO 6: Divide los datos en conjuntos de entrenamiento (70%) y prueba (30%)
    """
    print(f"\n" + "="*80)
    print("PASO 6: DIVISIÓN EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA")
    print("="*80)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y if len(np.unique(y)) > 1 else None
    )
    
    print(f"División completada (test_size={test_size*100:.0f}%):")
    print(f"  - Entrenamiento: {len(X_train)} muestras ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
    print(f"  - Prueba: {len(X_test)} muestras ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")
    
    # Verificar distribución de clases
    unique, counts_train = np.unique(y_train, return_counts=True)
    unique, counts_test = np.unique(y_test, return_counts=True)
    
    print(f"\nDistribución de clases:")
    print(f"  - Entrenamiento: {dict(zip(unique, counts_train))}")
    print(f"  - Prueba: {dict(zip(unique, counts_test))}")
    
    return X_train, X_test, y_train, y_test

# ============================================================================
# PIPELINE COMPLETO: TODOS LOS PASOS INTEGRADOS
# ============================================================================

def prepare_data_pipeline(df, feature_columns, target_column, 
                         remove_duplicates=True, 
                         handle_nulls=True, 
                         null_strategy='mean',
                         handle_outliers=False,
                         outlier_method='remove',
                         scaling_method='standardize',
                         test_size=0.3,
                         random_state=42,
                         verbose=True):
    """
    PIPELINE COMPLETO DE PREPARACIÓN DE DATOS
    
    Ejecuta todos los pasos en orden:
    1. Exploración inicial
    2. Limpieza de datos
    3. Codificación de variables categóricas
    4. Normalización/Estandarización
    5. División en train/test
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, label_encoder
    """
    
    if verbose:
        print("\n" + "="*80)
        print("INICIANDO PIPELINE DE PREPARACIÓN DE DATOS")
        print("="*80)
    
    # PASO 1: Ya cargado (df)
    if verbose:
        print(f"\nPASO 1: DATASET CARGADO")
        print(f"  - Shape: {df.shape}")
    
    # PASO 2: Exploración inicial
    if verbose:
        print(f"\nPASO 2: EXPLORACIÓN INICIAL")
        exploration = explore_data(df)
        print(f"  - Filas: {exploration['shape'][0]}")
        print(f"  - Columnas: {exploration['shape'][1]}")
        print(f"  - Valores nulos: {exploration['total_nulls']}")
        print(f"  - Duplicados: {exploration['duplicated_rows']}")
    
    # PASO 3: Limpieza de datos
    if verbose:
        print(f"\nPASO 3: LIMPIEZA DE DATOS")
    
    df_clean = clean_data(
        df, 
        remove_duplicates=remove_duplicates,
        handle_nulls=handle_nulls,
        null_strategy=null_strategy,
        handle_outliers=handle_outliers,
        outlier_method=outlier_method
    )
    
    # PASO 4: Codificación y preparación
    X, y, label_encoder = preprocess_data(df_clean, feature_columns, target_column)
    
    # PASO 5: Estandarización/Normalización
    # Nota: Se hace DESPUÉS del split para evitar data leakage
    # Por ahora preparamos para el split
    
    # PASO 6: División en train/test
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=test_size, random_state=random_state)
    
    # PASO 5 (aplicado correctamente): Escalar SOLO con datos de entrenamiento
    if verbose:
        print(f"\n" + "="*80)
        print(f"PASO 5: ESTANDARIZACIÓN (después del split)")
        print("="*80)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if verbose:
        print(f"Escalado completado correctamente (sin data leakage)")
        print(f"  - X_train escalado: {X_train_scaled.shape}")
        print(f"  - X_test escalado: {X_test_scaled.shape}")
        print("\n" + "="*80)
        print("PIPELINE COMPLETADO EXITOSAMENTE")
        print("="*80 + "\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, label_encoder

def get_data_info(df):
    """
    Obtiene información básica del dataset
    """
    info = {
        'shape': df.shape,
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.to_dict(),
        'null_counts': df.isnull().sum().to_dict(),
        'numeric_columns': df.select_dtypes(include=[np.number]).columns.tolist(),
        'categorical_columns': df.select_dtypes(include=['object']).columns.tolist()
    }
    return info

def handle_missing_values(df, strategy='mean'):
    """
    Maneja valores faltantes en el dataset
    """
    df_clean = df.copy()
    
    if strategy == 'mean':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].mean())
    elif strategy == 'median':
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].fillna(df_clean[numeric_cols].median())
    elif strategy == 'mode':
        for col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0] if not df_clean[col].mode().empty else 0)
    elif strategy == 'drop':
        df_clean = df_clean.dropna()
    
    return df_clean
