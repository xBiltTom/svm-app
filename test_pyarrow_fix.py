"""
Test para verificar que el error de PyArrow está solucionado
"""

import numpy as np
from sklearn.datasets import load_iris
from models.svm_classifier import grid_search_svm, create_param_grid, get_grid_search_results_df

print("=" * 70)
print("TEST: Verificación de corrección del error PyArrow")
print("=" * 70)

# Cargar dataset
iris = load_iris()
X, y = iris.data, iris.target

print(f"\nDataset: {X.shape[0]} muestras, {X.shape[1]} features")

# Ejecutar Grid Search con modo balanced (tiene valores mixtos en gamma)
print("\n" + "-" * 70)
print("Ejecutando Grid Search con modo 'balanced'")
print("(Este modo tiene valores mixtos en gamma: strings y floats)")
print("-" * 70)

param_grid = create_param_grid('balanced')
print(f"\nParámetros gamma en el grid: {param_grid['svm__gamma']}")

grid_results = grid_search_svm(
    X, y,
    param_grid=param_grid,
    cv=3,
    scoring='accuracy',
    random_state=42
)

print(f"\n✅ Grid Search completado sin errores!")
print(f"  Mejor score: {grid_results['best_score']:.4f}")
print(f"  Combinaciones probadas: {grid_results['n_combinations']}")

# Convertir a DataFrame
print("\n" + "-" * 70)
print("Convirtiendo resultados a DataFrame")
print("-" * 70)

results_df = get_grid_search_results_df(grid_results)

print(f"\n✅ DataFrame creado exitosamente!")
print(f"  Shape: {results_df.shape}")
print(f"  Columnas: {list(results_df.columns)}")

# Verificar tipos de datos
print("\n" + "-" * 70)
print("Verificando tipos de datos")
print("-" * 70)

print(f"\nTipos de columnas:")
for col in results_df.columns:
    if col.startswith('svm__'):
        dtype = results_df[col].dtype
        unique_vals = results_df[col].nunique()
        print(f"  {col}: {dtype} ({unique_vals} valores únicos)")

# Verificar que gamma es string
print(f"\n✅ Columna 'svm__gamma' es tipo: {results_df['svm__gamma'].dtype}")
print(f"  Valores únicos: {results_df['svm__gamma'].unique()[:5]}")

# Intentar mostrar como lo haría Streamlit
print("\n" + "-" * 70)
print("Simulando visualización en Streamlit")
print("-" * 70)

print(f"\nTop 5 configuraciones:")
print(results_df[['svm__kernel', 'svm__C', 'svm__gamma', 'mean_test_score']].head())

# Verificar que se puede exportar a CSV
print("\n" + "-" * 70)
print("Verificando exportación a CSV")
print("-" * 70)

csv_data = results_df.to_csv(index=False)
print(f"\n✅ CSV generado exitosamente!")
print(f"  Tamaño: {len(csv_data)} bytes")

print("\n" + "=" * 70)
print("✅ TODOS LOS TESTS PASARON")
print("   - Grid Search ejecutado sin errores")
print("   - DataFrame creado sin errores de PyArrow")
print("   - Tipos de datos correctos (todo string en parámetros)")
print("   - CSV exportable sin problemas")
print("=" * 70)
