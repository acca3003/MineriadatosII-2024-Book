import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import chi2_contingency
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


def create_expert_bucket_numeric_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Transforma una columna numérica en el DataFrame en categorías discretas (buckets)
    predefinidas según reglas expertas para las características 'StudyTimeWeekly' y 'Absences'.
    Elimina la columna original y agrega una nueva con el sufijo '_bucket'.

    :param df: DataFrame que contiene las características a transformar.
    :param feature_name: Nombre de la columna numérica a transformar. Debe ser 'Absences' o 'StudyTimeWeekly'.
    :return: DataFrame con la columna transformada y la columna original eliminada.

    :raises ValueError: Si `feature_name` no es 'Absences' ni 'StudyTimeWeekly'.
    """
    features_name_list = ['Absences', 'StudyTimeWeekly']
    if feature_name not in features_name_list:
        raise ValueError(f"feature {feature_name} no disponible. Las features posibles son: {features_name_list}")

    conditions = []
    choices = []

    if feature_name == 'StudyTimeWeekly':
        conditions = [
            (df[feature_name] < 3.0),
            (df[feature_name] >= 3.0) & (df[feature_name] < 6.0),
            (df[feature_name] >= 6.0) & (df[feature_name] < 9.0),
            (df[feature_name] >= 9.0) & (df[feature_name] < 12.0),
            (df[feature_name] >= 12.0) & (df[feature_name] < 15.0),
            (df[feature_name] >= 15.0)
        ]
        choices = ["<3", "3-6", "6-9", "9-12", "12-15", ">=15"]
    elif feature_name == 'Absences':
        conditions = [
            (df[feature_name] == 0),
            (df[feature_name] < 5),
            (df[feature_name] >= 5) & (df[feature_name] < 10),
            (df[feature_name] >= 10) & (df[feature_name] < 20),
            (df[feature_name] >= 20.0)
        ]
        choices = [
            "no-absence", "less_5_absences", "bewteen_5_10_absences", "bewteen_10_20_absences", "more_20_absences"
        ]
    df[f'{feature_name}_bucket'] = np.select(conditions, choices)
    df = df.drop(columns=feature_name)
    return df


def get_category_feature(df: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    """
    Convierte las columnas especificadas en el DataFrame a tipo 'category'.

    :param df: DataFrame que contiene las columnas a convertir.
    :param feature_name: Lista de nombres de las columnas a convertir a tipo 'category'.
    :return: DataFrame con las columnas convertidas a tipo 'category'.
    """
    df[feature_name] = df[feature_name].astype('category')
    return df


def plot_categorical_distribution(df: pd.DataFrame, feature: str, target: str, ax: plt.Axes) -> None:
    """
    Genera un gráfico de barras para mostrar la distribución de una variable categórica
    en función de un target binario.

    :param df: DataFrame con los datos.
    :param feature: Nombre de la variable categórica o dummy a graficar.
    :param target: Nombre de la variable target binaria.
    :param ax: Eje donde se pintará el gráfico.
    :return: None. La función muestra el gráfico.
    """
    sns.countplot(data=df, x=feature, hue=target, palette='Set1', ax=ax)
    ax.set_title(f'Distribución de {feature} por {target}')
    ax.set_xlabel(feature)
    ax.set_ylabel('Frecuencia')
    ax.legend(title=target, loc='upper right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)


def cramers_v(df: pd.DataFrame, var1: str, var2: str) -> float:
    """
    Calcula la V de Cramer entre dos variables categóricas en un DataFrame.

    :param df: DataFrame que contiene las variables categóricas.
    :param var1: Nombre de la primera variable categórica.
    :param var2: Nombre de la segunda variable categórica.
    :return: Valor de la V de Cramer entre las dos variables.
    """

    # tabla de contingencia
    contingency_table = pd.crosstab(df[var1], df[var2])

    # prueba Chi-cuadrado
    chi2_stat, p, dof, expected = chi2_contingency(contingency_table, correction=False)

    # número de observaciones
    n = contingency_table.sum().sum()

    # número de categorías
    k1 = contingency_table.shape[0]
    k2 = contingency_table.shape[1]

    # cálculo de la V de Cramer
    phi2 = chi2_stat / n
    min_dim = min(k1 - 1, k2 - 1)
    v_cramer = np.sqrt(phi2 / min_dim)

    return v_cramer


def apply_threshold(probabilidad: float, threshold: int) -> int:
    """
    Toma de decisión de la etiqueta a partir de un threshold dado

    :param probabilidad: Valor de probabilidad dado por el modelo (entre 0 y 1)
    :param threshold: Umbral de decisión propuesto por el usuario
    :return: Valor que representa la decisión del usuario a partir del umbral
    1 si la probabilidad alcanza el valor del umbral, 0 en otro caso
    """
    return 1 if probabilidad >= threshold else 0


def precision_recall_curve_plot(threshold: list, precision: list, recall: list) -> None:
    """
    Grafica las curvas de precisión y recall en función de diferentes umbrales de clasificación

    :param threshold: Lista de umbrales utilizados para calcular precisión y recall
    :param precision: Lista de valores de precisión correspondientes a cada umbral
    :param recall: Lista de valores de recall correspondientes a cada umbral
    :return: None. La función solo muestra el gráfico
    """
    plt.figure(figsize=(10, 6))

    plt.plot(threshold, recall, marker='o', label='Recall', color='b')
    plt.plot(threshold, precision, marker='o', label='Precisión', color='r')

    plt.title('Recall y Precisión vs Umbrales de Clasificación')
    plt.xlabel('Umbral')
    plt.ylabel('Valor')
    plt.legend(loc='lower left')
    plt.grid(True)
    plt.show()


def reporting_metrics(y_true: pd.Series, y_pred: pd.Series, sample: str) -> None:
    """
    Presenta el resultado de la bondad de ajuste del modelo. Utiliza las siguientes métricas:
    - precision
    - recall
    - f1-score

    :param y_true: Serie target con los valores reales
    :param y_pred: Serie target con los valores predichos
    :param sample: Identificador del tipo de muestra de trabajo ("train" o "test")
    :return: None. La función solo escribe los resultados de las métricas
    """
    print(sample)
    print("")
    print(confusion_matrix(y_true, y_pred))
    print(f"precision: {round(precision_score(y_true, y_pred), 4)}")
    print(f"recall: {round(recall_score(y_true, y_pred), 4)}")
    print(f"f1-score: {round(f1_score(y_true, y_pred), 4)}")
