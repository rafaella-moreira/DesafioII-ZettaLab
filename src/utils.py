import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

def calcular_pesos(
    df,
    variaveis,
    target,
    param_grid=None,
    cv=5,
    scoring='r2'
):
    """
    Calcula os pesos das variáveis explicativas utilizando Regressão Ridge,
    com normalização Min-Max e otimização de hiperparâmetros via GridSearchCV.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados originais.
    variaveis (list): Lista com os nomes das variáveis explicativas.
    target (str): Nome da variável alvo.
    param_grid (dict, opcional): Grade de hiperparâmetros para o Ridge.
                                 Padrão: {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}.
    cv (int, opcional): Número de folds para validação cruzada. Padrão: 5.
    scoring (str, opcional): Métrica de avaliação do modelo. Padrão: 'r2'.

    Retorna:
    pesos (pd.Series): Série com os pesos normalizados das variáveis.
    best_model (Ridge): Melhor modelo Ridge ajustado.
    """

    if param_grid is None:
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

    # Normalização
    scaler = MinMaxScaler()
    df_norm = df.copy()
    df_norm[variaveis] = scaler.fit_transform(df_norm[variaveis])

    X = df_norm[variaveis]
    y = df_norm[target]

    # Modelo Ridge + GridSearch
    ridge = Ridge()

    grid = GridSearchCV(
        estimator=ridge,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring
    )

    grid.fit(X, y)

    best_model = grid.best_estimator_

    # Cálculo dos pesos (coeficientes absolutos normalizados)
    pesos = pd.Series(
        np.abs(best_model.coef_),
        index=variaveis
    )

    pesos = pesos / pesos.sum()

    return pesos.sort_values(ascending=False), best_model




def calcular_indice_socioeconomico(
    df,
    variaveis,
    pesos,
    coluna_estado='estado',
    bins=(0, 0.30, 0.60, 1),
    labels=('baixo', 'médio', 'alto')
):
    """
    Calcula o índice socioeconômico a partir de uma combinação linear ponderada
    das variáveis explicativas e classifica os resultados em faixas.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados normalizados.
    variaveis (list): Lista das variáveis utilizadas no cálculo do índice.
    pesos (pd.Series): Pesos normalizados das variáveis (obtidos via regressão).
    coluna_estado (str, opcional): Nome da coluna que identifica o estado.
    bins (tuple, opcional): Intervalos para classificação do índice.
    labels (tuple, opcional): Rótulos das classes do índice.

    Retorna:
    pd.DataFrame: DataFrame com índice socioeconômico e classificação adicionados.
    """

    df_resultado = df.copy()

    # Garante alinhamento entre variáveis e pesos
    X = df_resultado[variaveis]

    # Índice socioeconômico ponderado
    df_resultado['indice_socioeconomico'] = (X * pesos).sum(axis=1)

    # Normalização Min-Max do índice
    minimo = df_resultado['indice_socioeconomico'].min()
    maximo = df_resultado['indice_socioeconomico'].max()

    df_resultado['indice_socioeconomico'] = (
        (df_resultado['indice_socioeconomico'] - minimo) /
        (maximo - minimo)
    )

    # Classificação do índice
    df_resultado['classificacao'] = pd.cut(
        df_resultado['indice_socioeconomico'],
        bins=bins,
        labels=labels
    )

    # Remove observações inválidas
    df_resultado.dropna(subset=['indice_socioeconomico', 'classificacao'], inplace=True)

    return df_resultado




def rodar_regressao_linear_stats(
    df,
    target,
    variaveis,
    test_size=0.2,
    random_state=42
):
    """
    Ajusta um modelo de Regressão Linear via statsmodels (OLS),
    avalia o desempenho em conjunto de teste e retorna o modelo estimado.

    Parâmetros:
    df (pd.DataFrame): Base de dados.
    target (str): Variável dependente.
    variaveis (list): Variáveis independentes.
    test_size (float, opcional): Proporção do conjunto de teste.
    random_state (int, opcional): Semente para reprodutibilidade.

    Retorna:
    dict contendo:
        - modelo (RegressionResults)
        - metricas (dict)
        - y_pred (np.array)
    """

    X = df[variaveis]
    y = df[target]

    X = sm.add_constant(X)

    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    modelo = sm.OLS(y_treino, X_treino).fit()

    y_pred = modelo.predict(X_teste)

    metricas = {
        'R2': r2_score(y_teste, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_teste, y_pred))
    }

    print(f'\nMODELO OLS – {target.upper()}')
    print('-' * 60)
    print(f"R²:   {metricas['R2']:.4f}")
    print(f"RMSE: {metricas['RMSE']:.4f}")

    return {
        'modelo': modelo,
        'metricas': metricas,
        'y_pred': y_pred
    }
