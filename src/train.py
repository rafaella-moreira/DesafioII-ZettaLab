import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression

def treinar_modelo_e_extrair_impacto(
    df,
    variaveis,
    target,
    modelo,
    param_grid=None,
    normalizar=True,
    test_size=0.2,
    random_state=42,
    cv=5,
    scoring='r2'
):
    """
    Treina um modelo supervisionado com otimização de hiperparâmetros,
    avalia seu desempenho e extrai a importância das variáveis de forma genérica.

    Compatível com modelos lineares, Random Forest, XGBoost, etc.

    Parâmetros:
    df (pd.DataFrame): DataFrame contendo os dados.
    variaveis (list): Lista de variáveis explicativas.
    target (str): Nome da variável alvo.
    modelo: Estimador scikit-learn (ex: Ridge(), RandomForestRegressor()).
    param_grid (dict, opcional): Grade de hiperparâmetros do GridSearchCV.
    normalizar (bool, opcional): Aplica MinMaxScaler nas variáveis. Padrão True.
    test_size (float, opcional): Proporção do conjunto de teste.
    random_state (int, opcional): Semente para reprodutibilidade.
    cv (int, opcional): Número de folds da validação cruzada.
    scoring (str, opcional): Métrica de avaliação.

    Retorna:
    dict contendo:
        - best_model
        - metricas
        - importancias (pd.Series)
        - df_norm (pd.DataFrame)
    """

    df_proc = df.copy()

    # Normalização (opcional)
    if normalizar:
        scaler = MinMaxScaler()
        df_proc[variaveis] = scaler.fit_transform(df_proc[variaveis])

    X = df_proc[variaveis]
    y = df_proc[target]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # GridSearch
    grid = GridSearchCV(
        estimator=modelo,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    # Previsão
    y_pred = best_model.predict(X_test)

    # Métricas
    metricas = {
        'R2': r2_score(y_test, y_pred),
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
    }

    # Extração genérica de importância
    if hasattr(best_model, 'coef_'):
        importancias = pd.Series(
            np.abs(best_model.coef_),
            index=variaveis
        )

    elif hasattr(best_model, 'feature_importances_'):
        importancias = pd.Series(
            best_model.feature_importances_,
            index=variaveis
        )

    else:
        raise ValueError(
            "O modelo não possui coef_ nem feature_importances_. "
            "Considere usar métodos como SHAP ou Permutation Importance."
        )

    importancias = importancias.sort_values(ascending=False)

    return {
        'best_model': best_model,
        'metricas': metricas,
        'importancias': importancias,
        'df_norm': df_proc
    }



def prever_por_estado_modelo(
    df,
    estados,
    variaveis,
    target,
    modelo,
    anos_treino=(2019, 2021),
    anos_teste=(2022, 2023),
    anos_futuros=(2024, 2025, 2026)
):
    """
    Treina um modelo de regressão por estado, avalia o desempenho no período
    de teste e gera previsões futuras do indicador alvo.

    Parâmetros:
    df (pd.DataFrame): Base de dados completa.
    estados (list): Lista de estados a serem analisados.
    variaveis (list): Variáveis explicativas.
    target (str): Variável alvo.
    modelo: Estimador de regressão (ex: LinearRegression(), Ridge(), RF, XGB).
    anos_treino (tuple): Intervalo de anos para treino.
    anos_teste (tuple): Intervalo de anos para teste.
    anos_futuros (tuple): Anos a serem previstos.

    Retorna:
    dict contendo:
        - metricas_gerais
        - comparacao_teste
        - previsoes_futuras
    """

    metricas_gerais = []
    comparacao_teste = []
    previsoes_futuras = []

    for estado in estados:

        df_estado = df[df['estado'] == estado].copy()
        df_estado = df_estado.sort_values('ano')

        train_df = df_estado[df_estado['ano'].between(*anos_treino)]
        test_df  = df_estado[df_estado['ano'].between(*anos_teste)]

        # Escalonadores
        scaler_X = MinMaxScaler()
        scaler_y = MinMaxScaler()

        X_train = scaler_X.fit_transform(train_df[variaveis])
        y_train = scaler_y.fit_transform(train_df[[target]]).ravel()

        X_test = scaler_X.transform(test_df[variaveis])
        y_test = scaler_y.transform(test_df[[target]]).ravel()

        # Treino do modelo principal
        modelo.fit(X_train, y_train)

        y_pred = modelo.predict(X_test)

        # Retorno à escala original
        y_test_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
        y_pred_real = scaler_y.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        for ano, real, prev in zip(test_df['ano'], y_test_real, y_pred_real):
            comparacao_teste.append({
                'estado': estado,
                'ano': ano,
                'valor_real': real,
                'valor_previsto': prev,
                'erro': prev - real
            })

        # Métricas
        metricas_gerais.append({
            'estado': estado,
            'R2': r2_score(y_test, y_pred) if len(y_test) >= 2 else np.nan,
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred))
        })

        # Normalização do tempo
        scaler_ano = MinMaxScaler()
        df_estado['ano_norm'] = scaler_ano.fit_transform(df_estado[['ano']])

        anos_fut = pd.DataFrame({'ano': anos_futuros})
        anos_fut['ano_norm'] = scaler_ano.transform(anos_fut[['ano']])

        # Projeção das variáveis explicativas
        variaveis_previstas = pd.DataFrame(index=anos_fut['ano'])

        for var in variaveis:
            lr_var = LinearRegression()
            lr_var.fit(df_estado[['ano_norm']], df_estado[var])
            variaveis_previstas[var] = lr_var.predict(
                anos_fut[['ano_norm']]
            )

        # Previsão do target futuro
        X_futuro = scaler_X.transform(variaveis_previstas)
        y_fut = modelo.predict(X_futuro)

        y_fut_real = scaler_y.inverse_transform(
            y_fut.reshape(-1, 1)
        ).ravel()

        for ano, valor in zip(anos_futuros, y_fut_real):
            previsoes_futuras.append({
                'estado': estado,
                'ano': ano,
                'rendimento_medio_previsto': valor
            })

    return {
        'metricas_gerais': pd.DataFrame(metricas_gerais),
        'comparacao_teste': pd.DataFrame(comparacao_teste),
        'previsoes_futuras': pd.DataFrame(previsoes_futuras)
    }
