# Análise Socioeconômica dos Estados do Sudeste do Brasil (2019–2023)

Este projeto tem como finalidade realizar uma análise dos dados socioeconômicos referentes aos estados da região Sudeste do Brasil, investigando as inter-relações entre os indicadores de renda, educação, saúde e emprego. A partir dessas variáveis, busca-se classificar os estados em diferentes níveis socioeconômicos classificados em alto, médio e baixo, de modo a caracterizar o grau de desenvolvimento regional.
Ademais, a pesquisa visa identificar os fatores que exercem maior influência sobre a formação desse índice, possibilitando compreender os elementos determinantes das disparidades socioeconômicas e prever quais variáveis tendem a impactar de forma mais significativa o desempenho social e econômico das unidades federativas analisadas. E a proposição de recomendações para políticas públicas, com foco nas variáveis que mais contribuem para a obtenção de um elevado índice socioeconômico regional, direcionando ações para os fatores de maior impacto identificados na análise, contribuindo para a tomada de decisão baseada em evidências e para o planejamento de intervenções mais eficientes e direcionadas.

## Base de Dados
Os dados utilizados neste estudo foram obtidos a partir das bases públicas do IBGE, especificamente da PNADC, abrangendo o período de 2019 a 2023. As variáveis analisadas incluem renda média real, taxa de desocupação, escolaridade média, cobertura de saúde, condição de ocupação nos domicílios, entre outras.
O pré-processamento dos dados envolveu a seleção das informações relevantes, a padronização temporal anual e a remoção de valores ausentes, garantindo a consistência e a qualidade das análises subsequentes.

## Metodologia
O pipeline analítico foi desenvolvido em Python, utilizando bibliotecas como pandas, matplotlib, seaborn, scikit-learn, entre outras. A coleta de dados foi realizada a partir de fontes públicas nacionais, seguida das etapas de pré-processamento, com o objetivo de assegurar a consistência, a integridade e a qualidade das informações analisadas.

Na fase de análise exploratória dos dados, foram empregadas matrizes de correlação para identificar relações positivas e negativas entre as variáveis, permitindo uma compreensão inicial das interdependências presentes no conjunto de dados. Nessa etapa, também foi realizado o cálculo inicial e a classificação primária dos índices socioeconômicos, utilizando o modelo de Regressão Linear, a partir da estimação dos pesos das variáveis explicativas em relação ao indicador principal. Com base nesses pesos, procedeu-se à classificação do índice socioeconômico dos estados.

Na etapa de modelagem estatística, foram avaliados três modelos — Regressão Linear, Random Forest e XGBoost — com o objetivo de prever o índice socioeconômico futuro dos estados. Para cada modelo, foi estimado o impacto das variáveis explicativas sobre o indicador principal, seguido do treinamento com dados de 2019 a 2021 e da validação com dados de 2022 e 2023, possibilitando a avaliação do desempenho por estado e por modelo. Posteriormente, os modelos foram utilizados para a projeção do indicador principal (Rendimento Médio Mensal) para anos futuros, com base nos padrões identificados.

A avaliação de desempenho, conduzida por meio de métricas apropriadas, indicou que o modelo XGBoost apresentou maior eficiência e robustez preditiva em comparação aos demais. Adicionalmente, foram empregadas técnicas de interpretação de modelos, como a utilização de SHAP associado à Regressão Linear, para a identificação e comparação da importância e do impacto das variáveis sobre o indicador principal. Por fim, recursos de visualização de dados, incluindo gráficos de dispersão, heatmaps e histogramas, foram utilizados para facilitar a interpretação dos resultados e a extração de insights relevantes.

## Resultados
Identificou-se que a variável rendimento médio real habitual do trabalho principal apresentou a maior relevância entre as analisadas, evidenciando forte correlação com diversos fatores socioeconômicos. Este resultado reforça a importância como indicador preditivo do nível socioeconômico regional. Ademais, o modelo XGBoost apresentou métricas de desempenho superiores em comparação aos demais modelos analisados, indicando maior precisão na previsão do índice socioeconômico. O modelo destacou-se, sobretudo, por apresentar menores erros médios no período de teste, bom desempenho na previsão do indicador principal para anos futuros e elevada consistência na estimação do impacto das variáveis explicativas sobre o indicador.

## Conclusão
Os resultados obtidos evidenciam que o rendimento médio real exerce influência significativa sobre os indicadores sociais, possibilitando a construção de um índice socioeconômico consistente e representativo para os estados da região Sudeste do Brasil.

## Tecnologias Utilizadas
- Python 3.11.9.
- pandas, numpy, matplotlib, seaborn, scikit-learn
- Jupyter Notebook  

## Como Reproduzir
1. Recomendado o uso da versão 3.11.9 do Python
2. Clone o repositório: 
```bash
git clone https://github.com/rafaella-moreira/Desafio-Zetta.git
```
3. Abra a aba do notebook:
```bash
 cd Desafio-Zetta/notebooks
```
4. Execute o notebook:
 - Análise e Exploração dos Dados:
```bash
jupyter notebook EDA.ipynb
```
 - Modelagem e Previsão dos Dados:
```bash
jupyter notebook model.ipynb
```

Ou para executar o arquivo completo:

```bash
jupyter notebook analisedados.ipynb
```
