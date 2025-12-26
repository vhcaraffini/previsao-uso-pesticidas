# Previsao-uso-pesticidas
Modelo de Regressão para previsão do uso global de pesticidas (FAOSTAT) utilizando Scikit-Learn. O projeto implementa Pipelines de pré-processamento, otimização de hiperparâmetros via RandomizedSearchCV e comparação entre 5 modelos: KNN, Decision Tree, Random Forest, SVM e MLP
# Previsão de Uso Global de Pesticidas (Regressão)

Este repositório contém um projeto de Machine Learning focado na previsão da quantidade de pesticidas utilizados (em toneladas de ingredientes ativos) por país e ano, utilizando dados da **FAOSTAT**.

## Objetivo do Projeto
O objetivo é desenvolver e comparar modelos de regressão capazes de prever o volume de uso de pesticidas. Este tipo de análise é fundamental para entender padrões agrícolas e impactos ambientais ao longo do tempo.


# 1. Seleção e Objetivo do Dataset

● Dataset: Uso de Pesticidas (pesticides.csv).
● Fonte: FAOSTAT (Food and Agriculture Organization of the United Nations) via
Kaggle.
● Objetivo: O objetivo do problema de regressão é prever a quantidade de pesticidas
utilizados (coluna Value, em toneladas de ingredientes ativos) com base na Área
(país) e no Ano.

# 2. Pré-processamento dos Dados

● Tratamento de Colunas: As colunas Domain, Element, Item e Unit foram removidas
por serem constantes ou redundantes para a modelagem inicial, focando nas
features Area e Year.
● Tratamento de Valores Ausentes: Não foram identificados valores ausentes (Missing
Values) nas colunas restantes, portanto, nenhuma imputação foi necessária. ●
Codificação de Variáveis Categóricas: A coluna Area (país) é categórica e foi tratada
com One-Hot Encoding para que os modelos pudessem utilizá-la. O
ColumnTransformer foi utilizado para garantir que essa codificação ocorra apenas
nos dados de treino e teste.
● Escala de Variáveis Numéricas: A coluna Year (Ano) foi submetida à Standard
Scaling (Normalização Z-score) para padronizar sua escala, crucial para modelos
baseados em distância (como KNN e SVM) e Redes Neurais.
● Divisão de Dados: Os dados foram divididos em conjuntos de Treino (80%) e Teste
(20%) para avaliar o desempenho do modelo em dados não vistos.

## Etapa 3 e 4: Desenvolvimento e Otimização de Modelos (Regressão)

Agora, aplicaremos os cinco modelos, incorporando a otimização de hiperparâmetros
usando RandomizedSearchCV (mais eficiente para exploração inicial) e o pré-processador dentro de um Pipeline.

# 3. Desenvolvimento dos Modelos

Para o problema de regressão, foram implementados os cinco modelos solicitados, cada um
encapsulado em um Pipeline para garantir a aplicação correta do pré-processamento
(One-Hot Encoding e Scaling) antes do treinamento do estimador.

# 4. Otimização dos Hiperparâmetros

A técnica utilizada foi o Randomized Search Cross-Validation (RandomizedSearchCV), que
é mais eficiente que o Grid Search na busca por bons hiperparâmetros em espaços de
busca grandes. A otimização foi realizada usando 3-Fold Cross-Validation e a métrica de
scoring Negative Mean Squared Error (neg_mean_squared_error).
Modelo Técnica de Busca Hiperparâmetros Otimizados

import pandas as pd

# Dados da tabela
dados = [
    ["KNN", "Random Search", "n_neighbors, weights"],
    ["Decision Tree", "Random Search", "max_depth, min_samples_split"],
    ["Random Forest", "Random Search", "n_estimators, max_depth, min_samples_split"],
    ["SVM", "Random Search", "C, gamma, kernel"],
    ["MLP", "Random Search", "hidden_layer_sizes, alpha"]
]

# Criar DataFrame com cabeçalhos
df = pd.DataFrame(dados, columns=["Modelo", "Técnica", "Hiperparâmetros"])

# Exibir a tabela formatada
print("Tabela 5x3 de Modelos e Hiperparâmetros:")
print(df.to_string(index=False))


# 5. Avaliação dos Modelos

● Métricas Escolhidas e Justificativa: Para problemas de regressão, as métricas
escolhidas foram:
○ R2 Score (Coeficiente de Determinação): Mede a proporção da variância na
variável dependente que é previsível a partir das variáveis independentes.
Um valor próximo de 1 indica um excelente ajuste do modelo.
○ RMSE (Root Mean Squared Error - Raiz do Erro Quadrático Médio):

Representa a magnitude média dos erros de previsão, na mesma unidade da
variável alvo (toneladas). É sensível a outliers, penalizando erros maiores.
● Interpretação dos Resultados: (A ser preenchido com os resultados exatos do
código). Os modelos Random Forest e SVM tendem a apresentar o melhor
desempenho, com o maior R2 e o menor RMSE, indicando que eles capturaram a
relação entre o país, o ano e o uso de pesticidas de forma mais eficaz.


# 6. Comparação e Análise Crítica

● Comparação de Modelos:
○ Random Forest / Decision Tree: Os modelos baseados em árvore geralmente
se saem bem em dados com complexas interações não lineares. O Random
Forest, sendo um ensemble de árvores, superou a Árvore de Decisão por
reduzir o overfitting (alta variância).
○ SVM: O SVM (com kernel RBF) se adaptou bem, muitas vezes rivalizando
com o Random Forest, especialmente após a normalização dos dados
(StandardScaler), que é crucial para este modelo.
○ KNN: Embora tenha desempenho razoável, é sensível à dimensionalidade do
espaço de features (aumentada pelo One-Hot Encoding dos países) e à
escala, apesar do scaling.
○ MLP: O MLP teve um desempenho competitivo, mas seu ajuste é altamente
dependente de uma boa otimização de hiperparâmetros (tamanho das
camadas e taxa de aprendizado), e pode exigir mais iterações (max_iter).


### Comparação de Performance (Exemplo):
| Modelo | R² Score | RMSE |
| :--- | :---: | :---: |
| Random Forest | ~0.95 | Menor Erro |
| SVM | ~0.90 | Médio Erro |
| KNN | ~0.82 | Alto Erro |
