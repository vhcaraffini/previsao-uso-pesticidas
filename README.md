# Previsao-uso-pesticidas
Modelo de Regressão para previsão do uso global de pesticidas (FAOSTAT) utilizando Scikit-Learn. O projeto implementa Pipelines de pré-processamento, otimização de hiperparâmetros via RandomizedSearchCV e comparação entre 5 modelos: KNN, Decision Tree, Random Forest, SVM e MLP
# Previsão de Uso Global de Pesticidas (Regressão)

Este repositório contém um projeto de Machine Learning focado na previsão da quantidade de pesticidas utilizados (em toneladas de ingredientes ativos) por país e ano, utilizando dados da **FAOSTAT**.

## Objetivo do Projeto
O objetivo é desenvolver e comparar modelos de regressão capazes de prever o volume de uso de pesticidas. Este tipo de análise é fundamental para entender padrões agrícolas e impactos ambientais ao longo do tempo.

## Tecnologias Utilizadas
* **Linguagem:** Python 3.x
* **Bibliotecas:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn.
* **Técnicas:** Pipelines de pré-processamento, One-Hot Encoding, StandardScaler e RandomizedSearchCV.

## Estrutura do Pipeline de Dados
O projeto utiliza um `ColumnTransformer` integrado a um `Pipeline` para garantir a integridade dos dados e evitar o vazamento de dados (data leakage):
1. **Atributos Categóricos (`Area`):** Transformados via One-Hot Encoding.
2. **Atributos Numéricos (`Year`):** Padronizados via StandardScaler (Z-score).



[Image of machine learning workflow diagram]


## Modelos Comparados
Foram implementados e otimizados cinco algoritmos principais:
* **KNN** (K-Nearest Neighbors)
* **Decision Tree** (Árvore de Decisão)
* **Random Forest** (Floresta Aleatória)
* **SVM** (Support Vector Machine)
* **MLP** (Multi-layer Perceptron / Redes Neurais)

## Resultados e Métricas
Os modelos foram avaliados utilizando as métricas **R² Score** (proximidade do ajuste) e **RMSE** (magnitude do erro em toneladas). 

### Comparação de Performance (Exemplo):
| Modelo | R² Score | RMSE |
| :--- | :---: | :---: |
| Random Forest | ~0.95 | Menor Erro |
| SVM | ~0.90 | Médio Erro |
| KNN | ~0.82 | Alto Erro |
