# Mestrado Mineração

Este projeto contém códigos e análises relacionados ao projeto de mestrado em mineração de dados.

## Estrutura do Projeto

- `mda.py`: Comparativo de classificadores
- `requirements.txt`: Dependências do projeto
- `requirements_streamlit.txt`: Dependências específicas para Streamlit

## Notebooks

- `notebook_estatistica.ipynb`: Análises estatísticas gerais (Friedman e Niemenyi)
- `notebook_experimento.ipynb`: Notebooks de experimentos

## Resultados

O projeto inclui diversos arquivos de resultados:
- Arquivos CSV com resultados das análises
- Gráficos de heatmap para diferentes métricas (Accuracy, F1 Score, Recall, ACSA)

## Como Executar

1. Clone o repositório
2. Crie um ambiente virtual:
   ```bash
   python -m venv venv
   ```
3. Ative o ambiente virtual:
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
4. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
5. Para executar a aplicação Streamlit:
   ```bash
   pip install -r requirements_streamlit.txt
   streamlit run mda.py
   ```
