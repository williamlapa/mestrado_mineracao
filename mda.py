import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Análise de Classificadores",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Análise de Classificadores de Machine Learning")

# Authors information
st.sidebar.markdown("""
### Autores
- Fagner Fernandes
  - [LinkedIn](linkedin.com/in/fagner-fernandes-38a25a3b)
- William Lapa Santos Filho
  - [LinkedIn](www.linkedin.com/in/william-lapa)
""")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Menu",
    ["Análise Estatística", "Parâmetros dos Classificadores", "Análise de Custo x Benefício", "Downloads"]
)

# Load data
@st.cache_data
def load_data():
    friedman_results = pd.read_csv('friedman_results.csv', index_col=0)
    nemenyi_accuracy = pd.read_csv('nemenyi_results_Accuracy.csv', index_col=0)
    nemenyi_f1 = pd.read_csv('nemenyi_results_F1 Score.csv', index_col=0)
    nemenyi_recall = pd.read_csv('nemenyi_results_Recall.csv', index_col=0)
    nemenyi_acsa = pd.read_csv('nemenyi_results_ACSA.csv', index_col=0)
    performance_metrics = pd.read_excel('results_with_cost_benefit.xlsx')
    return friedman_results, nemenyi_accuracy, nemenyi_f1, nemenyi_recall, nemenyi_acsa, performance_metrics

# Function to create heatmap
def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title(title)
    return plt

if menu == "Análise Estatística":
    st.header("Resultados das Análises Estatísticas")
    
    try:
        # Load data
        friedman_results, nemenyi_accuracy, nemenyi_f1, nemenyi_recall, nemenyi_acsa, _ = load_data()
        
        # Display Friedman results with full P-value precision
        st.subheader("Resultados do Teste de Friedman")
        
        # Format P-value to show all decimal places
        friedman_results_formatted = friedman_results.copy()
        friedman_results_formatted['P-value'] = friedman_results_formatted['P-value'].apply(lambda x: f"{x:.20f}")
        
        # Display formatted results
        st.dataframe(friedman_results_formatted)
        
        # Display Nemenyi results with tabs
        st.subheader("Resultados do Teste de Nemenyi")
        tab1, tab2, tab3, tab4 = st.tabs(["Accuracy", "F1 Score", "Recall", "ACSA"])
        
        with tab1:
            st.pyplot(plot_heatmap(nemenyi_accuracy, "Teste de Nemenyi - Accuracy"))
            
        with tab2:
            st.pyplot(plot_heatmap(nemenyi_f1, "Teste de Nemenyi - F1 Score"))
            
        with tab3:
            st.pyplot(plot_heatmap(nemenyi_recall, "Teste de Nemenyi - Recall"))
            
        with tab4:
            st.pyplot(plot_heatmap(nemenyi_acsa, "Teste de Nemenyi - ACSA"))
            
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")

elif menu == "Parâmetros dos Classificadores":
    st.header("Parâmetros dos Classificadores")
    
    try:
        # Load performance metrics
        _, _, _, _, _, performance_metrics = load_data()
        
        # Dictionary with classifier parameters and optimization info
        classifiers = {
            'KNN': {
                'params': {
                    'n_neighbors': 1,
                    'p': 1,
                    'weights': 'uniform'
                },
                'optimization': 'Bayesian Search',
                'default_desc': None,  # Não necessário pois está otimizado
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'KNN', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'KNN', 'Memory Usage (MB)'].values[0]
                }
            },
            'SVM': {
                'params': 'Parâmetros padrão',
                'optimization': 'Nenhuma otimização',
                'default_desc': """
                - kernel='rbf' (Função de kernel Gaussiana)
                - C=1.0 (Parâmetro de regularização)
                - gamma='scale' (Coeficiente do kernel)
                - probability=True (Habilita estimativas de probabilidade)
                """,
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'SVM', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'SVM', 'Memory Usage (MB)'].values[0]
                }
            },
            'Decision Tree': {
                'params': {
                    'criterion': 'entropy',
                    'max_depth': 36,
                    'min_samples_leaf': 1,
                    'min_samples_split': 2
                },
                'optimization': 'Bayesian Search',
                'default_desc': None,  # Não necessário pois está otimizado
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Decision Tree', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Decision Tree', 'Memory Usage (MB)'].values[0]
                }
            },
            'LVQ': {
                'params': 'Parâmetros padrão',
                'optimization': 'Nenhuma otimização',
                'default_desc': """
                - n_neighbors=3 (Número de vizinhos)
                - weights='distance' (Ponderação por distância)
                - prototypes_per_class=3 (Protótipos por classe)
                """,
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'LVQ', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'LVQ', 'Memory Usage (MB)'].values[0]
                }
            },
            'MLP': {
                'params': 'Parâmetros padrão',
                'optimization': 'Nenhuma otimização',
                'default_desc': """
                - hidden_layer_sizes=(100,) (Uma camada oculta com 100 neurônios)
                - activation='relu' (Função de ativação ReLU)
                - solver='adam' (Otimizador Adam)
                - learning_rate='constant' (Taxa de aprendizado constante)
                - max_iter=200 (Máximo de iterações)
                """,
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'MLP', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'MLP', 'Memory Usage (MB)'].values[0]
                }
            },
            'Ensemble Neural Network': {
                'params': 'Parâmetros padrão',
                'optimization': 'Nenhuma otimização',
                'default_desc': """
                Conjunto de 3 MLPs com:
                - hidden_layer_sizes=(100,) (Uma camada oculta com 100 neurônios)
                - activation='relu' (Função de ativação ReLU)
                - solver='adam' (Otimizador Adam)
                - max_iter=200 (Máximo de iterações)
                """,
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Ensemble Neural Network', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Ensemble Neural Network', 'Memory Usage (MB)'].values[0]
                }
            },
            'Stacking': {
                'params': 'Parâmetros padrão',
                'optimization': 'Nenhuma otimização',
                'default_desc': """
                Meta-classificador: Regressão Logística com:
                - C=1.0 (Parâmetro de regularização)
                - solver='lbfgs' (Otimizador LBFGS)
                - max_iter=100 (Máximo de iterações)
                
                Classificadores base:
                - Random Forest
                - SVM
                - KNN
                """,
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Stacking', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Stacking', 'Memory Usage (MB)'].values[0]
                }
            },
            'Random Forest': {
                'params': {
                    'max_depth': 28,
                    'n_estimators': 97
                },
                'optimization': 'Optuna',
                'default_desc': None,  # Não necessário pois está otimizado
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Random Forest', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'Random Forest', 'Memory Usage (MB)'].values[0]
                }
            },
            'XGBoost': {
                'params': {
                    'objective': 'binary:logistic',
                    'enable_categorical': False,
                    'eval_metric': 'logloss',
                    'learning_rate': 0.09504317284612004,
                    'max_depth': 6,
                    'n_estimators': 188
                },
                'optimization': 'Optuna',
                'default_desc': None,  # Não necessário pois está otimizado
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'XGBoost', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'XGBoost', 'Memory Usage (MB)'].values[0]
                }
            },
            'LightGBM': {
                'params': 'Parâmetros padrão',
                'optimization': 'Nenhuma otimização',
                'default_desc': """
                - learning_rate=0.1 (Taxa de aprendizado)
                - n_estimators=100 (Número de árvores)
                - max_depth=-1 (Profundidade máxima ilimitada)
                - num_leaves=31 (Número máximo de folhas)
                - min_child_samples=20 (Amostras mínimas por nó folha)
                - objective='binary' (Objetivo para classificação binária)
                """,
                'performance': {
                    'training_time': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'LightGBM', 'Training Time (s)'].values[0],
                    'memory_usage': performance_metrics.loc[performance_metrics['Unnamed: 0'] == 'LightGBM', 'Memory Usage (MB)'].values[0]
                }
            }
        }
        
        # Create tabs for each classifier
        tabs = st.tabs(list(classifiers.keys()))
        
        # Display information in each tab
        for tab, (classifier_name, info) in zip(tabs, classifiers.items()):
            with tab:
                st.subheader(f"{classifier_name}")
                
                # Display optimization method
                st.markdown(f"**Método de Otimização:** {info['optimization']}")
                
                # Display parameters
                st.markdown("**Parâmetros:**")
                if isinstance(info['params'], dict):
                    for param, value in info['params'].items():
                        st.write(f"- {param}: {value}")
                else:
                    st.write(info['params'])
                    if info['default_desc']:
                        st.markdown("**Descrição dos Parâmetros Padrão:**")
                        st.markdown(info['default_desc'])
                
                # Display performance metrics
                st.markdown("**Métricas de Desempenho:**")
                st.write(f"- Tempo de Treinamento: {info['performance']['training_time']:.2f} segundos")
                st.write(f"- Uso de Memória: {info['performance']['memory_usage']:.2f} MB")
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")

elif menu == "Análise de Custo x Benefício":
    st.header("Análise de Custo x Benefício dos Classificadores")
    
    try:
        # Load performance metrics
        _, _, _, _, _, performance_metrics = load_data()
        
        # Criar DataFrame com métricas de desempenho
        cost_benefit_df = pd.DataFrame({
            'Classificador': performance_metrics['Unnamed: 0'],
            'Tempo de Treinamento (s)': performance_metrics['Training Time (s)'].round(2),
            'Uso de Memória (MB)': performance_metrics['Memory Usage (MB)'].round(2),
            'Accuracy': performance_metrics['Accuracy'].round(4),
            'F1 Score': performance_metrics['F1 Score'].round(4),
            'Recall': performance_metrics['Recall'].round(4),
            'ACSA': performance_metrics['ACSA'].round(4)
        })
        
        # Exibir tabela com todas as métricas
        st.subheader("Tabela Comparativa")
        st.dataframe(cost_benefit_df.style.highlight_max(axis=0, color='lightgreen', subset=['Accuracy', 'F1 Score', 'Recall', 'ACSA'])
                                        .highlight_min(axis=0, color='lightpink', subset=['Tempo de Treinamento (s)', 'Uso de Memória (MB)']))
        
        # Criar gráficos
        st.subheader("Visualizações")
        
        # Criar duas colunas para os gráficos
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico de Tempo de Treinamento
            fig_time = plt.figure(figsize=(10, 6))
            plt.barh(cost_benefit_df['Classificador'], cost_benefit_df['Tempo de Treinamento (s)'])
            plt.title('Tempo de Treinamento por Classificador')
            plt.xlabel('Tempo (segundos)')
            plt.ylabel('Classificador')
            st.pyplot(fig_time)
        
        with col2:
            # Gráfico de Uso de Memória
            fig_memory = plt.figure(figsize=(10, 6))
            plt.barh(cost_benefit_df['Classificador'], cost_benefit_df['Uso de Memória (MB)'])
            plt.title('Uso de Memória por Classificador')
            plt.xlabel('Memória (MB)')
            plt.ylabel('Classificador')
            st.pyplot(fig_memory)
        
        # Análise de correlação entre métricas
        st.subheader("Correlação entre Métricas de Custo e Desempenho")
        
        # Calcular correlações
        correlation_metrics = cost_benefit_df[['Tempo de Treinamento (s)', 'Uso de Memória (MB)', 
                                             'Accuracy', 'F1 Score', 'Recall', 'ACSA']].corr()
        
        # Plotar mapa de calor das correlações
        fig_corr = plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_metrics, annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Correlação entre Métricas')
        st.pyplot(fig_corr)
        
        # Adicionar algumas observações
        st.subheader("Observações")
        st.markdown("""
        - Os valores em verde na tabela indicam os melhores resultados para métricas de desempenho (Accuracy, F1 Score, Recall, ACSA)
        - Os valores em rosa na tabela indicam os menores valores para métricas de custo (Tempo de Treinamento, Uso de Memória)
        - O mapa de calor mostra a correlação entre as métricas, onde:
            - Valores próximos a 1 indicam forte correlação positiva
            - Valores próximos a -1 indicam forte correlação negativa
            - Valores próximos a 0 indicam pouca ou nenhuma correlação
        """)
        
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")

elif menu == "Downloads":
    st.header("Downloads dos Notebooks")
    
    st.markdown("""
    ### Notebooks Disponíveis
    
    Aqui você pode baixar os notebooks utilizados na análise:
    """)
    
    # Function to read file as bytes
    def get_binary_file_downloader_html(file_path, file_label):
        with open(file_path, 'rb') as f:
            data = f.read()
        return st.download_button(
            label=f"Download {file_label}",
            data=data,
            file_name=file_path.split('/')[-1],
            mime='application/x-ipynb+json'
        )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Análise Estatística")
        st.markdown("Notebook contendo a análise estatística dos resultados usando os testes de Friedman e Nemenyi.")
        get_binary_file_downloader_html('notebook_estatistica.ipynb', 'Análise Estatística')
    
    with col2:
        st.markdown("#### Experimentos")
        st.markdown("Notebook contendo os experimentos realizados com os diferentes classificadores.")
        get_binary_file_downloader_html('notebook_experimento.ipynb', 'Experimentos')
