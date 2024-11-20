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

# Sidebar menu
menu = st.sidebar.selectbox(
    "Menu",
    ["Análise Estatística", "Parâmetros dos Classificadores", "Downloads"]
)

# Load data
@st.cache_data
def load_data():
    friedman_results = pd.read_csv('friedman_results.csv', index_col=0)
    nemenyi_accuracy = pd.read_csv('nemenyi_results_Accuracy.csv', index_col=0)
    nemenyi_f1 = pd.read_csv('nemenyi_results_F1 Score.csv', index_col=0)
    nemenyi_recall = pd.read_csv('nemenyi_results_Recall.csv', index_col=0)
    nemenyi_acsa = pd.read_csv('nemenyi_results_ACSA.csv', index_col=0)
    return friedman_results, nemenyi_accuracy, nemenyi_f1, nemenyi_recall, nemenyi_acsa

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
        friedman_results, nemenyi_accuracy, nemenyi_f1, nemenyi_recall, nemenyi_acsa = load_data()
        
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
    
    # Dictionary with classifier parameters and optimization info
    classifiers = {
        'KNN': {
            'params': {
                'n_neighbors': 1,
                'p': 1,
                'weights': 'uniform'
            },
            'optimization': 'Bayesian Search',
            'default_desc': None  # Não necessário pois está otimizado
        },
        'SVM': {
            'params': 'Parâmetros padrão',
            'optimization': 'Nenhuma otimização',
            'default_desc': """
            - kernel='rbf' (Função de kernel Gaussiana)
            - C=1.0 (Parâmetro de regularização)
            - gamma='scale' (Coeficiente do kernel)
            - probability=True (Habilita estimativas de probabilidade)
            """
        },
        'Decision Tree': {
            'params': {
                'criterion': 'entropy',
                'max_depth': 36,
                'min_samples_leaf': 1,
                'min_samples_split': 2
            },
            'optimization': 'Bayesian Search',
            'default_desc': None  # Não necessário pois está otimizado
        },
        'LVQ': {
            'params': 'Parâmetros padrão',
            'optimization': 'Nenhuma otimização',
            'default_desc': """
            - n_neighbors=3 (Número de vizinhos)
            - weights='distance' (Ponderação por distância)
            - prototypes_per_class=3 (Protótipos por classe)
            """
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
            """
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
            """
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
            """
        },
        'Random Forest': {
            'params': {
                'max_depth': 28,
                'n_estimators': 97
            },
            'optimization': 'Optuna',
            'default_desc': None  # Não necessário pois está otimizado
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
            'default_desc': None  # Não necessário pois está otimizado
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
            """
        }
    }
    
    # Create tabs for each classifier
    tabs = st.tabs(list(classifiers.keys()))
    
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
