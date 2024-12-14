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
st.markdown("<h1 style='text-align: center;'>Mineração de Dados</h1>", unsafe_allow_html=True)
st.title("Análise de Classificadores de Machine Learning")

# Authors information
st.sidebar.image("cin.png", use_column_width=True)

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
    ["Análise Estatística", 
     "Parâmetros dos Classificadores",
     "Desempenho dos modelos",
     "Balanceamento classes ADASYN",
     "Base Treino x Teste",
     "Explainable AI",
     "Downloads"]
)

# Add development information
st.sidebar.markdown("""
Aplicativo desenvolvido em Dezembro de 2024 como produto dos estudos realizados na disciplina **Mineração de Dados**
""")

# Load data
@st.cache_data
def load_data():
    friedman_results = pd.read_csv('friedman_results.csv', index_col=0)
    nemenyi_accuracy = pd.read_csv('nemenyi_results_Accuracy.csv', index_col=0)
    nemenyi_f1 = pd.read_csv('nemenyi_results_F1 Score.csv', index_col=0)
    nemenyi_recall = pd.read_csv('nemenyi_results_Recall.csv', index_col=0)
    nemenyi_acsa = pd.read_csv('nemenyi_results_ACSA.csv', index_col=0)
    performance_metrics = pd.read_excel('results_with_cost_benefit.xlsx')
    adasyn_results = pd.read_excel('results_adasyn_comparison.xlsx')
    train_test_results = pd.read_excel('results_train_teste_comparison.xlsx')
    return friedman_results, nemenyi_accuracy, nemenyi_f1, nemenyi_recall, nemenyi_acsa, performance_metrics, adasyn_results, train_test_results

# Function to create heatmap
def plot_heatmap(data, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(data, annot=True, cmap='YlGnBu', fmt='.3f')
    plt.title(title)
    return plt

def show_parametros_classificadores():
    st.title("Parâmetros dos Classificadores")
    
    st.markdown("""
    ### Parâmetros Otimizados via Busca Bayesiana
    
    Os parâmetros dos classificadores foram otimizados utilizando a técnica de Busca Bayesiana (Bayesian Search), 
    que é um método eficiente para otimização de hiperparâmetros. Esta técnica utiliza princípios bayesianos para 
    guiar a busca pelos melhores parâmetros, sendo mais eficiente que a busca em grade tradicional.
    """)

    # Create tabs for each classifier
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "KNN", 
        "Decision Tree", 
        "Random Forest", 
        "XGBoost", 
        "LightGBM"
    ])

    with tab1:
        st.subheader("K-Nearest Neighbors (KNN)")
        st.markdown("""
        - **algorithm**: 'ball_tree'
        - **leaf_size**: 20
        - **n_neighbors**: 1
        - **weights**: 'distance'
        """)

    with tab2:
        st.subheader("Decision Tree")
        st.markdown("""
        - **criterion**: 'gini'
        - **max_depth**: 10
        - **min_samples_leaf**: 1
        - **min_samples_split**: 2
        """)

    with tab3:
        st.subheader("Random Forest")
        st.markdown("""
        - **max_depth**: 10
        - **max_features**: 'sqrt'
        - **min_samples_leaf**: 1
        - **min_samples_split**: 2
        - **n_estimators**: 11
        """)

    with tab4:
        st.subheader("XGBoost")
        st.markdown("""
        - **colsample_bytree**: 1.0
        - **learning_rate**: 0.3
        - **max_depth**: 8
        - **n_estimators**: 200
        - **subsample**: 1.0
        """)

    with tab5:
        st.subheader("LightGBM")
        st.markdown("""
        - **colsample_bytree**: 1.0
        - **learning_rate**: 0.3
        - **min_child_samples**: 10
        - **n_estimators**: 200
        - **num_leaves**: 20
        - **subsample**: 0.5
        """)

if menu == "Análise Estatística":
    st.header("Resultados das Análises Estatísticas")
    
    try:
        # Load data
        friedman_results, nemenyi_accuracy, nemenyi_f1, nemenyi_recall, nemenyi_acsa, _, _, _ = load_data()
        
        # 1. Teste de Normalidade
        st.subheader("1. Teste de Normalidade")
        st.markdown("""
        O teste de Shapiro-Wilk foi aplicado para verificar a normalidade dos dados. Os resultados indicaram que:
        
        - Os dados **não seguem uma distribuição normal** (Rejeitamos a hipótese nula)
        - O teste foi aplicado na feature 'f9' da base de dados
        
        Isso justifica o uso de testes não-paramétricos (Friedman e Nemenyi) para a análise estatística dos resultados.
        """)
        
        # 2. Teste de Friedman
        st.subheader("2. Teste de Friedman")
        
        # Format P-value to show all decimal places
        friedman_results_formatted = friedman_results.copy()
        friedman_results_formatted['P-value'] = friedman_results_formatted['P-value'].apply(lambda x: f"{x:.20f}")
        
        # Display formatted results
        st.dataframe(friedman_results_formatted)
        
        # 3. Teste pos-hoc de Nemenyi
        st.subheader("3. Teste pos-hoc de Nemenyi")
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
    show_parametros_classificadores()

elif menu == "Desempenho dos modelos":
    st.title("Desempenho dos Modelos")
    
    try:
        # Load performance data
        df = pd.read_excel('results_antes_apos_otimizacao.xlsx')
        
        # Get the number of models (half of the rows since we have before/after for each)
        n_models = len(df) // 2
        
        # Separate data before and after optimization
        df_antes = df.iloc[:n_models]
        df_depois = df.iloc[n_models:]
        
        # Create tabs for different views
        tab1, tab2 = st.tabs(["Métricas de Performance", "Recursos Computacionais"])
        
        with tab1:
            st.subheader("Métricas de Performance antes e após otimização")
            
            # Create a comparison table
            comparison_df = pd.DataFrame({
                'Modelo': df_depois['Modelo'],
                'Accuracy Antes': df_antes['Accuracy'].round(4),
                'Accuracy Depois': df_depois['Accuracy'].round(4),
                'F1 Score Antes': df_antes['F1 Score'].round(4),
                'F1 Score Depois': df_depois['F1 Score'].round(4),
                'Recall Antes': df_antes['Recall'].round(4),
                'Recall Depois': df_depois['Recall'].round(4),
                'ACSA Antes': df_antes['ACSA'].round(4),
                'ACSA Depois': df_depois['ACSA'].round(4)
            })
            
            # Display the comparison table with highlighting
            st.dataframe(comparison_df.style.highlight_max(axis=0, subset=[col for col in comparison_df.columns if col != 'Modelo']))
            
        with tab2:
            st.subheader("Recursos Computacionais")
            
            # Create two columns for the graphs
            col1, col2 = st.columns(2)
            
            with col1:
                # Processing Time Comparison
                fig_time = plt.figure(figsize=(10, 6))
                x = np.arange(len(df_antes['Modelo']))
                width = 0.35
                
                plt.bar(x - width/2, df_antes['Training Time (s)'], width, label='Antes', color='lightcoral')
                plt.bar(x + width/2, df_depois['Training Time (s)'], width, label='Depois', color='lightgreen')
                
                plt.xlabel('Modelos')
                plt.ylabel('Tempo (segundos)')
                plt.title('Tempo de Processamento: Antes vs Depois da Otimização')
                plt.xticks(x, df_antes['Modelo'], rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                
                st.pyplot(fig_time)
                
            with col2:
                # Memory Usage Comparison
                fig_memory = plt.figure(figsize=(10, 6))
                
                plt.bar(x - width/2, df_antes['Memory Usage (MB)'], width, label='Antes', color='lightcoral')
                plt.bar(x + width/2, df_depois['Memory Usage (MB)'], width, label='Depois', color='lightgreen')
                
                plt.xlabel('Modelos')
                plt.ylabel('Memória (MB)')
                plt.title('Uso de Memória: Antes vs Depois da Otimização')
                plt.xticks(x, df_antes['Modelo'], rotation=45, ha='right')
                plt.legend()
                plt.tight_layout()
                
                st.pyplot(fig_memory)
            
            # Add observations
            st.markdown("""
            ### Observações:
            - As barras em vermelho claro representam o desempenho antes da otimização
            - As barras em verde claro representam o desempenho após a otimização
            - O tempo de processamento é medido em segundos
            - O uso de memória é medido em Megabytes (MB)
            """)
            
            # Display raw data in expandable section
            with st.expander("Ver dados brutos"):
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Antes da Otimização")
                    st.dataframe(df_antes)
                with col2:
                    st.subheader("Após a Otimização")
                    st.dataframe(df_depois)
            
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        st.error("Estrutura do DataFrame:")
        st.write(df.head())

elif menu == "Balanceamento classes ADASYN":
    st.header("Balanceamento de Classes usando ADASYN")
    
    try:
        # Load ADASYN results
        _, _, _, _, _, _, adasyn_results, _ = load_data()
        
        tab1, tab2 = st.tabs(["Distribuição de Classes", "Resultados"])
        
        with tab1:
            st.subheader("Distribuição de Classes")
            st.image("class_distribution.png", caption="Distribuição de Classes antes e depois do ADASYN", use_column_width=True)
            
        with tab2:
            st.subheader("Resultados após ADASYN")
            
            # Display the results dataframe
            if not adasyn_results.empty:
                st.dataframe(adasyn_results, use_container_width=True)
            else:
                st.error("Não foi possível carregar os resultados do ADASYN.")
                
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")

elif menu == "Base Treino x Teste":
    st.header("Comparação entre Base de Treino e Teste")
    
    try:
        # Load train-test comparison results
        _, _, _, _, _, _, _, train_test_results = load_data()
        
        # Display the results
        st.dataframe(train_test_results)
            
    except Exception as e:
        st.error(f"Erro ao carregar os dados de comparação treino-teste: {str(e)}")

elif menu == "Explainable AI":
    st.header("Explainable AI - LIME Results")
    
    tab1, tab2 = st.tabs(["KNN", "XGBoost"])
    
    with tab1:
        st.subheader("LIME Results for KNN")
        
        # Display KNN parameters
        st.markdown("### KNN Parameters")
        st.markdown("""
        - **n_neighbors**: 1
        - **weights**: 'uniform'
        - **p**: 1 (Manhattan distance)
        """)
        
        # Display LIME results image
        st.image("lime_results_knn.png", caption="LIME Results for KNN", use_column_width=True)
        
    with tab2:
        st.subheader("LIME Results for XGBoost")
        
        # Display XGBoost parameters
        st.markdown("### XGBoost Parameters")
        st.markdown("""
        - **objective**: binary:logistic
        - **enable_categorical**: False
        - **eval_metric**: logloss
        - **learning_rate**: 0.09504
        - **max_depth**: 6
        - **n_estimators**: 188
        """)
        
        # Display LIME results image
        st.image("lime_results_xgboost.png", caption="LIME Results for XGBoost", use_column_width=True)

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
