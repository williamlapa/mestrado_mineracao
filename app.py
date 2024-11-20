import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import ADASYN
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
import openml
import warnings
warnings.filterwarnings('ignore')

def load_data():
    try:
        # Carregar o dataset do OpenML
        dataset = openml.datasets.get_dataset(722)
        
        # Obter o dataframe
        df = dataset.get_data(dataset_format='dataframe')[0]
        
        # A última coluna é a target
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Converter para numpy array
        X = X.values
        y = y.values
            
        # Garantir que y seja um array 1D
        if y is not None and y.ndim > 1:
            y = y.ravel()
            
        if y is None:
            raise ValueError("Dataset não contém a variável target (y)")
            
        return X, y
    except Exception as e:
        st.error(f"Erro ao carregar os dados: {str(e)}")
        return None, None

def preprocess_data(X, y):
    try:
        # Verificar se os dados são válidos
        if X is None or y is None:
            raise ValueError("Dados de entrada inválidos (X ou y é None)")
            
        # Remover colunas com todos os valores zero
        zero_cols = np.where(~X.any(axis=0))[0]
        X = np.delete(X, zero_cols, axis=1)
        
        # Converter para float64
        X = X.astype(np.float64)
        
        # Preencher valores ausentes com a média
        for col in range(X.shape[1]):
            mask = np.isnan(X[:, col])
            if np.any(mask):
                X[mask, col] = np.mean(X[~mask, col])
        
        # Label encoding para a variável target
        le = LabelEncoder()
        y = le.fit_transform(y)
        # Garantir que y continue unidimensional
        if y.ndim > 1:
            y = y.ravel()
        
        # Split dos dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scaling dos dados
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Aplicar ADASYN para balancear as classes
        adasyn = ADASYN(random_state=42)
        X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train_scaled, y_train)
        
        return X_train_resampled, X_test_scaled, y_train_resampled, y_test
    except Exception as e:
        st.error(f"Erro no pré-processamento dos dados: {str(e)}")
        return None, None, None, None

def create_classifiers():
    # Criar classificadores base com seus hiperparâmetros
    classifiers = {
        'KNN': {
            'model': KNeighborsClassifier(n_neighbors=5),
            'params': {
                'n_neighbors': 5,
                'weights': 'uniform',
                'algorithm': 'auto',
                'metric': 'minkowski'
            }
        },
        'SVM': {
            'model': SVC(probability=True, kernel='rbf', C=1.0),
            'params': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale',
                'probability': True
            }
        },
        'Decision Tree': {
            'model': DecisionTreeClassifier(max_depth=5),
            'params': {
                'criterion': 'gini',
                'max_depth': 5,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        },
        'Neural Network': {
            'model': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000),
            'params': {
                'hidden_layer_sizes': (100, 50),
                'activation': 'relu',
                'solver': 'adam',
                'max_iter': 1000,
                'learning_rate': 'constant'
            }
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100),
            'params': {
                'n_estimators': 100,
                'criterion': 'gini',
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            }
        },
        'XGBoost': {
            'model': XGBClassifier(objective='binary:logistic', eval_metric='logloss'),
            'params': {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'learning_rate': 0.1,
                'max_depth': 3,
                'n_estimators': 100
            }
        },
        'LightGBM': {
            'model': LGBMClassifier(),
            'params': {
                'boosting_type': 'gbdt',
                'num_leaves': 31,
                'max_depth': -1,
                'learning_rate': 0.1,
                'n_estimators': 100
            }
        },
        'Ensemble Neural Network': {
            'model': VotingClassifier(
                estimators=[
                    ('mlp1', MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000)),
                    ('mlp2', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000)),
                    ('mlp3', MLPClassifier(hidden_layer_sizes=(75, 50, 25), max_iter=1000))
                ],
                voting='soft'
            ),
            'params': {
                'voting': 'soft',
                'mlp1__hidden_layer_sizes': (100,),
                'mlp2__hidden_layer_sizes': (50, 25),
                'mlp3__hidden_layer_sizes': (75, 50, 25),
                'max_iter': 1000
            }
        }
    }
    
    return classifiers

def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Treinar o modelo
    model.fit(X_train, y_train)
    
    # Fazer previsões
    y_pred = model.predict(X_test)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, f1, recall, conf_matrix

def evaluate_all_models(classifiers, X_train, X_test, y_train, y_test):
    """Avalia todos os modelos e retorna um DataFrame com os resultados."""
    results = []
    
    for name, clf_dict in classifiers.items():
        try:
            with st.spinner(f'Avaliando {name}...'):
                model = clf_dict['model']
                accuracy, f1, recall, conf_matrix = evaluate_model(
                    model, X_train, X_test, y_train, y_test
                )
                
                results.append({
                    'Modelo': name,
                    'Acurácia': f'{accuracy:.4f}',
                    'F1 Score': f'{f1:.4f}',
                    'Recall': f'{recall:.4f}'
                })
        except Exception as e:
            st.error(f"Erro ao avaliar {name}: {str(e)}")
            results.append({
                'Modelo': name,
                'Acurácia': 'Erro',
                'F1 Score': 'Erro',
                'Recall': 'Erro'
            })
    
    return pd.DataFrame(results)

def main():
    st.title('Avaliação de Modelos de Machine Learning')
    st.write('Este aplicativo compara diferentes classificadores em um conjunto de dados de classificação binária.')
    
    # Carregar e preprocessar dados
    with st.spinner('Carregando e preprocessando dados...'):
        X, y = load_data()
        if X is None or y is None:
            st.error("Não foi possível carregar os dados. Por favor, verifique os logs de erro acima.")
            return
            
        X_train, X_test, y_train, y_test = preprocess_data(X, y)
        if any(v is None for v in [X_train, X_test, y_train, y_test]):
            st.error("Erro durante o pré-processamento dos dados. Por favor, verifique os logs de erro acima.")
            return
    
    # Criar classificadores
    classifiers = create_classifiers()
    
    # Criar abas
    tab1, tab2 = st.tabs(["Avaliação Individual", "Comparação de Modelos"])
    
    with tab1:
        # Seleção do modelo
        selected_model = st.selectbox(
            'Selecione o modelo para avaliar:',
            list(classifiers.keys())
        )
        
        # Mostrar hiperparâmetros do modelo selecionado
        st.subheader('Hiperparâmetros do Modelo')
        params = classifiers[selected_model]['params']
        
        # Criar colunas para organizar os hiperparâmetros
        cols = st.columns(3)
        for i, (param, value) in enumerate(params.items()):
            col_idx = i % 3
            with cols[col_idx]:
                st.metric(
                    label=param,
                    value=str(value),
                    help=f'Hiperparâmetro: {param}'
                )
        
        if st.button('Avaliar Modelo'):
            try:
                with st.spinner(f'Avaliando {selected_model}...'):
                    model = classifiers[selected_model]['model']
                    accuracy, f1, recall, conf_matrix = evaluate_model(
                        model, X_train, X_test, y_train, y_test
                    )
                    
                    # Exibir resultados
                    st.subheader('Resultados da Avaliação')
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric('Acurácia', f'{accuracy:.4f}')
                    with col2:
                        st.metric('F1 Score', f'{f1:.4f}')
                    with col3:
                        st.metric('Recall', f'{recall:.4f}')
                    
                    st.subheader('Matriz de Confusão')
                    st.write(pd.DataFrame(
                        conf_matrix,
                        columns=['Previsto Negativo', 'Previsto Positivo'],
                        index=['Real Negativo', 'Real Positivo']
                    ))
            except Exception as e:
                st.error(f"Erro ao avaliar o modelo: {str(e)}")
    
    with tab2:
        st.subheader('Comparação de Todos os Modelos')
        if st.button('Avaliar Todos os Modelos'):
            try:
                # Avaliar todos os modelos
                results_df = evaluate_all_models(classifiers, X_train, X_test, y_train, y_test)
                
                # Mostrar tabela de resultados
                st.subheader('Resultados Comparativos')
                st.dataframe(
                    results_df.style.highlight_max(
                        subset=['Acurácia', 'F1 Score', 'Recall'],
                        color='lightgreen'
                    ),
                    hide_index=True,
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Erro ao avaliar todos os modelos: {str(e)}")

if __name__ == '__main__':
    main()
