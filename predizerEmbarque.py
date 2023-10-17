# Imports para manipulação de dados e cálculos numéricos
import numpy as np
import pandas as pd

# Imports relacionados a ajustes de data e hora
from datetime import datetime, timedelta

# Imports para manipulação de sistema operacional
import os

# Imports para pré-processamento e transformação de dados
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer

# Imports para codificação de variáveis categóricas
from category_encoders import TargetEncoder

# Imports relacionados a algoritmos de aprendizado de máquina
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Imports para construção de pipelines e transformações de colunas
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Imports relacionados a métricas de avaliação de modelos
from sklearn.metrics import accuracy_score

class Aprendizado:
    def __init__(self, df_treino:pd.DataFrame, df_predizer:pd.DataFrame, dir_previsao, nome_arquivo='previsao_embarques'):
        self.seed = 10
        self.num_dias = 35
        self.kfold = KFold(n_splits=10, random_state=self.seed, shuffle=True)
        self.features = Features()
        self.numeric_features = self.features.numeric()
        self.categorical_features = self.features.categorical()
        self.hiperparametros = Hiperparametros()
        self.transformador = TransformadorDados()
        self.df_treino = df_treino
        self.df_predizer = df_predizer
        self.dir_previsao = dir_previsao
        self.nome_arquivo = nome_arquivo

        self.numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler()),
            ('std_scaler', StandardScaler())
        ])

        self.categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('target_encoder', TargetEncoder())
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.numeric_transformer, self.numeric_features),
                ('cat', self.categorical_transformer, self.categorical_features)
            ])
        
        self.X_treino = None
        self.y_treino = None
        self.X_predizer = None
        self.informacoes_melhor_modelo = None
        self.melhor_modelo = None
        self.modelo_otimizado = None
        
    def dividir_dados_treino(self):
        df = self.transformador.transformar(self.df_treino, treino=True)
        df_recortado = self.transformador.recortar_dataframe(df, num_dias=self.num_dias)
        X = pd.DataFrame(df_recortado, columns = self.numeric_features + self.categorical_features)
        y = pd.DataFrame(df_recortado, columns = [self.features.target()[1]])
        self.X_treino = X
        self.y_treino = y.values

    def dividir_dados_predicao(self):
        df = self.transformador.transformar(self.df_predizer, treino=False)
        self.X_predizer = pd.DataFrame(df, columns = self.numeric_features + self.categorical_features)
        self.X_predizer.reset_index(drop=True)

    def identificar_melhor_modelo(self):
        melhor_score = 0
        for nome_modelo, params in self.hiperparametros.get_hiperparametros().items():
            modelo = params['model']
            hiperparametros = params['params']
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', modelo)])
            score = cross_val_score(pipeline, self.X_treino, self.y_treino, cv=self.kfold, error_score='raise')
            score_medio = np.mean(score)

            if score_medio > melhor_score:
                melhor_score = score_medio
                self.informacoes_melhor_modelo = (nome_modelo, modelo, score_medio, score, hiperparametros)
                self.hiperparametros = hiperparametros
                self.melhor_modelo = modelo  

    def imprimir_informacoes_modelo_otimizado(self):
        print('A acurácia do modelo é: %.2f%%' % (self.modelo_otimizado.score(self.X_treino,self.y_treino) *100))
  
    def imprimir_informacoes_modelo(self):
        nome_modelo, modelo, score_medio, scores, hiperparametros = self.informacoes_melhor_modelo
        print("Nome do modelo:", nome_modelo)
        print("Modelo:", modelo)
        print(f"Score médio: {score_medio:.2f}")
        print("Scores unitários:")
        for i, score in enumerate(scores):
            print(f"    Fold {i+1}: {score:.2f}")
        print("Hiperparâmetros:")
        for parametro, valores in hiperparametros.items():
            print(f"    {parametro}: {valores}")
    
    def prever_e_salvar(self):
        previsoes = self.modelo_otimizado.predict(self.X_predizer)
        df_previsoes = pd.DataFrame(previsoes, columns=['dia_semana_data_pcp'])
        df_final = pd.concat([self.df_predizer, df_previsoes], axis=1)
        df_final = df_final.reset_index(drop=True)
        file_path = os.path.join(self.dir_previsao, (self.nome_arquivo+'.xlsx'))
        df_final.to_excel(file_path, index=False)

        print(df_final)
    
    def otimizar_modelo_com_hiperparametros(self):
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', self.melhor_modelo)])
        modelo = GridSearchCV(estimator=pipeline, param_grid=self.hiperparametros, cv=self.kfold)
        modelo.fit(self.X_treino, self.y_treino)
        self.modelo_otimizado = modelo.best_estimator_

    
class TransformadorDados:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.limites_peso = [0, 500, 1500, 5000, float('inf')]
        self.rotulos_peso = [500, 1500, 5000, 28000]
        self.features = Features()

    def transformar(self, df, treino):
        if treino:
            df = df[df[self.features.target()[0]] == df[self.features.reference()[0]]]
            df = df[df[self.features.decision()[0]] == 'Faturada']
            df[self.features.target()[0]] = pd.to_datetime(df[self.features.target()[0]], format='%d/%m/%Y', errors='coerce')
            df[self.features.target()[1]] = df[self.features.target()[0]].dt.dayofweek
            colunas_desejadas = [self.features.target()[0]] + self.features.categorical() + [self.features.numeric()[0]] + [self.features.numeric()[2]] + [self.features.target()[1]] + self.features.avulsos()
        else:
            colunas_desejadas = self.features.categorical() + [self.features.numeric()[0]] + [self.features.numeric()[2]] + self.features.avulsos()

        df = df[colunas_desejadas]

        df[self.features.avulsos()[0]] = pd.to_datetime(df[self.features.avulsos()[0]], format='%d/%m/%Y', errors='coerce')
        df[self.features.numeric()[1]] = df[self.features.avulsos()[0]].dt.dayofweek

        # Remove linhas com valores NaN na coluna 'Peso Líquido Estimado'
        df.dropna(subset=[self.features.avulsos()[1]], inplace=True)
        df[self.features.avulsos()[1]] = pd.to_numeric(df[self.features.avulsos()[1]], errors='coerce')
        df[self.features.avulsos()[1]].fillna(0, inplace=True)

        # Agora você pode aplicar pd.cut
        df[self.features.numeric()[3]] = pd.cut(df[self.features.avulsos()[1]], bins=self.limites_peso, labels=self.rotulos_peso)
        df.dropna(subset=self.features.numeric()[3], inplace=True)
        df[self.features.numeric()[3]] = df[self.features.numeric()[3]].astype(int)

        df[self.features.numeric()[0]] = df[self.features.numeric()[0]].astype(int)
        df[self.features.numeric()[2]] = df[self.features.numeric()[2]].astype(int)

        df.dropna(inplace=True)

        return df

    def recortar_dataframe(self, df, num_dias):
        data_mais_recente = df[self.features.target()[0]].max()
        data_inicio = data_mais_recente - timedelta(days=num_dias)
        df_recortado = df.loc[(df[self.features.target()[0]] >= data_inicio) & (df[self.features.target()[0]] <= data_mais_recente)]
        return df_recortado

class Features:
    def numeric(self):
       return ['Código Terceiro', 'numero_dia_acordada', 'Terceiro Centralizador', 'faixa_de_peso']
    
    def categorical(self):
       return ['LINHA', 'Cidade', 'UF']
    
    def target(self):
        return ['Data PCP', 'numero_dia_pcp']
    
    def decision(self):
        return ['Situação da Entrega']

    def reference(self):
        return ['Data de Embarque']
    
    def avulsos(self):
        return ['Data Acordada', 'Peso Líquido Estimado']
    
class Hiperparametros:
    def get_hiperparametros(self):
        return {
            'Random Forest': {
                'model': RandomForestClassifier(),
                'params': {
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4],
                    'classifier__max_features': ['auto', 'sqrt', 'log2']
                }
            },
            'Decision Tree': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'classifier__criterion': ['gini', 'entropy'],
                    'classifier__splitter': ['best', 'random'],
                    'classifier__max_depth': [None, 10, 20, 30],
                    'classifier__min_samples_split': [2, 5, 10],
                    'classifier__min_samples_leaf': [1, 2, 4]
                }
            },
            'SVM': {
                'model': SVC(),
                'params': {
                    'classifier__C': [0.1, 1, 10],
                    'classifier__kernel': ['linear', 'rbf', 'poly'],
                    'classifier__gamma': ['scale', 'auto', 0.1, 1],
                    'classifier__degree': [2, 3, 4]
                }
            },
            'KNN': {
                'model': KNeighborsClassifier(),
                'params': {
                    'classifier__n_neighbors': [3, 5, 7, 9],
                    'classifier__weights': ['uniform', 'distance'],
                    'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree'],
                    'classifier__p': [1, 2]
                }
            },
            'Gradient Boosting': {
                'model': GradientBoostingClassifier(),
                'params': {
                    'classifier__learning_rate': [0.01, 0.1, 0.2],
                    'classifier__n_estimators': [100, 200, 300],
                    'classifier__max_depth': [3, 4, 5]
                                }
            },
            'AdaBoost': {
                'model': AdaBoostClassifier(),
                'params': {
                    'classifier__n_estimators': [50, 100, 200],
                    'classifier__learning_rate': [0.01, 0.1, 0.2]
                }
            }
        }