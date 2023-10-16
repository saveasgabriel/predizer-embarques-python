import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os

from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from category_encoders import TargetEncoder
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

class Aprendizado:
    def __init__(self):
        self.seed = 10
        self.kfold = KFold(n_splits=10, random_state=self.seed, shuffle=True)
        self.features = Features()
        self.numeric_features = self.features.numeric()
        self.categorical_features = self.features.categorical()
        self.hiperparametros = Hiperparametros()

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

    """def prever_com_modelo_otimizado(self, modelo_otimizado, X_novos_dados):
        X_preprocess = pd.DataFrame(X_novos_dados, columns=self.numeric_features + self.categorical_features)
        previsoes = modelo_otimizado.predict(X_preprocess)
        return previsoes"""

    def identificar_melhor_modelo(self, X, y):
        melhor_score = 0
        X_preprocess = pd.DataFrame(X, columns = self.numeric_features + self.categorical_features)
        y_np = y.values

        for nome_modelo, params in self.hiperparametros.get_hiperparametros().items():
            modelo = params['model']
            hiperparametros = params['params']
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', modelo)])
            score = cross_val_score(pipeline, X_preprocess, y_np, cv=self.kfold, error_score='raise')
            score_medio = np.mean(score)

            if score_medio > melhor_score:
                melhor_score = score_medio
                self.informacoes_melhor_modelo = (nome_modelo, modelo, score_medio, score, hiperparametros)
                self.hiperparametros = hiperparametros
                self.melhor_modelo = modelo  


    def imprimir_informacoes_modelo_otimizado(self):
        #print('A acurácia do modelo é: %.2f%%' % (self.modelo_otimizado.score(X_treino,y_treino) *100))
        pass

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


    """
    def prever_e_salvar(self, modelo_otimizado, X_novos_dados, caminho_saida):
        X_preprocess = pd.DataFrame(X_novos_dados, columns=self.numeric_features + self.categorical_features)
        previsoes = modelo_otimizado.predict(X_preprocess)
        df_previsoes = pd.DataFrame(previsoes, columns=['Previsões'])
        df_final = pd.concat([X_preprocess, df_previsoes], axis=1)
        file_path = os.path.join(caminho_saida, 'nome_arquivo.xlsx')
        df_final.to_excel(file_path, index=False)"""
    
    def otimizar_modelo_com_hiperparametros(self, X, y):

        X_preprocess = pd.DataFrame(X, columns=self.numeric_features + self.categorical_features)
        y_np = y.values
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', self.melhor_modelo)])
        modelo = GridSearchCV(estimator=pipeline, param_grid=self.hiperparametros, cv=self.kfold)

        print(self.melhor_modelo)
        print(self.hiperparametros)

        modelo.fit(X_preprocess, y_np)
        self.modelo_otimizado = modelo.best_estimator_

    
class TransformadorDados:
    def __init__(self, treino=False):
        self.treino = treino
        self.label_encoder = LabelEncoder()
        self.limites_peso = [0, 500, 1500, 5000, float('inf')]
        self.rotulos_peso = [500, 1500, 5000, 28000]

    def transformar(self, df):




        
        if self.treino == True:
            df = df.drop_duplicates()
            df = df.drop(df[df['Data PCP'] != df['Data de Embarque']].index)
            df = df.drop(df[df['Situação da Entrega'] != 'Faturada'].index)
            df.dropna()
            df['Data PCP'] = pd.to_datetime(df['Data PCP'], format='%d/%m/%Y', errors='coerce')
            df['numero_dia_pcp'] = df['Data PCP'].dt.dayofweek
            colunas_desejadas = ['Data PCP', 'LINHA', 'Código Terceiro', 'Cidade', 'UF', 'Peso Líquido Estimado', 'Terceiro Centralizador', 'Data Acordada', 'numero_dia_pcp']
        else:
            colunas_desejadas = ['LINHA', 'Código Terceiro', 'Cidade', 'UF', 'Peso Líquido Estimado', 'Terceiro Centralizador', 'Data Acordada']
            df = df.drop_duplicates()
            df.dropna()

        df = df[colunas_desejadas]

        df['Data Acordada'] = pd.to_datetime(df['Data Acordada'], format='%d/%m/%Y', errors='coerce')
        df['numero_dia_acordada'] = df['Data Acordada'].dt.dayofweek


        # Remove linhas com valores NaN na coluna 'Peso Líquido Estimado'
        df.dropna(subset=['Peso Líquido Estimado'], inplace=True)

        # Garanta que os valores da coluna 'Peso Líquido Estimado' sejam do tipo float
        df['Peso Líquido Estimado'] = pd.to_numeric(df['Peso Líquido Estimado'], errors='coerce')

        # Lida com valores nulos na coluna 'Peso Líquido Estimado'
        df['Peso Líquido Estimado'].fillna(0, inplace=True)

        # Agora você pode aplicar pd.cut
        df['faixa_de_peso'] = pd.cut(df['Peso Líquido Estimado'], bins=self.limites_peso, labels=self.rotulos_peso)
        df.dropna(subset=['faixa_de_peso'], inplace=True)
        df['faixa_de_peso'] = df['faixa_de_peso'].astype(int)

        df['Código Terceiro'] = df['Código Terceiro'].astype(int)
        df['Terceiro Centralizador'] = df['Terceiro Centralizador'].astype(int)

        return df
    
    def recortar_dataframe(self, df, num_dias):
        data_mais_recente = df['Data PCP'].max()
        data_inicio = data_mais_recente - timedelta(days=num_dias)
        df_recortado = df.loc[(df['Data PCP'] >= data_inicio) & (df['Data PCP'] <= data_mais_recente)]
        return df_recortado
    
class Features:
    def numeric(self):
       return ['Código Terceiro', 'numero_dia_acordada', 'Terceiro Centralizador', 'faixa_de_peso']
    
    def categorical(self):
       return ['LINHA', 'Cidade', 'UF']
    
    def target(self):
        return ['Data PCP']
    
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