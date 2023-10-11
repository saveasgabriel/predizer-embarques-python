import numpy as np
from datetime import datetime, timedelta
import pandas as pd

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
import matplotlib.pyplot as plt

class Aprendizado:
    def __init__(self, seed=10):
        self.seed = seed
        self.kfold = KFold(n_splits=10, random_state=self.seed, shuffle=True)
        self.numeric_features = ['Código Terceiro', 'numero_dia_acordada', 'Terceiro Centralizador', 'faixa_de_peso']
        self.categorical_features = ['LINHA', 'Cidade', 'UF']

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

        self.modelos = {
            'Random Forest': RandomForestClassifier(),
            'Decision Tree': DecisionTreeClassifier(),
            'SVM': SVC(),
            'KNN': KNeighborsClassifier(),
            'Gradient Boosting': GradientBoostingClassifier(),
            'AdaBoost': AdaBoostClassifier()
        }

        # Hiperparâmetros para cada modelo
        self.hiperparametros = {
            'Random Forest': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__max_features': ['auto', 'sqrt', 'log2']
            },
            'Decision Tree': {
                'classifier__criterion': ['gini', 'entropy'],
                'classifier__splitter': ['best', 'random'],
                'classifier__max_depth': [None, 10, 20, 30],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            'SVM': {
                'classifier__C': [0.1, 1, 10],
                'classifier__kernel': ['linear', 'rbf', 'poly'],
                'classifier__gamma': ['scale', 'auto', 0.1, 1],
                'classifier__degree': [2, 3, 4]
            },
            'KNN': {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree'],
                'classifier__p': [1, 2]
            },
            'Gradient Boosting': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__learning_rate': [0.01, 0.1, 0.2],
                'classifier__max_depth': [3, 4, 5],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4]
            },
            'AdaBoost': {
                'classifier__n_estimators': [50, 100, 200],
                'classifier__learning_rate': [0.01, 0.1, 0.2]
            }
        }

    def identificar_melhor_modelo(self, X: pd.DataFrame, y: pd.Series):
        melhor_modelo = None
        melhor_score = 0
        X_preprocess = pd.DataFrame(X, columns=self.numeric_features + self.categorical_features)
        y_np = y.values

        for nome_modelo, modelo in self.modelos.items():
            pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', modelo)])
            score = cross_val_score(pipeline, X_preprocess, y_np, cv=self.kfold, error_score='raise')
            score_medio = np.mean(score)

            if score_medio > melhor_score:
                parametros = self.hiperparametros.get(nome_modelo, None)
                melhor_score = score_medio
                melhor_modelo = (nome_modelo, modelo, score, parametros)

        return melhor_modelo
    
    def otimizar_modelo_com_hiperparametros(self, X: pd.DataFrame, y: pd.Series, modelo, hiperparametros):
        X_preprocess = pd.DataFrame(X, columns=self.numeric_features + self.categorical_features)
        y_np = y.values
        pipeline = Pipeline(steps=[('preprocessor', self.preprocessor), ('classifier', modelo)])
        modelo_otimizado = GridSearchCV(estimator=pipeline, param_grid=hiperparametros, cv=self.kfold)
        score = cross_val_score(modelo_otimizado, X_preprocess, y_np, cv=self.kfold, error_score='raise')
        return modelo_otimizado, score
    
    
    
    

class TransformadorDados:
    def __init__(self, treino=False):
        self.treino = treino
        self.label_encoder = LabelEncoder()
        self.limites_peso = [0, 500, 1500, 5000, float('inf')]
        self.rotulos_peso = [500, 1500, 5000, 28000]

    def transformar(self, df):
        
        print(type(df))
        
        if self.treino:
            df = df.drop_duplicates()
            df = df.drop(df[df['Data PCP'] != df['Data de Embarque']].index)
            df = df.drop(df[df['Situação da Entrega'] != 'Faturada'].index)
            df.dropna()

        colunas_desejadas = ['Data PCP', 'LINHA', 'Código Terceiro', 'Cidade', 'UF', 'Peso Líquido Estimado', 'Terceiro Centralizador', 'Produto', 'Data Acordada']
        df = df[colunas_desejadas]

        df['Data Acordada'] = pd.to_datetime(df['Data Acordada'], format='%d/%m/%Y', errors='coerce')
        df['numero_dia_acordada'] = df['Data Acordada'].dt.dayofweek

        df['Data PCP'] = pd.to_datetime(df['Data PCP'], format='%d/%m/%Y', errors='coerce')
        df['numero_dia_pcp'] = df['Data PCP'].dt.dayofweek
        
        # Adicionando encoders nas colunas LINHA, Cidade e UF
        label_encoder = LabelEncoder()
        df['Linha_encoded'] = label_encoder.fit_transform(df['LINHA'])
        df['Cidade_encoded'] = label_encoder.fit_transform(df['Cidade'])
        df['Uf_encoded'] = label_encoder.fit_transform(df['UF'])


        # Remove linhas com valores NaN na coluna 'Peso Líquido Estimado'
        df.dropna(subset=['Peso Líquido Estimado'], inplace=True)

        # Garanta que os valores da coluna 'Peso Líquido Estimado' sejam do tipo float
        df['Peso Líquido Estimado'] = pd.to_numeric(df['Peso Líquido Estimado'], errors='coerce')

        # Lida com valores nulos na coluna 'Peso Líquido Estimado'
        df['Peso Líquido Estimado'].fillna(0, inplace=True)

        # Agora você pode aplicar pd.cut
        df['faixa_de_peso'] = pd.cut(df['Peso Líquido Estimado'], bins=self.limites_peso, labels=self.rotulos_peso)
        df['faixa_de_peso'] = df['faixa_de_peso'].astype(int)

        df['Código Terceiro'] = df['Código Terceiro'].astype(int)
        df['Terceiro Centralizador'] = df['Terceiro Centralizador'].astype(int)

        return df
    
    def recortar_dataframe(self, df, num_dias):
        data_mais_recente = df['Data PCP'].max()
        data_inicio = data_mais_recente - timedelta(days=num_dias)
        df_recortado = df.loc[(df['Data PCP'] >= data_inicio) & (df['Data PCP'] <= data_mais_recente)]
        return df_recortado