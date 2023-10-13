import pandas as pd
from predizerEmbarque import Aprendizado, TransformadorDados
import warnings


warnings.filterwarnings('ignore')


num_dias = 35
num_seed = 10


aprendizado = Aprendizado(seed=num_seed) 

df_original = pd.read_excel("base_completa_sem_filtro.xlsx", sheet_name="completo")

transformador = TransformadorDados(treino=True)

df_transformado = transformador.transformar(df_original)
df_treino_recortado = transformador.recortar_dataframe(df_transformado, num_dias)

colunas_features = ['Código Terceiro', 'LINHA', 'Cidade', 'UF', 'numero_dia_acordada', 'Terceiro Centralizador', 'faixa_de_peso']
colunas_target = ['numero_dia_pcp']
X_treino = df_treino_recortado[colunas_features]
y_treino = df_treino_recortado[colunas_target]

aprendizado = Aprendizado(10)

modelo_inicial = aprendizado.identificar_melhor_modelo(X_treino, y_treino)

modelo = modelo_inicial[1]
hyperparametros = modelo_inicial[3]

print(modelo)

print(hyperparametros)

modelo_otimizado = aprendizado.otimizar_modelo_com_hiperparametros(X_treino, y_treino, modelo, hyperparametros)

print(modelo_otimizado)
#Imprimir os resultados

#print(scores)


    
    
