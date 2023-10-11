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
df_treino_recortado.dropna()
colunas_features = ['Código Terceiro', 'LINHA', 'Cidade', 'UF', 'numero_dia_acordada', 'Terceiro Centralizador', 'faixa_de_peso']
colunas_target = ['numero_dia_pcp']
X_treino = df_treino_recortado[colunas_features]
y_treino = df_treino_recortado[colunas_target]

aprendizado = Aprendizado(10)

scores = aprendizado.identificar_melhor_modelo(X_treino, y_treino)

# Imprimir os resultados

print(scores)


    
    
