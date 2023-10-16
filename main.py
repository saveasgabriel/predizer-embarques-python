import pandas as pd
from predizerEmbarque import Aprendizado, TransformadorDados
import warnings


warnings.filterwarnings('ignore')


num_dias = 35
num_seed = 10


df_original = pd.read_excel("base_completa_sem_filtro.xlsx", sheet_name="completo")

df_teste = pd.read_excel("futuro.xlsx", sheet_name="completo")

transformador = TransformadorDados(treino=True)

df_transformado = transformador.transformar(df_original)

df_treino_recortado = transformador.recortar_dataframe(df_transformado, num_dias)

colunas_features = ['Código Terceiro', 'LINHA', 'Cidade', 'UF', 'numero_dia_acordada', 'Terceiro Centralizador', 'faixa_de_peso']

colunas_target = ['numero_dia_pcp']

print(df_treino_recortado)

X_treino = df_treino_recortado[colunas_features]

y_treino = df_treino_recortado[colunas_target]




aprendizado = Aprendizado()

aprendizado.identificar_melhor_modelo(X_treino, y_treino)
aprendizado.imprimir_informacoes_modelo()
aprendizado.otimizar_modelo_com_hiperparametros(X_treino, y_treino)
print('A acurácia do modelo é: %.2f%%' % (aprendizado.modelo_otimizado.score(X_treino,y_treino) *100))


"""
modelo = modelo_inicial[1]

hyperparametros = modelo_inicial[3]

print(modelo)

print(hyperparametros)

modelo_otimizado = aprendizado.otimizar_modelo_com_hiperparametros(X_treino, y_treino, modelo, hyperparametros)

print(modelo_otimizado)

print('A acurácia do modelo é: %.2f%%' % (modelo_otimizado.score(X_treino,y_treino) *100))


transformador_teste = TransformadorDados(treino=False)
df_teste_transformado = transformador_teste.transformar(df_teste)
print(df_teste_transformado)

diretorio_previsao = 'previsoes\\'

aprendizado.prever_e_salvar(modelo_otimizado, df_teste_transformado, diretorio_previsao)


"""

    
    
