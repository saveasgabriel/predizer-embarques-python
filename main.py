import pandas as pd
from predizerEmbarque import Aprendizado, TransformadorDados
import warnings


warnings.filterwarnings('ignore')


df_treino = pd.read_excel("base_completa_sem_filtro.xlsx", sheet_name="completo")
df_teste = pd.read_excel("futuro.xlsx", sheet_name="completo")

aprendizado = Aprendizado(df_treino)
aprendizado.dividir_dados_treino()
aprendizado.identificar_melhor_modelo()
aprendizado.imprimir_informacoes_modelo()
aprendizado.otimizar_modelo_com_hiperparametros()
aprendizado.imprimir_informacoes_modelo_otimizado()




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

    
    
