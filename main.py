import pandas as pd
from predizerEmbarque import Aprendizado
import warnings





def fazer():
    warnings.filterwarnings('ignore')
    df_treino = pd.read_excel("base_completa_sem_filtro.xlsx", sheet_name="completo")
    df_teste = pd.read_excel("futuro.xlsx", sheet_name="completo")
    diretorio_previsao = 'previsoes/'
    aprendizado = Aprendizado(df_treino, df_teste, diretorio_previsao)
    aprendizado.dividir_dados_treino()
    aprendizado.identificar_melhor_modelo()
    aprendizado.imprimir_informacoes_modelo()

    """aprendizado.otimizar_modelo_com_hiperparametros()
    aprendizado.imprimir_informacoes_modelo_otimizado()
    aprendizado.dividir_dados_predicao()
    aprendizado.prever_e_salvar()
    return 'Deu certo!'"""

