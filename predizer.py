import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta
from sklearn.svm import SVC  
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_excel("base_dados_cajamar_embarques.xlsx")


#-----------------------------------------------------------------------------------------------------
df['data_pcp'] = pd.to_datetime(df['data_pcp'], format='%d/%m/%Y')
df['dia_semana'] = df['data_pcp'].dt.strftime('%A')
df['primeira_quinzena'] = df['data_pcp'].apply(lambda x: x.day <= 15)


limites_peso = [0, 500, 1500, 5000, float('inf')]
rotulos_peso = ['até 500', 'até 1500','até 5000', 'mais de 5000']

df['faixa_de_peso'] = pd.cut(df['peso_liquido'], bins=limites_peso, labels=rotulos_peso)

colunas_features = ['linha', 'terceiro_cliente', 'cidade', 'uf', 'primeira_quinzena', 'faixa_de_peso']
X = df[colunas_features]
y = df['dia_semana']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', sparse_output=False)
#-----------------------------------------------------------------------------------------------------

print(X_train)

#X_train_encoded = encoder.feature_names_in_(X_train)

#print(X_train_encoded)


modelos = {
    'Random Forest': RandomForestClassifier(random_state=25),
    'Decision Tree': DecisionTreeClassifier(random_state=25),
    'SVM': SVC(random_state=25),
    'KNN': KNeighborsClassifier()
}


resultados = {}  

for nome_modelo, modelo in modelos.items():
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    modelo.fit(X_train_encoded, y_train)
    y_pred = modelo.predict(X_test_encoded)
    acuracia = accuracy_score(y_test, y_pred)
    resultados[nome_modelo] = acuracia

"""
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train_encoded, y_train)
y_pred = modelo.predict(X_test_encoded)
"""

for nome_modelo, acuracia in resultados.items():
    print(f'Acurácia do modelo {nome_modelo}: {acuracia * 100:.2f}%')


"""
nova_entrada = pd.DataFrame({
    'linha': ['CAJAMAR MOÍDA ATM'],
    'terceiro_cliente': ['JBS S/A - BAU'],
    'cidade': ['BAURU'],
    'uf': ['SP'],
    'primeira_quinzena': [False],
    
})
"""


#"""
nova_entrada = pd.DataFrame({
    'linha': ['CAJ HB ATM '],
    'terceiro_cliente': ['CARREFOUR COMERCIO E INDUSTRIA LTDA'],
    'cidade': ['ITAPEVI'],
    'uf': ['SP'],
    'primeira_quinzena': [False],
    'faixa_de_peso': ['até 500']
    #'nome_produto': ['CARNE MOIDA FORMATADA RESFRIADA DE BOVINO (SELECTION) ATP']
})
#"""

nova_entrada_encoded = encoder.transform(nova_entrada)
dia_previsto = modelo.predict(nova_entrada_encoded)

def data_por_semana_ano(ano, semana, dia_semana_ingles):
    data_referencia = datetime.strptime(f'{ano}-W{semana}-1', '%Y-W%W-%w')

    mapeamento_dias_semana = {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }

    numero_dia_semana = mapeamento_dias_semana.get(dia_semana_ingles, None)

    if numero_dia_semana is not None:
        dias_diferenca = numero_dia_semana - data_referencia.weekday()
        data_prevista = data_referencia + timedelta(days=dias_diferenca)
        data_formatada = data_prevista.strftime('%d/%m/%Y')
        return data_formatada
    else:
        return "Dia da semana inválido"

ano = 2023
semana = 39
data_prevista = data_por_semana_ano(ano, semana, dia_previsto[0])
print(f'Data prevista: {dia_previsto[0]} - {data_prevista}')

