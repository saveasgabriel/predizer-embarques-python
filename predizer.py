import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, timedelta

df = pd.read_excel("base_dados_cajamar_embarques.xlsx")

df['data_pcp'] = pd.to_datetime(df['data_pcp'], format='%d/%m/%Y')
df['dia_semana'] = df['data_pcp'].dt.strftime('%A')

colunas_features = ['linha', 'terceiro_cliente', 'cidade', 'uf', 'nome_produto']
X = df[colunas_features]
y = df['dia_semana']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

encoder = OneHotEncoder(sparse=False, handle_unknown='ignore', sparse_output=False)
X_train_encoded = encoder.fit_transform(X_train)
X_test_encoded = encoder.transform(X_test)

modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train_encoded, y_train)

y_pred = modelo.predict(X_test_encoded)

acuracia = accuracy_score(y_test, y_pred)
print(f'Acurácia do modelo: {acuracia * 100:.2f}%')

nova_entrada = pd.DataFrame({
    'linha': ['CAJAMAR MOÍDA ATM'],
    'terceiro_cliente': ['JBS S/A - BAU'],
    'cidade': ['BAURU'],
    'uf': ['SP'],
    'nome_produto': ['CARNE MOIDA BOV RESF (DIANTEIRO ATM)']
})
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

    # Encontre o número do dia da semana correspondente
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