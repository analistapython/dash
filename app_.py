import requests
import pandas as pd
from prophet import Prophet
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI  # Certifique-se de que esta biblioteca está instalada
from langchain.prompts import PromptTemplate

# Definir o timeout padrão para as requisições
DEFAULT_TIMEOUT = 10

# Função para obter dados históricos do Bitcoin usando CryptoCompare
def get_historical_data(limit=2000):
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
        'fsym': 'BTC',
        'tsym': 'USD',
        'limit': limit,
        'aggregate': 1
    }
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        data = response.json()
        if 'Data' in data['Data']:
            prices = data['Data']['Data']
            df = pd.DataFrame(prices)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volumeto'].astype(float)
            return df[['timestamp', 'close', 'volume']]
        else:
            st.error("Erro: Resposta da API não contém dados esperados.")
            st.stop()
    else:
        st.error(f"Erro ao obter dados históricos: {response.status_code}")
        st.stop()

# Função para obter dados históricos do Ethereum usando CryptoCompare
def get_ethereum_data(limit=2000):
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
        'fsym': 'ETH',
        'tsym': 'USD',
        'limit': limit,
        'aggregate': 1
    }
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        data = response.json()
        if 'Data' in data['Data']:
            prices = data['Data']['Data']
            df = pd.DataFrame(prices)
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            df['close'] = df['close'].astype(float)
            return df[['timestamp', 'close']]
        else:
            st.error("Erro: Resposta da API não contém dados esperados.")
            st.stop()
    else:
        st.error(f"Erro ao obter dados históricos: {response.status_code}")
        st.stop()

# Função para obter o preço em tempo real do Bitcoin e a taxa de câmbio BRL/USD usando CoinGecko
def get_realtime_price():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': 'bitcoin,usd,brl',
        'vs_currencies': 'usd,brl'
    }
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        data = response.json()
        return data['bitcoin']['usd'], data['bitcoin']['brl'], data['usd']['brl']
    else:
        st.error(f"Erro ao obter preço em tempo real: {response.status_code}")
        st.stop()

# Função para obter o preço do ouro em USD usando CoinGecko
def get_gold_price():
    url = 'https://api.coingecko.com/api/v3/simple/price'
    params = {
        'ids': 'tether-gold',
        'vs_currencies': 'usd'
    }
    response = requests.get(url, params=params, timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        data = response.json()
        return data['tether-gold']['usd']
    else:
        st.error(f"Erro ao obter preço do ouro: {response.status_code}")
        st.stop()

# Função para obter a dominância do Bitcoin
def get_btc_dominance():
    url = 'https://api.coingecko.com/api/v3/global'
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        data = response.json()
        return data['data']['market_cap_percentage']['btc']
    else:
        st.error(f"Erro ao obter dominância do Bitcoin: {response.status_code}")
        st.stop()

# Função para obter o índice de medo e ganância
def get_fear_greed_index():
    url = 'https://api.alternative.me/fng/?limit=1'
    response = requests.get(url, timeout=DEFAULT_TIMEOUT)
    if response.status_code == 200:
        data = response.json()
        return data['data'][0]['value'], data['data'][0]['value_classification']
    else:
        st.error(f"Erro ao obter índice de medo e ganância: {response.status_code}")
        st.stop()

# Função para projetar o preço futuro do Bitcoin
def forecast_price(bitcoin_data):
    df = bitcoin_data[['timestamp', 'close']].rename(columns={'timestamp': 'ds', 'close': 'y'})
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast_df = model.predict(future)
    return forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

# Função para calcular o Mayer Multiple
def calculate_mayer_multiple(bitcoin_data):
    bitcoin_data['200d_MA'] = bitcoin_data['close'].rolling(window=200).mean()
    bitcoin_data['Mayer Multiple'] = bitcoin_data['close'] / bitcoin_data['200d_MA']
    return bitcoin_data

# Função para calcular indicadores técnicos
def calculate_technical_indicators(bitcoin_data):
    bitcoin_data['50d_MA'] = bitcoin_data['close'].rolling(window=50).mean()
    bitcoin_data['200d_MA'] = bitcoin_data['close'].rolling(window=200).mean()
    delta = bitcoin_data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    bitcoin_data['RSI'] = 100 - (100 / (1 + rs))
    return bitcoin_data

# Função para análise usando ChatGPT via LangChain
def analyze_dashboard(historical_data, forecast_data, btc_usd, btc_brl, gold_brl_per_gram, btc_dominance, fear_greed_value, fear_greed_classification):
    analysis_text = f"""
    Aqui está a análise do dashboard do Bitcoin:
    
    1. **Preço Atual do Bitcoin**:
       - USD: ${btc_usd:,.2f}
       - BRL: R${btc_brl:,.2f}

    2. **Preço do Ouro por Grama**:
       - BRL: R${gold_brl_per_gram:,.2f}

    3. **Dominância do Bitcoin no Mercado**:
       - {btc_dominance:.2f}%

    4. **Índice de Medo e Ganância**:
       - Valor: {fear_greed_value}
       - Classificação: {fear_greed_classification}

    5. **Mayer Multiple Atual**:
       - Valor: {historical_data['Mayer Multiple'].iloc[-1]:.2f}

    6. **Estatísticas Históricas**:
       - Menor Valor Histórico (BRL): R${historical_data['close_brl'].min():,.2f}
       - Maior Valor Histórico (BRL): R${historical_data['close_brl'].max():,.2f}
       - Média Histórica (BRL): R${historical_data['close_brl'].mean():,.2f}

    7. **Projeção Futura**:
       - Para os próximos 30 dias, o modelo prevê um preço médio de R${forecast_data['yhat'].mean() * usd_brl:,.2f} com limites entre R${forecast_data['yhat_lower'].mean() * usd_brl:,.2f} e R${forecast_data['yhat_upper'].mean() * usd_brl:,.2f}.

    8. **Indicadores Técnicos**:
       - Média Móvel de 50 dias: R${historical_data['50d_MA'].iloc[-1]:,.2f}
       - Média Móvel de 200 dias: R${historical_data['200d_MA'].iloc[-1]:,.2f}
       - RSI: {historical_data['RSI'].iloc[-1]:.2f}

    Esta análise fornece uma visão abrangente do estado atual do Bitcoin, bem como suas tendências históricas e previsões futuras.
    """

    # Configurar a LangChain
    llm = OpenAI(api_key='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')  # Note a mudança aqui
    prompt = PromptTemplate(input_variables=["text"], template="{text}")
    chain = LLMChain(llm=llm, prompt=prompt)

    # Obter a análise
    response = chain.run(text=analysis_text)
    return response

# Obtenha dados históricos
historical_data = get_historical_data()

# Obtenha dados históricos do Ethereum
ethereum_data = get_ethereum_data()

# Obtenha o preço em tempo real e a taxa de câmbio BRL/USD
btc_usd, btc_brl, usd_brl = get_realtime_price()

# Obtenha o preço do ouro em USD e converta para BRL por grama
gold_usd_per_ounce = get_gold_price()
gold_usd_per_gram = gold_usd_per_ounce / 31.1035  # Convertendo para preço por grama
gold_brl_per_gram = gold_usd_per_gram * usd_brl

# Obtenha a dominância do Bitcoin
btc_dominance = get_btc_dominance()

# Obtenha o índice de medo e ganância
fear_greed_value, fear_greed_classification = get_fear_greed_index()

# Converta preços históricos para BRL
historical_data['close_brl'] = historical_data['close'] * usd_brl
ethereum_data['close_brl'] = ethereum_data['close'] * usd_brl

# Calcule indicadores técnicos
historical_data = calculate_technical_indicators(historical_data)

# Projeção futura
forecast_data = forecast_price(historical_data)

# Calcule o Mayer Multiple
historical_data = calculate_mayer_multiple(historical_data)

# Análise do dashboard
analysis = analyze_dashboard(historical_data, forecast_data, btc_usd, btc_brl, gold_brl_per_gram, btc_dominance, fear_greed_value, fear_greed_classification)

# Crie o dashboard usando Streamlit
st.title("Análise do Preço do Bitcoin")

# Exiba a análise do ChatGPT
st.write("### Análise da Linguagem Natural")
st.write(analysis)

# Filtro de data
st.write("### Filtro de Data")
start_date = st.date_input("Data Inicial", value=historical_data['timestamp'].min())
end_date = st.date_input("Data Final", value=historical_data['timestamp'].max())

# Filtra os dados com base nas datas selecionadas
filtered_data = historical_data[
    (historical_data['timestamp'] >= pd.to_datetime(start_date)) &
    (historical_data['timestamp'] <= pd.to_datetime(end_date))
]

# Cards com estatísticas históricas
st.write("### Estatísticas Históricas")
col1, col2, col3 = st.columns(3)
col1.metric(label="Menor Valor Histórico (BRL)", value=f"R${historical_data['close_brl'].min():,.2f}")
col2.metric(label="Maior Valor Histórico (BRL)", value=f"R${historical_data['close_brl'].max():,.2f}")
col3.metric(label="Média Histórica (BRL)", value=f"R${historical_data['close_brl'].mean():,.2f}")

# Cards com o valor do Bitcoin em tempo real e do ouro
st.write("### Informações em Tempo Real")
col4, col5, col6 = st.columns(3)
col4.metric(label="Preço do Bitcoin (USD)", value=f"${btc_usd:,.2f}")
col5.metric(label="Preço do Bitcoin (BRL)", value=f"R${btc_brl:,.2f}")
col6.metric(label="Preço do Ouro por Grama (BRL)", value=f"R${gold_brl_per_gram:,.2f}")

# Card com o Mayer Multiple
st.write("### Mayer Multiple Atual")
st.metric(label="Mayer Multiple Atual", value=f"{historical_data['Mayer Multiple'].iloc[-1]:.2f}")

# Card com a dominância do Bitcoin
st.write("### Dominância do Bitcoin")
st.metric(label="Dominância do Bitcoin (%)", value=f"{btc_dominance:.2f}%")

# Card com o índice de medo e ganância
st.write("### Índice de Medo e Ganância")
st.metric(label="Índice de Medo e Ganância", value=f"{fear_greed_value} ({fear_greed_classification})")

# Gráfico da Projeção Futura
st.write("### Projeção Futura para os Próximos 30 Dias")
forecast_fig = go.Figure()
forecast_fig.add_trace(go.Scatter(
    x=forecast_data['ds'], y=forecast_data['yhat'] * usd_brl,
    mode='lines', name='Preço Previsto', line=dict(color='greenyellow')
))
forecast_fig.add_trace(go.Scatter(
    x=forecast_data['ds'], y=forecast_data['yhat_lower'] * usd_brl,
    mode='lines', name='Limite Inferior', fill='tonexty', line=dict(color='tomato')
))
forecast_fig.add_trace(go.Scatter(
    x=forecast_data['ds'], y=forecast_data['yhat_upper'] * usd_brl,
    mode='lines', name='Limite Superior', fill='tonexty', line=dict(color='blue')
))
forecast_fig.update_layout(
    title='Projeção Futura do Preço do Bitcoin (BRL)',
    xaxis_title='Data', yaxis_title='Preço (BRL)'
)
st.plotly_chart(forecast_fig)

# Gráfico do Preço Histórico do Bitcoin
st.write("### Preço Histórico do Bitcoin")
historical_fig = px.line(
    filtered_data, x='timestamp', y='close_brl',
    labels={'timestamp': 'Data', 'close_brl': 'Preço (BRL)'},
    title='Preço Histórico do Bitcoin (BRL)'
)
historical_fig.update_traces(line=dict(color='green'))
st.plotly_chart(historical_fig)

# Gráfico do Mayer Multiple
st.write("### Mayer Multiple do Bitcoin")
mayer_fig = px.line(
    filtered_data, x='timestamp', y='Mayer Multiple',
    labels={'timestamp': 'Data', 'Mayer Multiple': 'Mayer Multiple'},
    title='Mayer Multiple do Bitcoin'
)
mayer_fig.update_traces(line=dict(color='orange'))
st.plotly_chart(mayer_fig)

# Gráfico do Volume de Transações
st.write("### Volume de Transações do Bitcoin")
volume_fig = px.bar(
    filtered_data, x='timestamp', y='volume',
    labels={'timestamp': 'Data', 'volume': 'Volume'},
    title='Volume de Transações do Bitcoin'
)
volume_fig.update_traces(marker_color='purple')
st.plotly_chart(volume_fig)

# Gráfico Comparativo com Ethereum
st.write("### Comparação com Ethereum")
comparison_fig = go.Figure()
comparison_fig.add_trace(go.Scatter(
    x=historical_data['timestamp'], y=historical_data['close_brl'],
    mode='lines', name='Bitcoin (BRL)', line=dict(color='blue')
))
comparison_fig.add_trace(go.Scatter(
    x=ethereum_data['timestamp'], y=ethereum_data['close_brl'],
    mode='lines', name='Ethereum (BRL)', line=dict(color='red')
))
comparison_fig.update_layout(
    title='Comparação de Preço: Bitcoin vs Ethereum (BRL)',
    xaxis_title='Data', yaxis_title='Preço (BRL)'
)
st.plotly_chart(comparison_fig)

# Para executar o Streamlit, utilize o comando: streamlit run app.py