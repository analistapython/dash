# Usando a imagem base oficial do Python
FROM python:3.9

# Definindo o diretório de trabalho dentro do contêiner
WORKDIR /app

# Copiando os arquivos requirements.txt para o diretório de trabalho
COPY requirements.txt .

# Instalando as dependências do Python
RUN pip install --no-cache-dir -r requirements.txt

# Copiando todo o código da aplicação para o diretório de trabalho
COPY . .

# Expondo a porta usada pelo Streamlit
EXPOSE 8501

# Comando para rodar o aplicativo Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]