from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import pandas as pd

# Carregue os dados do Excel em um DataFrame
data = pd.read_excel('dataset.xlsx')

# Crie uma instância do ChatBot
bot = ChatBot('Meu ChatBot')

# Crie um treinador do tipo ListTrainer
trainer = ListTrainer(bot)

# Obtenha as colunas de perguntas e respostas
perguntas = data['pergunta'].astype(str).tolist()
respostas = data['resposta'].astype(str).tolist()

# Prepare as conversas como uma lista de strings
conversas = []
for pergunta, resposta in zip(perguntas, respostas):
    conversas.append(pergunta)
    conversas.append(resposta)

# Treine o chatbot com as perguntas e respostas do Excel
trainer.train(conversas)

# Inicie um loop para interagir com o chatbot
print("Bem-vindo ao ChatBot! Digite 'sair' para sair.")
while True:
    user_input = input("Usuário: ")

    if user_input.lower() == 'sair':
        break

    # Obtenha uma resposta do chatbot com base na entrada do usuário
    response = bot.get_response(user_input)

    print("ChatBot:", response)
