## IMPORTAÇÃO DAS BIBLIOTECAS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy

## PRÉ-PROCESSAMENTO DOS DADOS

# Carregando o conjunto de dados em um DataFrame
df = pd.read_excel('dataset.xlsx')

# Verificando os primeiros registros do DataFrame
#print(df.head())

## PRÉ-PROCESSAMENTO DE TEXTO

# Download das stopwords do NLTK
#nltk.download('stopwords')

# Download do tokenizer do NLTK
#nltk.download('punkt')

# Carregando as stopwords
stop_words = set(stopwords.words('portuguese'))


# Carregar o modelo pré-treinado em português
nlp = spacy.load('pt_core_news_sm')

# Função para remover nomes próprios de um texto
def remove_proper_names(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if not token.ent_type_]
    return ' '.join(tokens)

# Lista de palavras de saudação e despedida
greetings = ['olá', 'oi', 'bom dia', 'boa tarde', 'boa noite']
farewells = ['tchau', 'até logo', 'adeus', 'obrigado', 'obrigada']

# Função para remover saudações e despedidas de um texto
def remove_greetings_and_farewells(text, greetings, farewells):
    tokens = text.split()
    filtered_tokens = [token for token in tokens if token.lower() not in greetings and token.lower() not in farewells]
    return ' '.join(filtered_tokens)

# Função para pré-processar uma string de texto
def preprocess_text(text):
    if isinstance(text, str):  # Verifica se o valor é uma string
        # Convertendo para minúsculas
        text = text.lower()
        # Tokenização das palavras
        tokens = word_tokenize(text)
        # Remoção das stopwords
        tokens = [token for token in tokens if token not in stop_words]
        # Junção dos tokens em uma string
        preprocessed_text = ' '.join(tokens)

        preprocessed_text = remove_proper_names(preprocessed_text)
        preprocessed_text = remove_greetings_and_farewells(preprocessed_text, greetings=greetings, farewells=farewells)

        return preprocessed_text
    else:
        return ''  # Retorna uma string vazia para valores nulos

# Pré-processando as perguntas e respostas
df['pergunta_processada'] = df['pergunta'].apply(preprocess_text)
df['resposta_processada'] = df['resposta'].apply(preprocess_text)
# df['pergunta_processada'] = df['pergunta_processada'].apply(remove_proper_names)
# df['resposta_processada'] = df['resposta_processada'].apply(remove_proper_names)
# df['pergunta_processada'] = df['pergunta_processada'].apply(remove_greetings_and_farewells, greetings=greetings, farewells=farewells)
# df['resposta_processada'] = df['resposta_processada'].apply(remove_greetings_and_farewells, greetings=greetings, farewells=farewells)

# Verificando os registros pré-processados
#print(df.head())

## VETORIZAÇÃO E TREINAMENTO DO MODELO
# Vetorização usando o TfidfVectorizer
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(df['pergunta_processada'])

# Função para encontrar a resposta adequada com base na pergunta do usuário
def find_best_answer(user_question):
    # Pré-processamento da pergunta do usuário
    processed_question = preprocess_text(user_question)
    # Vetorização da pergunta do usuário
    user_question_vector = vectorizer.transform([processed_question])
    # Cálculo da similaridade de cosseno entre a pergunta do usuário e as perguntas do conjunto de dados
    similarity_scores = cosine_similarity(user_question_vector, question_vectors)[0]
    # Índice da pergunta com a maior similaridade
    best_question_index = similarity_scores.argmax()
    # Resposta correspondente à pergunta com maior similaridade
    best_answer = df.loc[best_question_index, 'resposta']
    return best_answer

# Testando o modelo com uma pergunta do usuário
user_question = "Como anexar documento?"
best_answer = find_best_answer(user_question)
print("Resposta:", best_answer)

## INTEGRAÇÃO EM UM CHATBOT