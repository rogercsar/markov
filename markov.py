import markovify
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
import json

# Inicializa o analisador de sentimentos e modelo de embeddings
sa = SentimentIntensityAnalyzer()
model = SentenceTransformer('all-MiniLM-L6-v2')

# Carrega o arquivo de respostas e cria a lista
with open('corpus.txt', 'r', encoding='utf-8') as file:
    respostas = [line.strip() for line in file.readlines()]

# Carrega os corpora para Markov
with open('corpus_positivo.txt', 'r', encoding='utf-8') as file:
    texto_positivo = file.read()
with open('corpus_negativo.txt', 'r', encoding='utf-8') as file:
    texto_negativo = file.read()
with open('corpus_neutro.txt', 'r', encoding='utf-8') as file:
    texto_neutro = file.read()

# Cria modelos Markov para cada sentimento
model_positivo = markovify.Text(texto_positivo, state_size=2)
model_negativo = markovify.Text(texto_negativo, state_size=2)
model_neutro = markovify.Text(texto_neutro, state_size=2)

# Inicializa o tradutor
tradutor = GoogleTranslator(source='pt', target='en')

# Carrega memória do contexto
def carregar_memoria(arquivo='memoria.json'):
    try:
        with open(arquivo, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

# Salva a memória do contexto
def salvar_memoria(memoria, arquivo='memoria.json'):
    with open(arquivo, 'w') as f:
        json.dump(memoria, f, indent=4)

# Função para calcular a similaridade
def calcular_similaridade(frase_usuario, respostas):
    maior_score = -1
    melhor_resposta = respostas[0]  # Default para a primeira resposta caso nada seja encontrado
    
    embeddings_usuario = model.encode(frase_usuario, convert_to_tensor=True)
    
    for resposta in respostas:
        embeddings_resposta = model.encode(resposta, convert_to_tensor=True)
        similaridade_embeddings = util.pytorch_cos_sim(embeddings_usuario, embeddings_resposta).item()
        
        if similaridade_embeddings > maior_score:
            maior_score = similaridade_embeddings
            melhor_resposta = resposta
            
    return melhor_resposta

# Função para atualizar a memória
def atualizar_memoria(memoria, usuario, frase_gerada, resposta_final):
    if usuario not in memoria:
        memoria[usuario] = []
    memoria[usuario].append({'entrada': frase_gerada, 'resposta': resposta_final})

# Carrega a memória inicial
memoria = carregar_memoria()

# Loop contínuo para manter o script rodando
while True:
    inUser = input('Roger: ')
    
    if inUser.lower() in ['sair', 'exit', 'quit']:
        print('Nia: Encerrando o chat.')
        salvar_memoria(memoria)
        break

    # Traduz a entrada do usuário
    frase = tradutor.translate(inUser)

    try:
        # Análise de sentimento
        sentimento = sa.polarity_scores(frase)

        # Seleciona o modelo Markov apropriado
        if sentimento['compound'] >= 0.5:
            frase_gerada = model_positivo.make_short_sentence(140) or "Posso ajudar com algo mais positivo?"
        elif sentimento['compound'] <= -0.5:
            frase_gerada = model_negativo.make_short_sentence(140) or "Entendo que é difícil, mas estou aqui para ajudar."
        else:
            frase_gerada = model_neutro.make_short_sentence(140) or "Entendi, me conte mais."

        # Seleciona a melhor resposta baseada em embeddings
        resposta_final = calcular_similaridade(frase_gerada, respostas)

        # Atualiza a memória com a nova interação
        atualizar_memoria(memoria, 'Roger', frase_gerada, resposta_final)
        print(f'Nia: {resposta_final}')

    except Exception as e:
        print('Nia: Erro ao gerar texto:', str(e))
