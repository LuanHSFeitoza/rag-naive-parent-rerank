from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank # Certifique-se de que 'cohere' e 'langchain-cohere' estão atualizados!

import os
import streamlit as st
import time 



# --- Configurações Globais ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["CO_API_KEY"] = st.secrets["CO_API_KEY"]

cohere_api_key = os.environ["CO_API_KEY"]

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")
llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=500)
my_directory = "./literatura_db"
pdf_file_path = "os-sertoes.pdf" # Caminho do PDF
inicio = time.time()

questions = [
    "1. Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?",
    "2. Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?",
    "3. Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?",
    "4. Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?",
    "5. Quais são os principais aspectos da crítica social e política presentes em 'Os Sertões'? Como esses aspectos refletem a visão do autor sobre o Brasil da época?"
]

# --- Definições de Funções ---

def documents_to_chunks (document_path):
    loader_pdf = PyPDFLoader(document_path, extract_images=False)
    pages = loader_pdf.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 4000,
        chunk_overlap = 20,
        length_function = len,
        add_start_index = True
    )
    chunks = text_splitter.split_documents(pages)
    return chunks

def get_or_create_vectordb(directory_db, embedding_model, chunks=None):
    # Se o diretório existe e chunks NÃO foram fornecidos (ou seja, carregando)
    if os.path.isdir(directory_db) and chunks is None:
        print(f"Carregando ChromaDB de '{directory_db}'...")
        vectordb = Chroma(
            persist_directory=directory_db,
            embedding_function=embedding_model
        )
    # Se o diretório não existe OU chunks foram fornecidos (ou seja, criando/atualizando)
    else:
        if not os.path.isdir(directory_db):
            os.makedirs(directory_db, exist_ok=True) # Garante que o diretório exista
            print(f"Diretório '{directory_db}' criado.")
        print(f"Criando/atualizando ChromaDB em '{directory_db}' com novos chunks...")
        vectordb = Chroma.from_documents(
            chunks,
            embedding=embedding_model,
            persist_directory=directory_db
        )
        vectordb.persist() # Persiste os dados em disco
        print(f"ChromaDB criado/atualizado e salvo em '{directory_db}'.")
    return vectordb

def get_rerank_retriever(vectordb, cohere_api_key):
    naive_retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    rerank = CohereRerank(model="rerank-v3.5", top_n=3, cohere_api_key=cohere_api_key)

    compressor_retriever = ContextualCompressionRetriever(
        base_compressor=rerank,
        base_retriever=naive_retriever
    )
    return compressor_retriever

def ask(question, retriever, llm_model): # Alterei 'llm' para 'llm_model' para evitar conflito de nome
    TEMPLATE = """
        Você é uma especialista em literatura contemporânea, focada em ajudar leitores curiosos a esclarecer dúvidas sobre esses livros.
        Query: {question}

        Contexto: {context}
    """
    rag_prompt = ChatPromptTemplate.from_template(TEMPLATE)

    setup_retrieval = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    output_parser = StrOutputParser()

    chain = setup_retrieval | rag_prompt | llm_model | output_parser

    answer = chain.invoke(question)
    return answer

# --- Fluxo Principal ---

# Verifica se o diretório do DB existe
if not os.path.isdir(my_directory):
    print(f"O diretório '{my_directory}' não existe. Processando PDF e criando DB...")
    
    # 1. Carrega e divide o PDF em chunks
    chunks = documents_to_chunks(pdf_file_path)
    
    # 2. Cria e persiste o ChromaDB com os chunks
    # Esta função agora gerencia a criação e persistência
    vectordb_instance = get_or_create_vectordb(my_directory, embedding_model, chunks=chunks)
    
    # 3. Obtém o retriever (com rerank) do DB recém-criado
    retriever = get_rerank_retriever(vectordb_instance, cohere_api_key)
    
    # 4. Faz a pergunta
    for q in questions:
        answer = ask(q, retriever, llm)
        print(f"Pergunta: {q}\nResposta: {answer}\n{'-'*80}")

    fim = time.time()
    tempo_execucao = fim - inicio
    print(f"Tempo total de execução: {tempo_execucao:.2f} segundos")

else:
    print(f"O diretório '{my_directory}' já existe. Carregando DB e retriever...")
    
    # 1. Carrega o ChromaDB existente do disco (não precisa re-chunkar o PDF)
    # Passamos 'chunks=None' para indicar que é um carregamento, não uma criação
    vectordb_instance = get_or_create_vectordb(my_directory, embedding_model, chunks=None)

    # 2. Obtém o retriever (com rerank) do DB carregado
    retriever = get_rerank_retriever(vectordb_instance, cohere_api_key)
    
    # 3. Faz a pergunta
    for q in questions:
        answer = ask(q, retriever, llm)
        print(f"Pergunta: {q}\nResposta: {answer}\n{'-'*80}")

    fim = time.time()
    tempo_execucao = fim - inicio
    print(f"Tempo total de execução: {tempo_execucao:.2f} segundos")

