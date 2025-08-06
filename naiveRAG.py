from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings 
from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.question_answering  import load_qa_chain

import os
import streamlit as st
import time 

# --- Configurações ---
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model="gpt-3.5-turbo", max_tokens=200)
my_directory = "./literatura_db" 
pdf_file_path = "os-sertoes.pdf"
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

def create_db (directory_db, chunks, embedding_model):
    db = Chroma.from_documents(
        chunks,
        embedding=embedding_model,
        persist_directory=directory_db
    )
    db.persist()
    print(f"ChromaDB criado e dados salvos em '{directory_db}'.") 
    return db 

def get_documents_db (directory_db, embedding_model, llm_model): 
    vectorDB = Chroma(
        persist_directory=directory_db,
        embedding_function=embedding_model
    )
    retriever = vectorDB.as_retriever(search_kwargs={"k":3})
    chain = load_qa_chain(llm_model, chain_type="stuff") 
    return retriever, chain

def ask(question, directory_db, embedding_model, llm_model): 
    docs_retriever, chain = get_documents_db(directory_db, embedding_model, llm_model) 
    context = docs_retriever.get_relevant_documents(question)
    answer = (
        chain(
            {
                "input_documents": context,
                "question": question
            },
            return_only_outputs=True
        )
    ) ['output_text']
    return answer

# --- Ação do Usuário e Fluxo Principal ---

if not os.path.isdir(my_directory): #
    print(f"O diretório '{my_directory}' não existe. Processando PDF e criando DB...")
    os.makedirs(my_directory, exist_ok=True) # Crie o diretório se não existir

    chunks = documents_to_chunks(pdf_file_path)
    
    
    db_instance = create_db(my_directory, chunks, embedding_model) 

    for q in questions:
        answer = ask(q, my_directory, embedding_model, llm)
        print(f"Pergunta: {q}\nResposta: {answer}\n{'-'*80}")


else:
    print(f"O diretório '{my_directory}' já existe. Assumindo DB já populado.")

    for q in questions:
        answer = ask(q, my_directory, embedding_model, llm)
        print(f"Pergunta: {q}\nResposta: {answer}\n{'-'*80}")


fim = time.time()
tempo_execucao = fim - inicio
print(f"Tempo total de execução: {tempo_execucao:.2f} segundos")