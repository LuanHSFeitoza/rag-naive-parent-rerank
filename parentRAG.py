from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

import os
import streamlit as st
import time

# --- Configurações Globais ---
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

def create_and_split_documents(document_path):
    loader_pdf = PyPDFLoader(document_path, extract_images=False)
    pages = loader_pdf.load_and_split()

    # Splitters para ParentDocumentRetriever
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=200)
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True
    )
    return pages, child_splitter, parent_splitter

def get_or_create_parent_retriever(directory_db, embedding_model, pages=None, child_splitter=None, parent_splitter=None):

    store = InMemoryStore()

    # Inicializa o vetorstore persistente (ChromaDB)
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=directory_db
    )

    # Inicializa o ParentDocumentRetriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter, # Necessário para splitar os filhos ao adicionar
        parent_splitter=parent_splitter, # Necessário para splitar os pais ao adicionar
        search_kwargs={"k": 3}
    )

    # Se 'pages' for fornecido, significa que estamos criando/populando o DB
    if pages is not None:
        print(f"Adicionando documentos ao ChromaDB em '{directory_db}'...")
        # Adiciona documentos ao retriever, o que popula o vectorstore e o docstore.
        retriever.add_documents(pages)
        # Persiste o vectorstore (os embeddings dos chunks)
        vectorstore.persist()
        print(f"Documentos adicionados e ChromaDB persistido em '{directory_db}'.")
    else:
        # Se 'pages' não for fornecido, estamos apenas carregando/reutilizando
        print(f"Carregando ChromaDB de '{directory_db}'.")

    return retriever

def ask_question(question, retriever, llm_model):
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
# Garante que o diretório para o ChromaDB exista
if not os.path.isdir(my_directory):
    print(f"O diretório '{my_directory}' não existe. Criando...")
    os.makedirs(my_directory, exist_ok=True) # Cria o diretório se não existir

    # Processa o PDF para obter páginas e splitters
    pages, child_splitter, parent_splitter = create_and_split_documents(pdf_file_path)

    # Cria e popula o retriever 
    docs_retriever = get_or_create_parent_retriever(
        my_directory, embedding_model, pages, child_splitter, parent_splitter
    )

else:
    print(f"O diretório '{my_directory}' já existe. Carregando retriever...")
 
    child_splitter_load = RecursiveCharacterTextSplitter(chunk_size=200)
    parent_splitter_load = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=20) # Simples para carregamento

    docs_retriever = get_or_create_parent_retriever(
        my_directory, embedding_model, 
        child_splitter=child_splitter_load,
        parent_splitter=parent_splitter_load
    )


# ask_question(user_question, docs_retriever, llm)
for q in questions:
    answer = ask_question(q, docs_retriever, llm)
    print(f"Pergunta: {q}\nResposta: {answer}\n{'-'*80}")

fim = time.time()
tempo_execucao = fim - inicio
print(f"Tempo total de execução: {tempo_execucao:.2f} segundos")