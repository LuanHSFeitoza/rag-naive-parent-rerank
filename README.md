# 📚 RAG - Naive | Parent | Rerank

Este repositório apresenta um estudo prático sobre três abordagens de **Retrieval-Augmented Generation (RAG)** usando **LangChain** e diferentes estratégias de recuperação de contexto a partir de um documento PDF — no caso, *Os Sertões* de Euclides da Cunha.

O objetivo é comparar **precisão, relevância e contexto** das respostas em cada método.

---

## 🚀 Abordagens testadas

| Método   | Descrição | Vantagens | Limitações |
|----------|-----------|-----------|------------|
| **Naive** 🥇 | Recupera os `k` documentos mais similares usando embeddings e entrega direto ao LLM. | Simples de implementar. | Pode trazer contexto irrelevante. |
| **Parent** 🧩 | Recupera *chunks-pai* maiores, preservando mais contexto do documento original. | Contexto mais rico, menos fragmentado. | Mais custo de processamento. |
| **Rerank** 🎯 | Usa *naive retrieval* e reclassifica resultados com **Cohere Rerank v3.5**. | Alta precisão nas respostas. | Depende de API extra (Cohere). |

---

## 🛠️ Tecnologias usadas

- [LangChain](https://www.langchain.com/)
- [OpenAI API](https://platform.openai.com/)
- [Cohere Rerank API](https://cohere.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Streamlit](https://streamlit.io/)

---

## 📂 Estrutura do projeto

```plaintext
rag-naive-parent-rerank/
│── naive.py     # Implementação básica
│── parent.py    # Recuperação com ParentDocumentRetriever
│── rerank.py    # Recuperação com Cohere Rerank
│── literatura_db/  # Base vetorial persistida
│── os-sertoes.pdf  # Documento de teste

## ▶️ Como executar

1. Clone o repositório
  bash
  git clone https://github.com/seuusuario/rag-naive-parent-rerank.git
  cd rag-naive-parent-rerank

2. Configure as chaves de API
  OPENAI_API_KEY="sua_chave_openai"
  CO_API_KEY="sua_chave_cohere"

3. Execute um dos Scripts
  streamlit run naive.py
  # ou
  streamlit run parent.py
  # ou
  streamlit run rerank.py

## Exemplos de pergunta

| **O modelo está com perguntas fixas que podem ser alteradas diretamente no script** 🥇 |

questions = [
    "1. Qual é a visão de Euclides da Cunha sobre o ambiente natural do sertão nordestino e como ele influencia a vida dos habitantes?",
    "2. Quais são as principais características da população sertaneja descritas por Euclides da Cunha? Como ele relaciona essas características com o ambiente em que vivem?",
    "3. Qual foi o contexto histórico e político que levou à Guerra de Canudos, segundo Euclides da Cunha?",
    "4. Como Euclides da Cunha descreve a figura de Antônio Conselheiro e seu papel na Guerra de Canudos?",
    "5. Quais são os principais aspectos da crítica social e política presentes em 'Os Sertões'? Como esses aspectos refletem a visão do autor sobre o Brasil da época?"
]

## 📌 Resultados observados

| **Naive** 🥇 | rápido, mas com respostas às vezes dispersas.
| **Parent** 🧩 | mais detalhado e contextualizado.
| **Rerank** 🎯 | respostas mais focadas e precisas.





