import os
import numpy as np
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from auth import Config

x = Config()
x.authenticate()

def load_docs(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    return loader.load()

def expand_query_with_prf(query, retriever, top_n=5, expansion_terms=5):
    docs = retriever.get_relevant_documents(query)[:top_n]
    if not docs:
        return query
    texts = [doc.page_content for doc in docs]
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = np.array(vectorizer.get_feature_names_out())
    avg_scores = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    top_indices = avg_scores.argsort()[::-1][:expansion_terms]
    expansion_words = feature_names[top_indices]
    return query + " " + " ".join(expansion_words)

def deduplicate_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            unique.append(doc)
    return unique

def initialize_pipeline(file_path):
    docs = load_docs(file_path)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        chunk_size=500,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)

    dense = vectorstore.as_retriever(search_kwargs={"k": 5})
    sparse = BM25Retriever.from_documents(chunks)
    sparse.k = 5
    hybrid = EnsembleRetriever(retrievers=[dense, sparse], weights=[0.5, 0.5])

    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0.2)

    return hybrid, sparse, llm, reranker

prompt_template = ChatPromptTemplate.from_template("""
You are a precise and factual assistant.
Answer the question based ONLY on the provided context.

Context:
{context}

Question:
{question}

Answer:
""")

def rerank(query, docs, reranker):
    pairs = [[query, d.page_content] for d in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked]

def answer_question_pipeline(query, hybrid, sparse, llm, reranker):
    expanded_query = expand_query_with_prf(query, sparse)
    results = hybrid.get_relevant_documents(expanded_query)
    results = deduplicate_docs(results)
    top_docs = rerank(query, results, reranker)
    context = "\n\n".join(doc.page_content for doc in top_docs[:5])
    prompt = prompt_template.format(context=context, question=query)
    return llm.invoke(prompt).content
