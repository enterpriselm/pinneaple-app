from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import sqlite3
import pandas as pd
import numpy as np
import pickle
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

app = FastAPI(title="PINNeAPPle API")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Config ===
DB_PATH = "search_results.db"
embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")

# === Utils ===
def get_connection():
    return sqlite3.connect(DB_PATH)

# === Models ===
class SemanticQuery(BaseModel):
    query: str
    top_k: int = 5

# === Endpoints ===
@app.get("/dashboard")
def dashboard():
    conn = get_connection()
    data = {}

    # Total papers & repos
    data["total_papers"] = int(pd.read_sql("SELECT COUNT(*) FROM arxiv_papers", conn).iloc[0, 0])
    data["total_repos"] = int(pd.read_sql("SELECT COUNT(*) FROM github_repos", conn).iloc[0, 0])

    # Papers por área
    papers_by_area = pd.read_sql("SELECT area, COUNT(*) as total FROM arxiv_papers GROUP BY area", conn)
    data["papers_by_area"] = papers_by_area.to_dict(orient="records")

    # Top autores de artigos
    authors = pd.read_sql("SELECT authors FROM arxiv_papers", conn)
    authors = authors.assign(authors=authors["authors"].str.split(",")).explode("authors")
    authors["authors"] = authors["authors"].str.strip()
    authors = authors.reset_index(drop=True)
    top_authors = authors["authors"].value_counts().head(5).reset_index()
    top_authors.columns = ["author", "total"]
    data["top_paper_authors"] = top_authors.to_dict(orient="records")

    # Top autores de repositórios
    top_repo_authors = pd.read_sql("""
        SELECT author, COUNT(*) as total
        FROM github_repos
        GROUP BY author
        ORDER BY total DESC
        LIMIT 5
    """, conn)
    data["top_repo_authors"] = top_repo_authors.to_dict(orient="records")

    conn.close()
    return data

@app.get("/papers")
def list_papers(area: str = '', subarea: str = '', author: str = '', title: str = ''):
    filters = []
    if area: filters.append(f"area LIKE '%{area}%'")
    if subarea: filters.append(f"subarea LIKE '%{subarea}%'")
    if author: filters.append(f"authors LIKE '%{author}%'")
    if title: filters.append(f"paper_name LIKE '%{title}%'")

    query = "SELECT * FROM arxiv_papers"
    if filters:
        query += " WHERE " + " AND ".join(filters)

    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    if 'embedding' in df.columns:
        df = df.drop(columns=['embedding'])
    return df.to_dict(orient="records")

@app.get("/papers/{paper_id}")
def paper_detail(paper_id: int):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM arxiv_papers WHERE id = {paper_id}", conn)
    conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail="Paper not found")
    if 'embedding' in df.columns:
        df = df.drop(columns=['embedding'])
    return df.iloc[0].to_dict()

@app.get("/repos")
def list_repositories(area: str = '', subarea: str = '', author: str = '', name: str = ''):
    filters = []
    if area: filters.append(f"area LIKE '%{area}%'")
    if subarea: filters.append(f"subarea LIKE '%{subarea}%'")
    if author: filters.append(f"author LIKE '%{author}%'")
    if name: filters.append(f"repo_name LIKE '%{name}%'")

    query = "SELECT * FROM github_repos"
    if filters:
        query += " WHERE " + " AND ".join(filters)

    conn = get_connection()
    df = pd.read_sql(query, conn)
    conn.close()
    if 'embedding' in df.columns:
        df = df.drop(columns=['embedding'])
    return df.to_dict(orient="records")

@app.get("/repos/{repo_id}")
def repo_detail(repo_id: int):
    conn = get_connection()
    df = pd.read_sql(f"SELECT * FROM github_repos WHERE id = {repo_id}", conn)
    conn.close()
    if df.empty:
        raise HTTPException(status_code=404, detail="Repository not found")
    if 'embedding' in df.columns:
        df = df.drop(columns=['embedding'])
    return df.iloc[0].to_dict()

@app.post("/semantic-search")
def semantic_search(query: SemanticQuery):
    conn = get_connection()
    cur = conn.cursor()

    # Embedding da query
    query_embedding = embedder.encode(query.query, normalize_embeddings=True)

    # Consultar os campos necessários no banco
    cur.execute("""
        SELECT id, paper_name, authors, abstract, publication_date, url, embedding
        FROM arxiv_papers 
        WHERE embedding IS NOT NULL
    """)

    results = []
    for row in cur.fetchall():
        id_, name, authors, abstract, pub_date, url, blob = row
        emb = pickle.loads(blob)
        score = float(np.dot(query_embedding, emb))
        results.append({
            "score": score,
            "id": id_,
            "paper_name": name,
            "authors": authors,
            "abstract": abstract,
            "publication_date": pub_date,
            "url": url,
        })

    conn.close()
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:query.top_k]

@app.post("/semantic-search-repo")
def semantic_search_repo(query: SemanticQuery):
    conn = get_connection()
    cur = conn.cursor()

    # Gerar embedding da query de busca
    query_embedding = embedder.encode(query.query, normalize_embeddings=True)

    # Buscar repositórios com embedding no banco
    cur.execute("""
        SELECT id, repo_name, author, repo_url, embedding
        FROM github_repos
        WHERE embedding IS NOT NULL
    """)

    results = []
    for row in cur.fetchall():
        id_, name, author, url, blob = row
        emb = pickle.loads(blob)
        score = float(np.dot(query_embedding, emb))
        results.append({
            "score": score,
            "id": id_,
            "repo_name": name,
            "author": author,
            "repo_url": url,
        })

    conn.close()
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:query.top_k]
