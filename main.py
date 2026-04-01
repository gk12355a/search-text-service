from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import numpy as np
import mysql.connector

import unicodedata
from thefuzz import fuzz

app = FastAPI(title="Text Semantic Search API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def remove_accents(s: str) -> str:
    """Hàm loại bỏ dấu Tiếng Việt (Ví dụ: 'Máy chiếu' -> 'may chieu')"""
    if not s: return ""
    s = unicodedata.normalize('NFD', str(s))
    return ''.join(c for c in s if unicodedata.category(c) != 'Mn').lower()

print("Loading model 'paraphrase-multilingual-MiniLM-L12-v2' (Tối ưu Tiếng Việt)...")
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
print("Model loaded successfully.")

import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', '192.168.23.11'),
    'port': int(os.getenv('DB_PORT', 6033)),
    'user': os.getenv('DB_USER', 'dtkien5'),
    'password': os.getenv('DB_PASSWORD', 'Kien2005@gym'),
    'database': os.getenv('DB_NAME', 'db2026_03_25'),
    'autocommit': True
}

def get_data_from_db(item_type: str):
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)
    try:
        if item_type == "users":
            cursor.execute("SELECT id, full_name, username FROM users")
            rows = cursor.fetchall()
            return [{"id": r["id"], "text": f"{r['full_name']} {r['username']}"} for r in rows]
        elif item_type == "rooms":
            cursor.execute("SELECT id, name, location, building_name FROM rooms")
            rows = cursor.fetchall()
            return [{"id": r["id"], "text": f"{r['name']} {r['location'] or ''} {r['building_name'] or ''}"} for r in rows]
        elif item_type == "devices":
            cursor.execute("SELECT id, name, description FROM devices")
            rows = cursor.fetchall()
            return [{"id": r["id"], "text": f"{r['name']} {r['description'] or ''}"} for r in rows]
        else:
            return []
    finally:
        cursor.close()
        conn.close()

import math
from collections import Counter

class BM25:
    """Thuật toán xếp hạng văn bản tiêu chuẩn nghiệp vụ (Best Matching 25)"""
    def __init__(self, corpus):
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = []
        self.idf = {}
        self.doc_len = []
        
        df = Counter()
        for doc in corpus:
            self.doc_len.append(len(doc))
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                df[word] += 1
                
        for word, freq in df.items():
            self.idf[word] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))

    def get_scores(self, query):
        scores = np.zeros(self.corpus_size)
        if self.corpus_size == 0 or not query: return scores
        k1 = 1.5
        b = 0.75
        for i in range(self.corpus_size):
            score = 0.0
            doc_len = self.doc_len[i]
            freqs = self.doc_freqs[i]
            for word in query:
                if word not in freqs:
                    continue
                freq = freqs[word]
                numerator = self.idf[word] * freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * doc_len / self.avgdl)
                score += (numerator / denominator)
            scores[i] = score
        return scores

def generate_ngrams(text, n=3):
    """Tạo chuỗi n-grams giúp tìm kiếm chống được gõ thừa/thiếu dấu cách và sai chính tả"""
    text = f" {text} "
    return [text[i:i+n] for i in range(len(text) - n + 1)]

def min_max_norm(scores):
    """Chuẩn hóa mảng điểm về thang 0.0 -> 1.0"""
    if len(scores) == 0: return scores
    min_v, max_v = np.min(scores), np.max(scores)
    if max_v - min_v == 0:
        return np.zeros_like(scores) if max_v == 0 else np.ones_like(scores)
    return (scores - min_v) / (max_v - min_v)

class SearchRequest(BaseModel):
    query: str
    type: str

@app.post("/api/semantic-search")
async def semantic_search(req: SearchRequest):
    if not req.query.strip():
        return {"success": True, "results": []}
    
    try:
        items = get_data_from_db(req.type)
        if not items:
            return {"success": True, "results": []}
            
        docs = [item["text"] for item in items]
        
        # 1. Điểm Ngữ Nghĩa (Semantic Cosine Similarity) -> Normalized
        doc_embeddings = model.encode(docs, normalize_embeddings=True)
        query_embedding = model.encode([req.query], normalize_embeddings=True)
        semantic_scores = min_max_norm(np.dot(doc_embeddings, query_embedding.T).flatten())
        
        # 2. Xóa dấu Tiếng Việt chuẩn bị cho Lexical BM25
        query_norm = remove_accents(req.query.strip())
        doc_norms = [remove_accents(doc) for doc in docs]
        
        # 3. Kỹ thuật Keyword Matching toàn vẹn (BM25 trên từ)
        q_words = query_norm.split()
        d_words = [doc.split() for doc in doc_norms]
        bm25_word = BM25(d_words)
        word_scores = min_max_norm(bm25_word.get_scores(q_words))
        
        # 4. Kỹ thuật N-gram BM25 xử lý triệt để lỗi chính tả, space-collision chặt chẽ
        # Việc tách chuỗi thành các cụm 3 ký tự (Tri-grams) giúp mô hình toán học cân bằng 
        # tỉ lệ overlap thay vì phụ thuộc hoàn toàn vào heuristic +2.5 dễ gây false positive.
        q_ngrams = generate_ngrams(query_norm, 3)
        d_ngrams = [generate_ngrams(doc, 3) for doc in doc_norms]
        bm25_ngram = BM25(d_ngrams)
        ngram_scores = min_max_norm(bm25_ngram.get_scores(q_ngrams))
        
        # 5. Tổng hợp điểm với Tỉ Trọng (Weights Optimization)
        # Giờ đây thuật toán thuần sức mạnh Toán học + AI:
        # Ngữ nghĩa chiếm 40% (Giúp tìm kiếm ý tưởng rộng)
        # BM25 Keyword chiếm 30% (Đảm bảo độ chính xác trên Tên riêng)
        # N-grams chiếm 30% (Chống lỗi typo, lỗi dấu cách tinh vi)
        final_scores = 0.40 * semantic_scores + 0.30 * word_scores + 0.30 * ngram_scores
        
        results = [{"id": items[i]["id"], "score": float(final_scores[i])} for i in range(len(docs))]
        
        # Sắp xếp kết quả trả về cho UI
        results.sort(key=lambda x: x["score"], reverse=True)
        return {"success": True, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8006, reload=True)
