import os
import re
import math
from collections import defaultdict, Counter
from typing import Literal

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

nltk.download("stopwords", quiet=True)
STOP_WORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")

def preprocess_text(text: str) -> list[str]:
    raw_tokens = TOKEN_PATTERN.findall(text.lower())
    return [STEMMER.stem(t) for t in raw_tokens if t not in STOP_WORDS and len(t) > 1]

class InvertedIndex:
    def __init__(self):
        self.doc_ids = []
        self.documents = {}
        self.inverted_index = defaultdict(dict)
        self.doc_freq = {}
        self.doc_lengths = {}
        self.vocabulary = {}
        self.num_docs = 0
        self.avg_doc_length = 0.0

    def build(self, df_corpus: pd.DataFrame):
        total_tokens = 0
        for row in df_corpus.itertuples(index=False):
            doc_id = str(row[0])
            title = str(row.title)
            text = str(row.text)
            self.doc_ids.append(doc_id)
            self.documents[doc_id] = {"title": title, "text": text}
            tokens = preprocess_text(f"{title} {text}".strip())
            self.doc_lengths[doc_id] = len(tokens)
            total_tokens += len(tokens)
            for term, tf in Counter(tokens).items():
                self.inverted_index[term][doc_id] = tf
        self.num_docs = len(self.doc_ids)
        self.avg_doc_length = total_tokens / self.num_docs if self.num_docs else 0.0
        for tid, term in enumerate(sorted(self.inverted_index.keys())):
            self.vocabulary[term] = tid
            self.doc_freq[term] = len(self.inverted_index[term])

    def get_snippet(self, doc_id: str, max_len: int = 240) -> str:
        text = self.documents[doc_id]["text"].strip()
        return text if len(text) <= max_len else text[:max_len].rsplit(" ", 1)[0] + "..."

class RetrievalEngine:
    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b
        N = index.num_docs
        self.tfidf_idf = {t: math.log10(N / df) for t, df in index.doc_freq.items()}
        self.bm25_idf = {t: math.log(1.0 + (N - df + 0.5) / (df + 0.5)) for t, df in index.doc_freq.items()}
        doc_sq = defaultdict(float)
        for term, postings in index.inverted_index.items():
            idf = self.tfidf_idf[term]
            for doc_id, tf in postings.items():
                w = (1.0 + math.log10(tf)) * idf
                doc_sq[doc_id] += w * w
        self.doc_norms = {d: math.sqrt(sq) for d, sq in doc_sq.items()}

    def search_tfidf(self, query: str, top_k: int = 10):
        q_counts = Counter(preprocess_text(query))
        q_weights, q_sq = {}, 0.0
        for t, q_tf in q_counts.items():
            if t in self.index.inverted_index:
                w = (1.0 + math.log10(q_tf)) * self.tfidf_idf[t]
                q_weights[t] = w
                q_sq += w * w
        q_norm = math.sqrt(q_sq)
        if q_norm == 0.0:
            return []
        scores = defaultdict(float)
        for t, w_tq in q_weights.items():
            idf = self.tfidf_idf[t]
            for doc_id, tf in self.index.inverted_index[t].items():
                scores[doc_id] += w_tq * ((1.0 + math.log10(tf)) * idf)
        ranked = [(d, dot / (q_norm * self.doc_norms[d])) for d, dot in scores.items() if self.doc_norms.get(d, 0) > 0]
        ranked.sort(key=lambda x: (-x[1], x[0]))
        return self._format(ranked[:top_k])

    def search_bm25(self, query: str, top_k: int = 10):
        q_tokens = preprocess_text(query)
        if not q_tokens:
            return []
        scores = defaultdict(float)
        avgdl = self.index.avg_doc_length
        for t in q_tokens:
            if t not in self.index.inverted_index:
                continue
            idf = self.bm25_idf[t]
            for doc_id, tf in self.index.inverted_index[t].items():
                dl = self.index.doc_lengths[doc_id]
                scores[doc_id] += idf * ((tf * (self.k1 + 1.0)) / (tf + self.k1 * (1.0 - self.b + self.b * (dl / avgdl))))
        ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        return self._format(ranked[:top_k])

    def _format(self, pairs):
        return [
            {
                "rank": i,
                "document_id": d,
                "score": round(float(s), 4),
                "title": self.index.documents[d]["title"],
                "snippet": self.index.get_snippet(d),
            }
            for i, (d, s) in enumerate(pairs, start=1)
        ]

    def search(self, query: str, model: str = "bm25", top_k: int = 10):
        m = model.strip().lower()
        if m == "bm25":
            return {"model": "bm25", "results": self.search_bm25(query, top_k)}
        if m == "tfidf":
            return {"model": "tfidf", "results": self.search_tfidf(query, top_k)}
        raise ValueError("Modelo inválido")

df_corpus = pd.read_parquet("corpus-00000-of-00001.parquet")

df_corpus["_id"] = df_corpus["_id"].astype(str)
df_corpus["title"] = df_corpus["title"].fillna("").astype(str)
df_corpus["text"] = df_corpus["text"].fillna("").astype(str)

index = InvertedIndex()
index.build(df_corpus)
engine = RetrievalEngine(index)

app = FastAPI(title="ICCD753 IR Service - NFCorpus")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    model: Literal["bm25", "tfidf"] = "bm25"
    top_k: int = Field(10, ge=1, le=100)

@app.get("/health")
def health():
    return {"status": "ok", "corpus": "BeIR/nfcorpus", "indexed_documents": index.num_docs}

@app.post("/search")
def search(req: SearchRequest):
    return engine.search(req.query, req.model, req.top_k)
