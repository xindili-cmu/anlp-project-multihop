import json
from fastapi import FastAPI
from rank_bm25 import BM25Okapi
from typing import Optional
from pydantic import BaseModel

app = FastAPI()

print("Loading HotpotQA and building BM25 index...")
with open("downloads/hotpotqa/dev.json") as f:
    data = json.load(f)

corpus = []
corpus_meta = []
for item in data:
    for title, sentences in zip(item['context']['title'], item['context']['sentences']):
        text = ' '.join(sentences)
        corpus.append(text)
        corpus_meta.append({"title": title, "paragraph_text": text, "corpus_name": "hotpotqa", "id": str(len(corpus_meta))})

tokenized_corpus = [doc.split() for doc in corpus]
bm25 = BM25Okapi(tokenized_corpus)
print(f"Index built: {len(corpus)} paragraphs")

class RetrieveRequest(BaseModel):
    query_text: str = ""
    max_hits_count: int = 3
    corpus_name: str = "hotpotqa"
    query_id: Optional[str] = None
    retrieval_method: Optional[str] = None

@app.get("/")
async def index():
    return {"message": "HotpotQA BM25 retriever"}

@app.post("/retrieve")
async def retrieve(request: RetrieveRequest):
    if not request.query_text.strip():
        return {"retrieval": []}
    tokenized_query = request.query_text.split()
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:request.max_hits_count]
    results = [corpus_meta[i] for i in top_indices]
    return {"retrieval": results}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
