import json
import os
from fastapi import FastAPI
from typing import Optional

app = FastAPI()

# 加载数据
with open("downloads/hotpotqa/dev.json") as f:
    data = json.load(f)

# 建立问题到context的索引
qid_to_context = {}
for item in data:
    qid_to_context[item["id"]] = {
        "titles": item["context"]["title"],
        "sentences": item["context"]["sentences"]
    }

# 全局当前问题ID
current_qid = None

@app.get("/")
async def index():
    return {"message": "HotpotQA retriever server"}

@app.post("/retrieve")
async def retrieve(query_text: str = "", 
                   max_hits_count: int = 3,
                   corpus_name: str = "hotpotqa",
                   query_id: Optional[str] = None):
    global current_qid
    
    if query_id:
        current_qid = query_id
    
    if current_qid and current_qid in qid_to_context:
        ctx = qid_to_context[current_qid]
        results = []
        for title, sents in zip(ctx["titles"], ctx["sentences"]):
            paragraph_text = " ".join(sents)
            results.append({
                "title": title,
                "paragraph_text": paragraph_text,
                "score": 1.0
            })
        return {"retrieval": results[:max_hits_count]}
    
    return {"retrieval": []}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
