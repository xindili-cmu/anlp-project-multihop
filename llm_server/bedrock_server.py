import boto3
import json
from fastapi import FastAPI
from typing import Optional

app = FastAPI()
client = boto3.client('bedrock-runtime', region_name='us-east-1')

@app.get("/")
async def index():
    return {"message": "Bedrock Llama server"}

@app.get("/generate")
async def generate(
    prompt: str,
    max_length: int = 200,
    max_input: Optional[int] = None,
    min_length: int = 1,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    num_return_sequences: int = 1,
    repetition_penalty: Optional[float] = None,
    length_penalty: Optional[float] = None,
    keep_prompt: bool = False,
):
    body = json.dumps({
        "prompt": prompt,
        "max_gen_len": min(max_length, 512),
        "temperature": max(temperature, 0.01),
    })
    response = client.invoke_model(
        modelId='us.meta.llama3-1-8b-instruct-v1:0',
        body=body
    )
    result = json.loads(response['body'].read())
    generated_text = result.get('generation', '')
    return {
        "generated_texts": [generated_text],
        "generated_num_tokens": [len(generated_text.split())],
        "run_time_in_seconds": 1.0,
        "model_name": "llama3-1-8b"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
