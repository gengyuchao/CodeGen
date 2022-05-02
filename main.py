from fastapi import FastAPI, Request
from onnx_model import predict

app = FastAPI()

@app.get('/')
async def get_root():
    return {'hello': 'world'}

@app.post('/api/codegen')
async def post_api_codegen(request: Request):
    """
    curl localhost:8000/api/codegen -H 'Content-Type: application/json' -d '{"inputs": "def avg(arr):\n\t"}'
    """
    body = await request.json()
    if not isinstance(body, dict):
        return dict(error='Invalid request body')
    inputs = body.get('inputs')
    if not isinstance(inputs, str):
        return dict(error='Invalid request body inputs')
    context = inputs
    print('context', context)
    results = predict(context)
    print('results', type(results), results)
    return [
        {'generated_text': results }
    ]

print('start')
