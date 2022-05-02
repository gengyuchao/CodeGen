from fastapi import FastAPI, Request
from model import print_time, tokenizer, model, predict

app = FastAPI()

@app.get('/')
async def get_root():
    return {'hello': 'world'}

@app.post('/api/codegen')
async def post_api_codegen(request: Request):
    body = await request.json()
    if not isinstance(body, dict):
        return dict(error='Invalid request body')
    inputs = body.get('inputs')
    if not isinstance(inputs, str):
        return dict(error='Invalid request body inputs')
    context = inputs
    print('context', context)
    results = predict(tokenizer, model, context, max_length=128)
    print('results', type(results), results)
    return [
        {'generated_text': x} for x in results.split('\n') if len(x) > 0
    ]

print('start')
