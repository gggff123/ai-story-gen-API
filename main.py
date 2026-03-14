from fastapi import FastAPI
from transformers import AutoTokenizer, AutoModelForCausalLM


model_id = "roneneldan/TinyStories-1M"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model     = AutoModelForCausalLM.from_pretrained(model_id)

app=FastAPI()
@app.get("/")
def home():
    return """TinyStories Text Generation API

This API generates short stories using the model:
roneneldan/TinyStories-1M

Available routes:

/            -> API information
/model       -> Returns model name
/gen?inp=TEXT -> Generate text from a prompt

Example request:
/gen?inp=Once upon a time

Example response:
Once upon a time there was a small dragon who lived in a quiet forest...

Documentation:
/docs -> Interactive API documentation
/redoc -> Alternative documentation page"""
#3 route home,gen,model
@app.get("/gen")
def generate(inp:str):
    prompt   = inp
    inputs  = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
    **inputs,
    max_new_tokens=200,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id,
    )
    return {"generated_text":tokenizer.decode(outputs[0], skip_special_tokens=True)}
@app.get("/model")
def model_info():
    return {"name":"TinyStories","created by":"roneneldan","params":"1m"}
