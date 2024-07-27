from flask import Flask, request
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM



app = Flask(__name__)


@app.route('/use-model', methods=['POST'])
def use_model():
    payload = request.get_json()
    print(payload['prompt'])
    tokenizer = AutoTokenizer.from_pretrained("./downloaded_models/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained("./downloaded_models/TinyLlama-1.1B-Chat-v1.0")
    # pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    pipe = pipeline("text-generation",
                    tokenizer=tokenizer,
                    model=model,
                    max_new_tokens=512,
                    do_sample=True,
                    temperature=0.9,
                    repetition_penalty=1.1)
    return pipe(payload['prompt'])[0]['generated_text']
    # messages = [
    #     {"role": "user", "content": "Who are you?"},
    # ]
    # pipe(messages)


# if __name__ == "main":
app.run()
