from transformers import GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, Trainer
from transformers.data import data_collator

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

model.train()
model.resize_token_embeddings(len(tokenizer))

def generate_response(question):
    input_ids = tokenizer.encode(question, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response


print("Let's chat! (type 'quit' to exit)")
while True:
    user_question = input("Digite sua pergunta: ")
    response = generate_response(user_question)
    print("Resposta: ", response)