from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("text2tracks_model")
model = T5ForConditionalGeneration.from_pretrained("text2tracks_model")

def recommend(prompt, k=3):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(input_ids, num_return_sequences=k, num_beams=max(k, 3))
    return [tokenizer.decode(o, skip_special_tokens=True) for o in outputs]

# Test prompts
for prompt in ["lofi beats", "jazz vibes", "pop anthems", "classic rock"]:
    print(f"Prompt: {prompt}")
    print("Recommendations:", recommend(prompt))