from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "unitary/toxic-bert"

print("⏳ Downloading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save locally into "bert/" folder
tokenizer.save_pretrained("app/bert")
model.save_pretrained("app/bert")
print("✅ Model and tokenizer saved in 'bert/'")
