
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    pipeline
)

texts = [
    "Dania walked through the quiet streets of Amman, realizing the city held more stories than she ever imagined."
    "Every night, she returned to her small desk, where the glow of her laptop became the only light in the room."
    "The moment Dania opened her old notebook, memories of her first project came rushing back like a forgotten dream."
    "She had always believed that every dataset carried a secret, waiting for someone patient enough to uncover it."
    "One cold evening, a sudden idea struck her—an idea that felt strangely alive, like it had been waiting for her."
    "As the rain tapped softly on her window, Dania typed the first line of code that would change everything."
    "She paused, realizing that she wasn’t just building a model; she was building a version of herself she had never met."
    "Some nights, the silence felt heavy, but Dania learned to find comfort in the rhythm of her thoughts."
    "The old library at the University of Jordan became her second home, where stories of past students whispered through the shelves."
    "Just when she felt lost, a single line in her code finally worked, lighting up her face with quiet triumph."
    "With every experiment, Dania felt the world around her shrink, until only the story she was writing truly existed."
    "She once feared failure, but now she saw it as a character in her story—a character that pushed her forward."
    "In the stillness of the early morning, Dania realized that dreams grow louder when the world is quiet."
    "She knew that the journey would be long, but every chapter begins with a single brave sentence."
    "Dania closed her eyes for a moment, imagining the future she was slowly constructing with each key she pressed."

]

model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

generator_before = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("=== BEFORE FINE-TUNING ===")
before_output = generator_before(
    "In the University of Jordan,",
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95
)[0]["generated_text"]
print(before_output)
print("\n")

tokenized = [
    tokenizer(
        t,
        truncation=True,
        padding="max_length",
        max_length=64
    )
    for t in texts
]

dataset = Dataset.from_dict({
    "input_ids": [item["input_ids"] for item in tokenized],
    "attention_mask": [item["attention_mask"] for item in tokenized],
})

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

training_args = TrainingArguments(
    output_dir="./distilgpt2-finetuned",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    logging_steps=5,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

print("=== TRAINING... ===")
trainer.train()
print("=== TRAINING DONE ===\n")

generator_after = pipeline("text-generation", model=model, tokenizer=tokenizer)

print("=== AFTER FINE-TUNING ===")
after_output = generator_after(
    "In the University of Jordan,",
    max_new_tokens=50,
    do_sample=True,
    top_k=50,
    top_p=0.95
)[0]["generated_text"]
print(after_output)
print()
