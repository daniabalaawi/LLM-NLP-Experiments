
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

qa = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

with open("article.txt", "r", encoding="utf-8") as f:
    article = f.read()

summary = summarizer(
    article,
    max_length=130,
    min_length=30,
    do_sample=False
)[0]["summary_text"]

print("=== Summary ===")
print(summary)
print("\n")

question = "What is the main idea of the article?"
result = qa(question=question, context=article)

print("=== Question Answering ===")
print("Question:", question)
print("Answer:", result["answer"])
