# How to Run Each Script

Below are instructions for running each of the three Python scripts included in this project.

---

1- text_generation.py
----------------------

This script generates Arabic text using a pretrained GPT-2 model.

### Requirement:
None.

### Run:
python text_generation.py

---

2- summarization_qa.py

This script summarizes an article and answers a question based on its content.

### Requirement:

A file named article.txt must be in the same directory.

### Run:
python summarization_qa.py

---

2- fine_tuning.py

This script fine-tunes the DistilGPT-2 model using a small custom dataset.

### Requirement:

None.

### Run:
python fine_tuning.py

---

### Notes:

Install required packages before running:

pip install transformers datasets accelerate
