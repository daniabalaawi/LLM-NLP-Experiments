
from transformers import pipeline

generator = pipeline("text-generation", model="aubmindlab/aragpt2-base")

prompts = [
    "في إحدى ليالي الشتاء الباردة",
    "تحب دانية الذهاب إلى عملها بسبب",
    "كان الأطفال يلعبون في الملعب ثم"
]

for i, prompt in enumerate(prompts, start=1):
    print("=" * 40)
    print(f"Prompt {i}: {prompt}")
    
    output = generator(
        prompt,
        max_length=40,
        do_sample=False,
        top_k=50,
        top_p=0.95
    )[0]["generated_text"]
    
    print("Generated Text:")
    print(output)
    print()
