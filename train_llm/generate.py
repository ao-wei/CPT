import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载分词器与模型 
# model_path = "/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/train_llm/saved/pretrain/pretrain-qwen-0.1B/"
model_path = "/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/train_llm/saved/continue_pretrain/cpt-qwen-0.1B/"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to("mps")

while True:
    prompt = input("User：")
    inputs = tokenizer([prompt], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        inputs=inputs["input_ids"], # 输入
        attention_mask=inputs["attention_mask"], # attention mask
        max_new_tokens=512, # 输出的最大token数
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=False, # 不做sample
        repetition_penalty=1.2 # 重复惩罚项
    )
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    print("Bot：", response)