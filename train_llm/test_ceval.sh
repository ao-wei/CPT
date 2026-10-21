uv run lm_eval --model hf \
    --model_args pretrained=/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/train_llm/saved/pretrain/pretrain-qwen-0.1B/ \
    --tasks 'ceval*' \
    --device mps \
    --batch_size 24 \
    --trust_remote_code

uv run lm_eval --model hf \
    --model_args pretrained=/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/train_llm/saved/continue_pretrain/cpt-qwen-0.1B/ \
    --tasks 'ceval*' \
    --device mps \
    --batch_size 24 \
    --trust_remote_code
