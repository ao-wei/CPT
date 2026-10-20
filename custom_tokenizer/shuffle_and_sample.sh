mkdir -p data
gshuf /Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/data_js/output/processed_dataset_512.jsonl \
    | sed 's/<|im_end|>//g' \
    | sed 's/<|im_start|>//g' \
    | head -1600000 \
    > data/processed_dataset_512_sample_160w.jsonl 