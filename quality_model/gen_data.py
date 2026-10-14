import os
import random
import time
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

from dotenv import load_dotenv
from openai import OpenAI

random.seed(42)

QA_QUALITY_PROMPT_TPL = """
你是一个自然语言处理的专家，你的任务是根据输入的文本来判断文本的质量。

以下是一些低质量文本的判断标准：
1.语句不通：语句严重不通顺且没有意义的话
2.特殊字符：标点符号，破折号，特殊字符，emoji，表情符号占比30%以上
3.广告营销：明显的广告推销，博彩，营销内容
4.低俗内容：色情，暴力，仇恨言论
5.隐私泄露：个人的敏感信息，联系方式等

请根据上面的判断标准输出你的结果，如果是低质量，就输出“Y”，否则输出“N”。
请直接输出最终的结果Y或者N，不要输出理由或者其他无关内容。

输入：{}
"""

load_dotenv()

llm_client = OpenAI(
    api_key=os.environ["API_KEY"],
    base_url=os.environ["BASE_URL"]
)

def chat(text, max_retry=3):
    def do_chat(text):
        prompt = QA_QUALITY_PROMPT_TPL.format(text)

        completions = llm_client.chat.completions.create(
            model="qwen3.7-flash-2026-07-15",
            messages=[
                {"role": "system", "content": "你是有用的人工智能助手。"},
                {"role": "user", "content": prompt}
            ]
        )

        return text, completions.choices[0].message.content

    while max_retry > 0:
        try:
            return do_chat(text)
        except Exception as e:
            print(f"API 调用失败：{type(e).__name__}: {e}", flush=True)

            max_retry -= 1

            if max_retry > 0:
                time.sleep(random.randint(1, 4))


    return

MAX_WORKERS = 50

test_sentences = []
fd = open("/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/quality_model/data/baike_qa/baike_qa_valid.json")

for idx, line in enumerate(fd):
    info = json.loads(line)
    test_sentences.append((idx, info["title"] + info["answer"]))

fw = open("/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/quality_model/data/output_quality_full_qwen.json", "w")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {idx: executor.submit(chat, text) for idx, text in test_sentences}
    for idx in tqdm(futures):
        future = futures[idx]
        result = future.result()
        if result is None:
            continue
        text, spam = result
        info = {"text": text, "spam": spam}
        # print(info)
        fw.write(json.dumps(info, ensure_ascii=False) + '\n')