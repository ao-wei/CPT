import json
import jieba

from itertools import islice
from tqdm import tqdm
from datasketch import MinHash, MinHashLSH
from concurrent.futures import ProcessPoolExecutor, as_completed


# ========== 计算单篇文档的 MinHash ==========

def process_dialog(idx, text, num_perm):
    words = jieba.lcut(text)
    minhash = MinHash(num_perm=num_perm)

    for word in words:
        minhash.update(word.encode("utf-8"))

    return idx, minhash


# ========== 主程序 ==========

def main():

    # 参数配置
    num_perm = 128
    threshold = 0.7
    max_workers = 2
    sample_size = 1425170

    # 读取数据
    input_path = (
        "/Users/aowei/2026summer/Technology/"
        "大模型训练营/阶段4/代码/quality_model/"
        "data/baike_qa/baike_qa_train.json"
    )

    dialogs = []

    with open(input_path, "r", encoding="utf-8") as fd:
        for line in tqdm(
            islice(fd, sample_size),
            total=sample_size,
            desc="Load Data"
        ):
            info = json.loads(line)
            dialogs.append(info["title"] + info["answer"])

    total_dialogs = len(dialogs)
    print(f"原始文档数量: {total_dialogs}")

    # 创建 LSH 索引
    lsh = MinHashLSH(
        threshold=threshold,
        num_perm=num_perm
    )

    minhashes = [None] * total_dialogs

    # 多进程生成 MinHash
    with ProcessPoolExecutor(max_workers=max_workers) as executor:

        futures = [
            executor.submit(
                process_dialog,
                idx,
                dialogs[idx],
                num_perm
            )
            for idx in range(total_dialogs)
        ]

        for future in tqdm(
            as_completed(futures),
            total=total_dialogs,
            desc="Generate Hash Vectors"
        ):

            idx, minhash = future.result()

            lsh.insert(f"dialog_{idx}", minhash)
            minhashes[idx] = minhash

    # 去重
    unique_dialogs = []
    seen = set()

    print_count = 0
    max_print = 3

    for idx in tqdm(range(total_dialogs), desc="Deduplicate docs"):

        if idx in seen:
            continue

        minhash = minhashes[idx]
        result = lsh.query(minhash)

        similar_idxs = [
            int(r.split("_")[1])
            for r in result
            if r != f"dialog_{idx}"
        ]

        unique_dialogs.append(dialogs[idx])
        seen.update(similar_idxs)

        if similar_idxs and print_count < max_print:
            print(f"\n输入：\n{dialogs[idx]}")
            print(f"\nLSH 检索到的相似候选（目标阈值={threshold}）：")

            for sim_idx in similar_idxs[:max_print]:
                print(f"\n文档 {sim_idx}:\n{dialogs[sim_idx]}")

            print("*" * 100)
            print_count += 1

        seen.add(idx)

    # 打印统计结果
    print("\n==== 去重结果 ====")
    print(f"原始文档数量: {total_dialogs}")
    print(f"去重后文档数量: {len(unique_dialogs)}")
    print(f"过滤文档数量: {total_dialogs - len(unique_dialogs)}")


if __name__ == "__main__":
    main()