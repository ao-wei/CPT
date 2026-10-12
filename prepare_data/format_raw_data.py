# -*- coding: utf-8 -*-
# --------------------------------------------
# 项目名称: 预训练前处理和格式化预训练语料 
# --------------------------------------------

import os
import re
import json
import time
import unicodedata
from tqdm import tqdm

MIN_LEN = 20
MAX_LEN = 768

class FormatHandler():
    def __init__(self, input_path, output_path, dataset_name):
        self.input_path = input_path
        self.output_path = output_path
        self.dataset_name = dataset_name

    def get_file_list(self):
        files = os.listdir(self.input_path)
        files = [i for i in files if ".json" in i]
        return files

    def process_one_line(self, idx, line):
        raise NotImplemented

    def process_one_file(self, file_path):
        line_count = 0
        jump_count = 0
        raw_data = open(self.input_path + "/" + file_path, "r", encoding="utf-8").readlines()

        line_count = len(raw_data)

        with open(self.output_path, "a") as fout:
            for idx, line in tqdm(enumerate(raw_data), total=line_count):
                idx, result = self.process_one_line(idx, line)
                if result:
                    fout.write(result)
                else:
                    jump_count += 1

        return line_count, jump_count

    def process_all(self):
        st = time.time()
        line_count_all = 0
        jump_count_all = 0
        file_list = self.get_file_list()
        print(f"[log][{self.dataset_name}] number of files is {len(file_list):d}")

        for file in file_list:
            line_count, jump_count = 0, 0

            try:
                line_count, jump_count = self.process_one_file(file)
            except Exception as e:
                print(f"[exception][{self.dataset_name}] process file {file} failed: {e}")

            line_count_all += line_count
            jump_count_all += jump_count

        print("[log][{}] timecost is {:.2f} s!".format(self.dataset_name, time.time() - st))
        print("[log][{}] line_count is {:d}, jump_count is {:d}".format(self.dataset_name, line_count_all, jump_count_all))

    def length_assurance(self, line) -> bool:
        if len(line) < MIN_LEN or len(line) > MAX_LEN:
            return False

        return True

    def zh_process(self, line) -> str:
        if line is None:
            return ""

        line = unicodedata.normalize("NFKC", line)

        line = re.sub(r'(\[\d+\]\s*)+$', '', line).rstrip()

        return line

    def qa2text(self, question, answer):
        return "<|im_start|>{}<|im_end|>".format(question) + \
            "<|im_start|>{}<|im_end|>".format(answer)

class BELLEFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(BELLEFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, idx, line):
        data = json.loads(line)
        if "instruction" not in data:
            return idx, None

        instruction = self.zh_process(data["instruction"])
        output = self.zh_process(data["output"])

        text = self.qa2text(instruction, output)

        if not self.length_assurance(text):
            return idx, None

        return idx, json.dumps({"text": text}, ensure_ascii=False) + '\n'

class FireflyFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(FireflyFormatHandler, self).__init__(input_path, output_path, dataset_name)

    def process_one_line(self, idx, line):
        data = json.loads(line)
        input_str = self.zh_process(data["input"])
        target_str = self.zh_process(data["target"])
        text = self.qa2text(input_str, target_str)

        if len(text) > MAX_LEN:
            text = text[:MAX_LEN-1] + "<|im_end|>"

        return idx, json.dumps({"text": text}, ensure_ascii=False) + "\n"

class WikiCNFormatHandler(FormatHandler):
    def __init__(self, input_path, output_path, dataset_name):
        super(WikiCNFormatHandler, self).__init__(input_path, output_path, dataset_name)
    
    def process_one_line(self, idx, line):
        line = json.loads(line)
        item = line["text"]
        text = self.zh_process(item)
        text = text[:MAX_LEN]
        if not self.length_assurance(text):
            return idx, None

        return idx, json.dumps({"text": text}, ensure_ascii=False) + "\n"


def main_run():
    input_path_root = "/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/prepare_data/raw_data"
    output_path_root = "/Users/aowei/2026summer/Technology/大模型训练营/阶段4/代码/prepare_data/formatted_data_512"

    if not os.path.exists(output_path_root):
        os.makedirs(output_path_root)

    dataset_process_info = {
        "wiki_baike": (input_path_root + "/wikibaike", WikiCNFormatHandler),
        "BELLE": (input_path_root + "/belle", BELLEFormatHandler),
        "firefly": (input_path_root + "/firefly", FireflyFormatHandler),
    }

    for dataset_name, info in dataset_process_info.items():
        input_path, Handler = info
        output_path = output_path_root + "/processed_{}.jsonl".format(dataset_name)
        if os.path.exists(output_path):
            os.remove(output_path)
        fh = Handler(input_path, output_path, dataset_name)
        fh.process_all()


if __name__ == "__main__":
    main_run()