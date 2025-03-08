import logging
import argparse
import numpy as np
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer



class Qwen2Onnx:
    def __init__(self):
        pass

    def load(self, model_path):
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.model.eval()
        logging.info(f'model loaded: {model_path}')

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        logging.info(f'tokenizer loaded: {model_path}')

    def run(self, prompt):
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        input_ids = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        generated_ids = self.model.generate(
            **input_ids,
            max_new_tokens=256
            )
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    def export(self, model_path):
        torch.onnx.export(
            self.model,
            (torch.zeros(1, 256, dtype=torch.long),),
            model_path,
            input_names=['input_ids'],
            output_names=['logits']
        )

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='export llm to various onnx.')
    
    parser.add_argument('-m', '--model', type=str, required=True, help='model path')
    parser.add_argument('-o', '--output', type=str, required=False, default='qwen.onnx', help='output path')
    args = parser.parse_args()

    logging.info(f'use model: {args.model}')

    qwen2onnx = Qwen2Onnx()
    qwen2onnx.load(args.model)

    logging.info('step 1: run torch model...\n')
    text = qwen2onnx.run('简要介绍3种杭州美食.')
    logging.info(f'output: {text}')
    logging.info('run torch model done.\n')


    logging.info('step 2: export onnx model...\n')
    qwen2onnx.export(args.output)
    logging.info(f'export onnx model done: {args.output}')
    