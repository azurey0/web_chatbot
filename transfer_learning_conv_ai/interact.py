# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
import warnings
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat

import torch
import torch.nn.functional as F
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer

from web_chatbot.transfer_learning_conv_ai.train import SPECIAL_TOKENS, build_input_from_segments, add_special_tokens_
from web_chatbot.transfer_learning_conv_ai.utils import get_dataset, download_pretrained_model


class Chatbot:
    def __init__(self, dataset_path='', dataset_cache='./dataset_cache', model='openai-gpt',
    model_checkpoint='', max_history=2, device='cuda', no_sample=False, max_length=20,
    min_length=1, seed=0, temperature=0.7, top_k=0, top_p=0.9):

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        self.device = device
        self.no_sample = no_sample
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.max_length = max_length
        self.min_length = min_length
        self.max_history = max_history
        self.history = []

        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__file__)
        logger.info(str(locals()))

        if model_checkpoint == "":
            if model == 'gpt2':
                raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
            else:
                model_checkpoint = download_pretrained_model()
        
        
        if seed != 0:
            random.seed(seed)
            torch.random.manual_seed(seed)
            torch.cuda.manual_seed(seed)


        logger.info("Get pretrained model and tokenizer")
        tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if model == 'gpt2' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
        self.tokenizer = tokenizer_class.from_pretrained(model_checkpoint)
        self.model = model_class.from_pretrained(model_checkpoint)
        self.model.to(device)
        add_special_tokens_(self.model, self.tokenizer)

        logger.info("Sample a personality")
        dataset = get_dataset(self.tokenizer, dataset_path, dataset_cache)
        personalities = [dialog["personality"] for dataset in dataset.values() for dialog in dataset]
        self.personality = random.choice(personalities)
        logger.info("Selected personality: %s", self.tokenizer.decode(chain(*self.personality)))
    
    def sample_sequence(self, personality, history, tokenizer, model, current_output=None):
        special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
        if current_output is None:
            current_output = []

        for i in range(self.max_length):
            instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

            input_ids = torch.tensor(instance["input_ids"], device=self.device).unsqueeze(0)
            token_type_ids = torch.tensor(instance["token_type_ids"], device=self.device).unsqueeze(0)

            logits = model(input_ids, token_type_ids=token_type_ids)
            if isinstance(logits, tuple):  # for gpt2 and maybe others
                logits = logits[0]
            logits = logits[0, -1, :] / self.temperature
            logits = self.top_filtering(logits, top_k=self.top_k, top_p=self.top_p)
            probs = F.softmax(logits, dim=-1)

            prev = torch.topk(probs, 1)[1] if self.no_sample else torch.multinomial(probs, 1)
            if i < self.min_length and prev.item() in special_tokens_ids:
                while prev.item() in special_tokens_ids:
                    if probs.max().item() == 1:
                        warnings.warn("Warning: model generating special token with probability 1.")
                        break  # avoid infinitely looping over special token
                    prev = torch.multinomial(probs, num_samples=1)

            if prev.item() in special_tokens_ids:
                break
            current_output.append(prev.item())

        return current_output
    
    def top_filtering(self, logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
                top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                    whose total probability mass is greater than or equal to the threshold top_p.
                    In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                    the threshold top_p.
                threshold: a minimal threshold to keep logits
        """
        assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
        top_k = min(top_k, logits.size(-1))
        if top_k > 0:
            # Remove all tokens with a probability less than the last token in the top-k tokens
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            # Compute cumulative probabilities of sorted tokens
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probabilities > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # Back to unsorted indices and set them to -infinity
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        indices_to_remove = logits < threshold
        logits[indices_to_remove] = filter_value

        return logits
    
    def chat(self, message):

        raw_text = message

        self.history.append(self.tokenizer.encode(raw_text))
        with torch.no_grad():
            out_ids = self.sample_sequence(self.personality, self.history, self.tokenizer, self.model)
        self.history.append(out_ids)
        self.history = self.history[-(2*self.max_history+1):]
        out_text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
        return out_text

if __name__ == "__main__":
    chat_bot = Chatbot()

    print(chat_bot.chat('hi'))

