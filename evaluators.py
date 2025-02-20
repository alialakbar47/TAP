import re
from fastchat.model import (
    get_conversation_template
)
from system_prompts import get_evaluator_system_prompt_for_judge, get_evaluator_system_prompt_for_on_topic
from language_models import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_evaluator(args):
    if "gpt" in args.evaluator_model:
        return GPTEvaluator(args)
    elif args.evaluator_model == "no-evaluator":
        return NoEvaluator(args)
    elif "llama" in args.evaluator_model:
        return LlamaEvaluator(args)
    else:
        raise ValueError(f"Unsupported evaluator model: {args.evaluator_model}")

class EvaluatorBase:
    def __init__(self, args):
        self.max_n_tokens = args.evaluator_max_n_tokens
        self.temperature = args.evaluator_temperature
        self.evaluator_name = args.evaluator_model
        self.goal = args.goal
        self.target_str = args.target_str
        self.system_prompt = get_evaluator_system_prompt_for_judge(args.goal, args.target_str)
        self.system_prompt_on_topic = get_evaluator_system_prompt_for_on_topic(args.goal)

    def get_evaluator_prompt(self, attack_prompt, target_response):
        prompt = f"[PROMPT]:{attack_prompt}\n[ASSISTANT'S RESPONSE]: {target_response}"
        return prompt

    def get_evaluator_prompt_on_topic(self, attack_prompt):
        prompt = f"[PROMPT]:{attack_prompt}"
        return prompt

    def judge_score(self, attack_prompt_list, target_response_list):
        raise NotImplementedError
    def process_output_judge_score(self, raw_output):
        if isinstance(raw_output, list):
          raw_output = raw_output[0]  # Extract the first element if it's a list
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
           output = 1
        return output


    def process_output_on_topic_score(self, raw_output):
       if isinstance(raw_output, list):
           raw_output = raw_output[0]  # Extract the first element if it's a list
       pattern = r'\[\[(yes|no)\]\]'
       match = re.search(pattern, raw_output.lower())
       output = int(match.group(1) == 'yes') if match else None
       if output is None:
           output = 1
       return output


class NoEvaluator(EvaluatorBase):
    def __init__(self, args):
        super(NoEvaluator, self).__init__(args)
        self.evaluator_model = None

    def judge_score(self, attack_prompt_list, target_response_list):
        return [1 for _ in attack_prompt_list]

    def on_topic_score(self, attack_prompt_list, original_prompt):
        return [1 for _ in attack_prompt_list]

class GPTEvaluator(EvaluatorBase):
    def __init__(self, args):
        super(GPTEvaluator, self).__init__(args)
        self.evaluator_model = GPT(model_name=self.evaluator_name)

    def create_conv(self, full_prompt, system_prompt=None):
        if system_prompt is None:
            system_prompt = self.system_prompt

        conv = get_conversation_template(self.evaluator_name)
        conv.set_system_message(system_prompt)
        conv.append_message(conv.roles[0], full_prompt)

        return conv.to_openai_api_messages()

    def judge_score(self, attack_prompt_list, target_response_list):
        convs_list = [
            self.create_conv(self.get_evaluator_prompt(prompt, response))
            for prompt, response in zip(attack_prompt_list, target_response_list)
        ]

        print(f'\tQuerying evaluator with {len(attack_prompt_list)} prompts (to evaluate judge scores)', flush=True)

        raw_outputs = self.evaluator_model.batched_generate(convs_list, 
                                                            max_n_tokens=self.max_n_tokens,
                                                            temperature=self.temperature)

        outputs = [self.process_output_judge_score(raw_output) for raw_output in raw_outputs]
        return outputs

    def on_topic_score(self, attack_prompt_list, original_prompt):
        convs_list = [
            self.create_conv(self.get_evaluator_prompt_on_topic(prompt), system_prompt=self.system_prompt_on_topic)
            for prompt in attack_prompt_list
        ]

        print(f'\tQuerying evaluator with {len(attack_prompt_list)} prompts (to evaluate on-topic scores)', flush=True)

        raw_outputs = self.evaluator_model.batched_generate(convs_list, 
                                                            max_n_tokens=self.max_n_tokens,
                                                            temperature=self.temperature)
        outputs = [self.process_output_on_topic_score(raw_output) for raw_output in raw_outputs]
        return outputs

class LlamaEvaluator(EvaluatorBase):
    def __init__(self, args):
        super(LlamaEvaluator, self).__init__(args)
        self.model_path = args.evaluator_model

        print(f"Loading LLaMA model from {self.model_path} on CPU...")
        self.evaluator_model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct"
        ).to("cpu").eval()  # ✅ Move model to CPU

        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-3.2-1B-Instruct",
            use_fast=False
        )

    def generate(self, prompt):
        # Tokenize the prompt
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids

        # Move inputs to CPU
        input_ids = input_ids.to("cpu")

        # Generate output
        outputs = self.evaluator_model.generate(
            input_ids=input_ids,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        # Decode outputs
        decoded_outputs = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        print(decoded_outputs)
        return decoded_outputs

    def judge_score(self, attack_prompt_list, target_response_list):
       prompts = [self.get_evaluator_prompt(prompt, response) for prompt, response in zip(attack_prompt_list, target_response_list)]
       print(f'\tQuerying LLaMA evaluator with {len(prompts)} prompts (to evaluate judge scores)', flush=True)
    
       raw_outputs = [self.generate(prompt) for prompt in prompts]
       outputs = [self.process_output_judge_score(output[0]) for output in raw_outputs]
       return outputs


    def on_topic_score(self, attack_prompt_list, original_prompt):
       prompts = [self.get_evaluator_prompt_on_topic(prompt) for prompt in attack_prompt_list]
       print(f'\tQuerying LLaMA evaluator with {len(prompts)} prompts (to evaluate on-topic scores)', flush=True)
    
       raw_outputs = [self.generate(prompt) for prompt in prompts]
       outputs = [self.process_output_on_topic_score(output[0]) for output in raw_outputs]
       return outputs

