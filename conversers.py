import common
from language_models import GPT, PaLM, HuggingFace, APIModelLlama7B, APIModelVicuna13B, GeminiPro
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from config import VICUNA_PATH, LLAMA_PATH, ATTACK_TEMP, TARGET_TEMP, ATTACK_TOP_P, TARGET_TOP_P, MAX_PARALLEL_STREAMS
from transformers.utils import logging
logging.set_verbosity_error()  # Only show errors

def load_target_model(args, system_prompt=None):
    target_llm = TargetLLM(model_name=args.target_model,
                           max_n_tokens=args.target_max_n_tokens,
                           temperature=TARGET_TEMP,
                           top_p=TARGET_TOP_P,
                           system_prompt=system_prompt)  # Pass system_prompt here
    return target_llm

def load_attack_and_target_models(args, system_prompt=None):
    attack_llm = AttackLLM(model_name=args.attack_model,
                           max_n_tokens=args.attack_max_n_tokens,
                           max_n_attack_attempts=args.max_n_attack_attempts,
                           temperature=ATTACK_TEMP,
                           top_p=ATTACK_TOP_P)
    preloaded_model = None
    if args.attack_model == args.target_model:
        print("Using same attack and target model. Using previously loaded model.")
        preloaded_model = attack_llm.model
    target_llm = TargetLLM(model_name=args.target_model,
                           max_n_tokens=args.target_max_n_tokens,
                           temperature=TARGET_TEMP,
                           top_p=TARGET_TOP_P,
                           preloaded_model=preloaded_model,
                           system_prompt=system_prompt)  # Pass system_prompt here
    return attack_llm, target_llm

class AttackLLM():
    def __init__(self, model_name, max_n_tokens, max_n_attack_attempts, temperature, top_p):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.max_n_attack_attempts = max_n_attack_attempts
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)

        if "vicuna" in model_name or "llama" in model_name:
            if "api-model" not in model_name:
                self.model.extend_eos_tokens()

    def get_attack(self, convs_list, prompts_list):
        assert len(convs_list) == len(prompts_list), "Mismatch between number of conversations and prompts."
        batchsize = len(convs_list)
        indices_to_regenerate = list(range(batchsize))
        valid_outputs = [None] * batchsize

        if len(convs_list[0].messages) == 0:
            init_message = """{\"improvement\": \"\",\"prompt\": \"""" 
        else:
            init_message = """{\"improvement\": \"""" 

        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            else:
                conv.append_message(conv.roles[1], init_message)
                full_prompts.append(conv.get_prompt()[:-len(conv.sep2)])

        for _ in range(self.max_n_attack_attempts):
            full_prompts_subset = [full_prompts[i] for i in indices_to_regenerate]
            outputs_list = []
            for left in range(0, len(full_prompts_subset), MAX_PARALLEL_STREAMS):
                right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts_subset))
                if right == left:
                    continue

                outputs_list.extend(
                    self.model.batched_generate(full_prompts_subset[left:right],
                                                max_n_tokens=self.max_n_tokens,
                                                temperature=self.temperature,
                                                top_p=self.top_p)
                )

            new_indices_to_regenerate = []
            for i, full_output in enumerate(outputs_list):
                orig_index = indices_to_regenerate[i]
                if "gpt" not in self.model_name:
                    full_output = init_message + full_output

                attack_dict, json_str = common.extract_json(full_output)
                if attack_dict is not None:
                    valid_outputs[orig_index] = attack_dict
                    convs_list[orig_index].update_last_message(json_str)
                else:
                    new_indices_to_regenerate.append(orig_index)

            indices_to_regenerate = new_indices_to_regenerate
            if not indices_to_regenerate:
                break

        return valid_outputs

class TargetLLM():
    def __init__(self, model_name, max_n_tokens, temperature, top_p, preloaded_model=None, system_prompt=None):
        self.model_name = model_name
        self.temperature = temperature
        self.max_n_tokens = max_n_tokens
        self.top_p = top_p
        self.system_prompt = system_prompt  # New parameter for system prompt

        if preloaded_model is None:
            self.model, self.template = load_indiv_model(model_name)
        else:
            self.model = preloaded_model
            _, self.template = get_model_path_and_template(model_name)

    def get_response(self, prompts_list):
        batchsize = len(prompts_list)
        convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []
        for conv, prompt in zip(convs_list, prompts_list):
            if self.system_prompt:
                conv.append_message(conv.roles[0], self.system_prompt)  # Add system prompt if provided
            conv.append_message(conv.roles[0], prompt)
            if "gpt" in self.model_name:
                full_prompts.append(conv.to_openai_api_messages())
            elif "palm" in self.model_name:
                full_prompts.append(conv.messages[-1][1])
            else:
                conv.append_message(conv.roles[1], None)
                full_prompts.append(conv.get_prompt())

        outputs_list = []
        for left in range(0, len(full_prompts), MAX_PARALLEL_STREAMS):
            right = min(left + MAX_PARALLEL_STREAMS, len(full_prompts))
            if right == left:
                continue

            outputs_list.extend(
                self.model.batched_generate(full_prompts[left:right],
                                            max_n_tokens=self.max_n_tokens,
                                            temperature=self.temperature,
                                            top_p=self.top_p)
            )

        return outputs_list

def load_indiv_model(model_name):
    model_path, template = get_model_path_and_template(model_name)
    common.MODEL_NAME = model_name

    if model_name in ["gpt-3.5-turbo", "gpt-4", 'gpt-4-1106-preview']:
        lm = GPT(model_name)
    elif model_name == "palm-2":
        lm = PaLM(model_name)
    elif model_name == "gemini-pro":
        lm = GeminiPro(model_name)
    elif model_name == 'llama-2-api-model':
        lm = APIModelLlama7B(model_name)
    elif model_name == 'vicuna-api-model':
        lm = APIModelVicuna13B(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype="auto").eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template

def get_model_path_and_template(model_name):
    full_model_dict = {
        "gpt-4": {"path": "gpt-4", "template": "gpt-4"},
        "vicuna": {"path": VICUNA_PATH, "template": "vicuna_v1.1"},
        "llama-2": {"path": LLAMA_PATH, "template": "llama-2"},
        "palm-2": {"path": "palm-2", "template": "palm-2"},
        "gemini-pro": {"path": "gemini-pro", "template": "gemini-pro"},
    }
    path, template = full_model_dict[model_name]["path"], full_model_dict[model_name]["template"]
    return path, template
