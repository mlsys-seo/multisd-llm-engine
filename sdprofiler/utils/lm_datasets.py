import random
from typing import Optional

import numpy as np
from datasets import load_dataset, concatenate_datasets, Dataset

import os, sys, json, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cached_data")

def get_system_prompt(model_name: str):
    if "llama" in model_name.lower():
        return "You are a helpful assistant. Keep your responses limited to short paragraph if possible."
    elif "qwen" in model_name.lower():
        return "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    else:
        return "You are a helpful assistant. Keep your responses limited to one short paragraph if possible."

def preprocess_mtbench(data):
    prompts = []
    for item in data:
        prompt = item['conversation_a'][0]["content"]
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_specbench(data):
    prompts = []
    for item in data:
        prompt = "\n".join(item['turns'])
        category = item['category']
        prompts.append({'prompt': prompt, 'category': category})
    return prompts

def preprocess_mbpp(data):
    prompts = []
    for item in data:
        prompt = f"Complete the following function based on the given description:\n\n{item['text']} and test case:\n\n{item['test_list']}\n\nFunction:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_chatgpt(data):
    prompts = []
    for item in data:
        prompt = f"{item['human_prompt']}\n\nAssistant:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_math500(data):
    prompts = []
    for item in data:
        prompt = f"Answer the following question and show your solution and answer. Please reason step by step, and put your final answer in latex format.:\n\n{item['problem']}\n\nSolution:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_gsm8k(data):
    prompts = []
    for item in data:
        article = item['question']
        prompt = f"Answer the following question and show your solution and answer. Please reason step by step, and put your final answer in latex format.:\n\n{article}\n\nSolution:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_dailymail(data):
    prompts = []
    for item in data:
        article = item['article']
        prompt = f"Summarize the following article:\n\n{article}\n\nSummary:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_humaneval(data):
    prompts = []
    for item in data:
        prompt = f"Complete the following function based on the given description:\n\n{item['prompt']}\n\nFunction:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_humanevalpack(data):
    prompts = []
    for item in data:
        prompt = f"Complete the following function based on the given description:\n\n{item['instruction']}\n\nFunction:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_wmt(data):
    prompts = []
    for item in data:
        source_text = item['translation']['en']
        prompt = f"Translate the following sentence into French:\n\n{source_text}\n\nTranslation:"
        prompts.append({'prompt': prompt})
    return prompts

def preprocess_hellaswag(data):
    prompts = []
    for item in data:
        context = item['ctx']
        activity_label = item['activity_label']
        prompt = f"Complete the following scenario: {context}"

        prompts.append({'prompt': prompt, 'label': activity_label})
    return prompts

def preprocess_sharegpt(data):
    if not isinstance(data, list):
        data = [data]
    prompts = []
    for item in data:
        if len(item['conversations']) < 3:
            continue
        
        text_conversation = ""
        conversations = []
        
        hook = False
        for message in item['conversations'][:3]:
            role = "user" if message['from'] == "human" else "assistant"
            if role == "assistant":
                if hook:
                    break
                hook = True
            text_conversation += f"{role}: {message['value']}\n"
            conversations.append({'role': role, 'content': message['value']})
        
        prompt = f"The following is a conversation between a user and an assistant. Continue the conversation based on the context:\n\n{text_conversation}\nAssistant:"
        prompts.append({'prompt': prompt, 'conversations': conversations})
    return prompts


def load_and_preprocess_dataset(dataset_name: str, num_samples: Optional[int] = None, seed: Optional[int] = None, is_train_set: bool = False):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)


    dataset_dir = os.path.join(TEMP_DIR, "datasets")

    os.makedirs(dataset_dir, exist_ok=True)
    is_train_str = "_train" if is_train_set else ""
    dataset_path = os.path.join(dataset_dir, f"{dataset_name}{is_train_str}_{num_samples}_processed_{seed}.json")

    if os.path.exists(dataset_path):
        print(f"Loading processed dataset from {dataset_path}")
        with open(dataset_path, "r") as f:
            return json.load(f)
    
    if dataset_name == 'humaneval':
        dataset_name = 'humanevalpack'
    

    if dataset_name == 'mtbench':
        data = load_dataset('lmsys/mt_bench_human_judgments', split=f'human')
        train_split = int(len(data) * 0.8)
        if is_train_set:
            data = data.select(range(train_split))
        else:
            data = data.select(range(train_split, len(data)))

        data = data.shuffle(seed=seed)
        if num_samples is not None:
            num_samples = min(num_samples, len(data))
            data = data.select(range(num_samples))
        processed_data = preprocess_mtbench(data)

    elif 'specbench' in dataset_name:

        # writing, roleplay, reasoning, math, coding, extraction, stem, humanities, translation, summarization, qa, math_reasoning, rag
        if "--" in dataset_name:
            dataset_name, category = dataset_name.split("--")
        else:
            category = None
        
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'specbench.jsonl'), 'r') as f:
            data = [json.loads(line) for line in f]
        

        # Rename category values for specific categories to "multi-turn"
        multi_turn_categories = ["writing", "roleplay", "reasoning", "math", "coding", "extraction", "stem", "humanities"]
        def rename_category(example):
            if example["category"] in multi_turn_categories:
                example["category"] = "multiturn"
            return example
        data = [rename_category(item) for item in data]


        grouped = {}
        for item in data:
            cat = item.get('category')
            if cat not in grouped:
                grouped[cat] = []
            grouped[cat].append(item)

        data = []
        for cat, items in grouped.items():
            n = len(items)
            k = math.ceil(n * 0.1)
            if is_train_set:
                selected = items[:k]
            else:
                selected = items[k:]
            data.extend(selected)

        data = {k: [d[k] for d in data] for k in data[0].keys()}
        data = Dataset.from_dict(data)
        data = data.shuffle(seed=seed)
        if category is not None:
            data = data.filter(lambda example: example["category"] == category)
        
        if num_samples is not None:
            num_samples = min(num_samples, len(data))
            data = data.select(range(num_samples))
        processed_data = preprocess_specbench(data)

    
    elif dataset_name == 'dailymail':
        if is_train_set:
            data = load_dataset('cnn_dailymail', '3.0.0', split=f'train')
        else:
            data = load_dataset('cnn_dailymail', '3.0.0', split=f'validation')
        data = data.shuffle(seed=seed)
        if num_samples is not None:
            num_samples = min(num_samples, len(data))
            data = data.select(range(num_samples))
        processed_data = preprocess_dailymail(data)
    
    # elif dataset_name == 'humaneval':
        # if is_train_set:
        #     data = load_dataset('openai_humaneval', split=f'train')
        # else:
        #     data = load_dataset('openai_humaneval', split=f'test')
        # data = data.shuffle(seed=seed)
        # data = data.select(range(num_samples))
        # processed_data = preprocess_humaneval(data)
        
        
    elif dataset_name == 'humanevalpack':
        python_data = load_dataset('bigcode/humanevalpack', "python", split=f'test', trust_remote_code=True)
        cpp_data = load_dataset('bigcode/humanevalpack', "cpp", split=f'test', trust_remote_code=True)
        java_data = load_dataset('bigcode/humanevalpack', "java", split=f'test', trust_remote_code=True)
        js_data = load_dataset('bigcode/humanevalpack', "js", split=f'test', trust_remote_code=True)
        rust_data = load_dataset('bigcode/humanevalpack', "rust", split=f'test', trust_remote_code=True)
        go_data = load_dataset('bigcode/humanevalpack', "go", split=f'test', trust_remote_code=True)
        data = concatenate_datasets([python_data, cpp_data, java_data, js_data, rust_data, go_data])
        data = data.shuffle(seed=1234)
        train_split = int(len(data) * 0.5)
        if is_train_set:
            data = data.select(range(1, train_split))
        else:
            data = data.select(range(train_split, len(data)))

        data = data.shuffle(seed=seed)
        if num_samples is not None:
            num_samples = min(num_samples, len(data))
            data = data.select(range(num_samples))
        processed_data = preprocess_humanevalpack(data)

    elif dataset_name == 'mbpp':
        if is_train_set:
            data = load_dataset('google-research-datasets/mbpp', split=f'train')
        else:
            data = load_dataset('google-research-datasets/mbpp', split=f'test')
        data = data.shuffle(seed=seed)
        if num_samples is not None:
            num_samples = min(num_samples, len(data))
            data = data.select(range(num_samples))
        processed_data = preprocess_mbpp(data)

    elif dataset_name == 'chatgpt':
        if is_train_set:
            data = load_dataset('MohamedRashad/ChatGPT-prompts', split=f'train')
            data = data.select(range(120))
        else:
            data = load_dataset('MohamedRashad/ChatGPT-prompts', split=f'train')
            data = data.select(range(120, len(data)))

        data = data.shuffle(seed=seed)
        if num_samples is not None:
            num_samples = min(num_samples, len(data))
            data = data.select(range(num_samples))
        processed_data = preprocess_chatgpt(data)

    elif dataset_name == 'gsm8k':
        if is_train_set:
            data = load_dataset('gsm8k', 'main', split=f'train')
        else:
            data = load_dataset('gsm8k', 'main', split=f'test')
        data = data.shuffle(seed=seed)
        if len(data) > num_samples:
            data = data.select(range(num_samples))
        processed_data = preprocess_gsm8k(data)


    elif dataset_name == 'math500':
        if is_train_set:
            data = load_dataset('HuggingFaceH4/MATH-500', split=f'test')
            data = data.select(range(250))
        else:
            data = load_dataset('HuggingFaceH4/MATH-500', split=f'test')
            data = data.select(range(250, 500))
        data = data.shuffle(seed=seed)
        if len(data) > num_samples:
            data = data.select(range(num_samples))
        processed_data = preprocess_math500(data)
        
    elif dataset_name == 'wmt':
        if is_train_set:
            data = load_dataset('wmt14', 'fr-en', split=f'train')
        else:
            data = load_dataset('wmt14', 'fr-en', split=f'validation')
        data = data.shuffle(seed=seed)
        data = data.select(range(num_samples))
        processed_data = preprocess_wmt(data)
    
    elif dataset_name == 'hellaswag':
        if is_train_set:
            data = load_dataset('hellaswag', split=f'train', trust_remote_code=True)
        else:
            data = load_dataset('hellaswag', split=f'validation', trust_remote_code=True)
        data = data.shuffle(seed=seed)
        if num_samples is not None:
            num_samples = min(num_samples, len(data))
            data = data.select(range(num_samples))
        processed_data = preprocess_hellaswag(data)
    
    elif dataset_name == 'sharegpt':
        from urllib.request import urlretrieve

        url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
        local_path = os.path.join(dataset_dir, "ShareGPT_V3_unfiltered_cleaned_split.json")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        # Download the file if it doesn't exist locally
        if not os.path.exists(local_path):
            print(f"Downloading ShareGPT dataset to {local_path}")
            urlretrieve(url, local_path)
        else:
            print(f"ShareGPT dataset already exists at {local_path}")

        # Load the dataset
        with open(local_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        train_split = int(len(data) * 0.8)
        if is_train_set:
            data = data[:train_split]
        else:
            data = data[train_split:]

        random.shuffle(data)
        if num_samples is not None:
            # If num_samples is specified, limit the dataset to that number of samples
            # we cut off num_turn < 3 conversations so we need iterate over the dataset
            processed_data = []
            idx = 0
            while len(processed_data) < num_samples:
                processed_data.extend(preprocess_sharegpt(data[idx]))
                idx += 1
        else:
            processed_data = preprocess_sharegpt(data)

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # save processed data
    with open(dataset_path, "w") as f:
        json.dump(processed_data, f)
    
    print(f"Saved processed dataset to {dataset_path}")
    return processed_data


def apply_chat_template(datasets, tokenizer):
    model_name = tokenizer.name_or_path
    system_prompt = get_system_prompt(model_name)
    
    if tokenizer.chat_template is not None:
        requests = [] # List of tuples (request_id, prompt)
        for idx, query in enumerate(datasets):
            # query: {'prompt': str, 'conversations': List[Dict[str, str]], 'label': str}
            if "label" in query:
                request_id = f'{idx}_{query["label"]}'
            else:
                request_id = str(idx)

            conversation = [{"role": "system", "content": system_prompt}]
            if 'conversations' in query:
                conversation.extend(query['conversations'])
            else:
                conversation.extend([
                    {"role": "user", "content": query['prompt']},
                    ])
            prompt = tokenizer.apply_chat_template(
                conversation,
                tokenize=False,
                add_generation_prompt=True)
            requests.append(prompt)

    else:
        requests = []
        for idx, query in enumerate(datasets):
            if "label" in query:
                request_id = f'{idx}_{query["label"]}'
            else:
                request_id = str(idx)
            requests.append(f"{system_prompt}\n\n{query['prompt']}")

    return requests

if __name__ == "__main__":
    num_samples = 256
    seed = 9999
    
    for dataset_name in ['specbench', 'specbench--multiturn', 'specbench--translation', 'specbench--summarization', 'specbench--qa', 'specbench--math_reasoning', 'specbench--rag']:
        data = load_and_preprocess_dataset(dataset_name, num_samples, seed)
        print(f"test set {dataset_name}: {len(data)}")

        
    for dataset_name in ['specbench', 'specbench--multiturn', 'specbench--translation', 'specbench--summarization', 'specbench--qa', 'specbench--math_reasoning', 'specbench--rag']:
        data = load_and_preprocess_dataset(dataset_name, num_samples, seed, is_train_set=True)
        print(f"train set {dataset_name}: {len(data)}")
