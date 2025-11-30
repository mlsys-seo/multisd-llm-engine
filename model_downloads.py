# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
from  huggingface_hub import login
# login("")

models = [
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    # "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",

    # "Qwen/Qwen2.5-32B-Instruct",
    # "Qwen/Qwen2.5-14B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-0.5B-Instruct",
    #"Qwen/Qwen2.5-7B-Instruct",
    #"Qwen/Qwen2.5-3B-Instruct",

    # "meta-llama/Llama-2-7b-chat-hf",
    # "amd/AMD-Llama-135m",
    # "Felladrin/Llama-160M-Chat-v1",


    # "Qwen/Qwen2.5-72B-Instruct",
    # "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-0.5B-Instruct",

    # "Qwen/Qwen3-8B",
    "AngelSlim/Qwen3-8B_eagle3",
    "Qwen/Qwen3-0.6B",


    "AngelSlim/Qwen3-32B_eagle3",
    "AngelSlim/Qwen3-14B_eagle3",
    "Qwen/Qwen3-0.6B",


    # "Qwen/Qwen2.5-14B",
    # "Qwen/Qwen2.5-1.5B",
    # "Qwen/Qwen2.5-0.5B",
    # "Qwen/Qwen2.5-7B",
    # "Qwen/Qwen2.5-3B",
    # "facebook/layerskip-llama2-7B",

    # "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int8",
    # "Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int8",
    # "Qwen/Qwen2.5-0.5B-Instruct-GPTQ-Int8",\

    #"meta-llama/CodeLlama-34b-hf",
    #"TinyLlama/TinyLlama_v1.1_math_code",
    #"amd/AMD-Llama-135m-code",
    
    #"meta-llama/Meta-Llama-3-8B-Instruct",
    #"Felladrin/Llama-68M-Chat-v1",
    #"Felladrin/Llama-160M-Chat-v1"

    # "meta-llama/Llama-2-13b-hf",
    # "TinyLlama/TinyLlama_v1.1",
    # "amd/AMD-Llama-135m",
    
]

for model in models:
    try:
        model = AutoModelForCausalLM.from_pretrained(model, cache_dir="/workspace/cache")
        tokenizer = AutoTokenizer.from_pretrained(model)
    except:
        print(f"model: {model} failed")
        continue
    print(f"model: {model}")
    print(f"tokenizer: {tokenizer}")
