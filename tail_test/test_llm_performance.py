import json 
from openai import OpenAI  
import os
import time
import argparse
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

def vllm_complete(args,truth,client,model,tokenizer,prompt,question,options_str, test_time = 5):
    prompts = prompt * test_time
    sampling_params = SamplingParams(temperature=0.8, max_tokens=1024)
    acc = 0
    outputs = model.generate(prompts, sampling_params)
    for output in outputs:
        LLM_answer = output.outputs[0].text   
        completion = client.chat.completions.create(  
        model="gpt-4o",  
        messages=[  
            {"role": "user", "content": f"Please response only in a single character in A,B,C,D,E,F: Here is a question: {question} options: {options_str}. Hers's is the candidate's response:{LLM_answer}.What is the candidate's choice? "}, ] )  
        remark = completion.choices[0].message.content
        if remark[0] == truth[0]:
            acc += 1
    return acc/test_time


def answer(client,ground_truth,prompt,args):
    if args.test_model_name == "gpt-4o":
        completion = client.chat.completions.create(  
        model=args.test_model_name,  
        messages=[{"role": "user", "content": prompt},]  ,
        temperature = 0.8 ,n = 5,)  
        answers = [choice.message.content for choice in completion.choices]
    elif args.test_model_name in ["claude-3-haiku-20240307","gemini-1.5-flash","gemini-1.5-pro"]:
        answers = []
        for i in range(5):
            completion = client.chat.completions.create(  
                model=args.test_model_name,  
                messages=[{"role": "user", "content": prompt},],
                temperature = 0.8,max_tokens = 1024)
            answers.append(completion.choices[0].message.content)
    acc = 0
    for LLM_answer in answers:
        completion = client.chat.completions.create(  
            model="gpt-4o",  
            messages=[  
                {"role": "user", "content": f"Please response only in a single character in A,B,C,D,E,F: Here is a question: {question} options: {options_str}. Hers's is the candidate's response:{LLM_answer}.What is the candidate's choice? "},  
            ]  
        )  
        remark = completion.choices[0].message.content
        if remark[0] == ground_truth[0]:
            acc += 1
    return acc/5

def test_llm_performance(args,client):
    with open(args.QA_save_path, 'r', encoding='utf-8') as file:  
        data = json.load(file) 
    result = []
    is_api_based = True
    if args.test_model_name not in ["gpt-4o","gemini-1.5-flash","gemini-1.5-pro"]:
        is_api_based = False

        tokenizer = AutoTokenizer.from_pretrained(args.test_model_name, trust_remote_code=True)
        model = LLM(
                model = args.test_model_name,
                trust_remote_code=True,
                enforce_eager=True,)

    matching_entries = []
    for depth in args.test_depth_list:
        for length in args.test_doc_length:
            matching_entry = next((item for item in data if item['depth'] == depth and item['document_length'] == length), None)
            matching_entries.append(matching_entry)
            if matching_entry is None:
                raise ValueError(f"No QA found for depth {depth} and token_length {length}. Please generate QA first.")

    
    for item in matching_entries:   
        token_lengths = item["document_length"]
        depth = item['depth']  
        question = item["QA"]['question']  
        options = item["QA"]['options']  
        document = "".join(item["whole_document"])
        context = item["context"]
        # doc_index = item["doc_index"]
        options_str = ' '.join([f"{key}: {value}" for key, value in options.items()])  
        ground_truth = item["QA"]["answer"]
        prompt = ["I will give you a multiple choice question and a corresponding document. Please provide your chain of thoughts." + f"{question} Options: {options_str}"+"Please answer the above question refer to this document only: " + document]
        if is_api_based:
            acc = answer(client,ground_truth,prompt,args)
        else:
            acc = vllm_complete(args,ground_truth,client,model,tokenizer,prompt,question,options_str)
        result.append({"depth":depth,"token_lengths":token_lengths,'result': acc})  
        
        print(f"lengths:{token_lengths},depth:{depth},Answer: {acc*100}%")
    

        
    model_name = args.test_model_name.split('/')[-1]
    if os.path.exists(args.test_result_save_dir + f"result_{model_name}.json"):
        with open(args.test_result_save_dir + f"result_{model_name}.json", 'r') as json_file:
            data = json.load(json_file)
        result = data + result
    
    with open(args.test_result_save_dir + f"result_{model_name}.json", 'w') as json_file:  
        json.dump(result, json_file)  
