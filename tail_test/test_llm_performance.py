import json 
from openai import OpenAI  
import os
import termcolor
import time
import argparse
import matplotlib.pyplot as plt
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from tail_test.benchmark_generation import gen_benchmark
from tail_test.visualize import visualize


def vllm_complete(truth,client,model,tokenizer,messages,question,options_str):
    sampling_params = SamplingParams(temperature=0.0, max_tokens=512)

    inputs = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = model.generate(prompts=[inputs], sampling_params=sampling_params)
    answers = outputs[0].output.outputs[0].text

    completion = client.chat.completions.create(  
        model="gpt-4o",  
        messages=[  
            {"role": "user", "content": f"Please response only in a single character in A,B,C,D,E,F: Here is a question: {question} options: {options_str}. Hers's is the candidate's response:{answers}.What is the candidate's choice? "},  
        ]  
    )  
        
    remarks = completion.choices[0].message.content.strip()
    return remarks[0]==truth[0],answers,remarks

def api_answer(client,ground_truth,messages,args,question,options_str):
    completions = client.chat.completions.create(
                model=args.test_model_name,
                messages=messages,
                max_tokens=256, temperature=0.0)
    answers = completions.choices[0].message.content.strip()
    completion = client.chat.completions.create(  
        model="gpt-4o",  
        messages=[  
            {"role": "user", "content": f"Please response only in a single character in A,B,C,D,E,F: Here is a question: {question} options: {options_str}. Hers's is the candidate's response:{answers}.What is the candidate's choice? "},  
        ]  
    )  
    remark = completion.choices[0].message.content
    return remark[0] == ground_truth[0],answers,remark

def test_llm_performance(args,client):
    model_name = args.test_model_name.split('/')[-1] if '/' in args.test_model_name else args.test_model_name
    with open(args.QA_save_path, 'r', encoding='utf-8') as file:  
        data = json.load(file) 
    
    result = []
    
    is_api_based = True
    if args.test_model_name not in ["gpt-4o","gemini-1.5-flash","gemini-1.5-pro","glm-4-flash"]:
        is_api_based = False

        tokenizer = AutoTokenizer.from_pretrained(args.test_model_name, trust_remote_code=True)
    
        model = LLM(
            model = args.test_model_name, 
            dtype="auto",   
            trust_remote_code=True,  
            seed=2024,
            tensor_parallel_size=1,
            enforce_eager=False)
        
    matching_entries = []
    for depth in args.test_depth_list:
        for length in args.test_doc_length:
            matching_items = [item for item in data if item['depth'] == depth and item['document_length'] == length]
            if not matching_items:
                raise ValueError(f"No QA found for depth {depth} and token_length {length}. Please generate QA first.")
            matching_entries.extend(matching_items)

    num = 0
    start_time = time.time()
    print(f"[TAIL] Start testing {model_name}. Total test sample numbers: {len(matching_entries)}")
    for item in matching_entries:   
        num += 1
        token_lengths = item["document_length"]
        depth = item['depth']  
        question = item["QA"]['question']  
        options = item["QA"]['options']  
        document = "".join(item["whole_document"])
        doc_index = item["doc_index"]
        options_str = ' '.join([f"{key}: {value}" for key, value in options.items()])  
        ground_truth = item["QA"]["answer"]
        messages = [{"role": "user", "content": "I will give you a multiple choice question and a corresponding document. Please think step by step to answer and provide your chain of thoughts." + f"Questions: {question} Options: {options_str}"+"Please answer the above question refer to this document only: " + document}] 

        if is_api_based:
            acc,answer,extract_answer = api_answer(client,ground_truth,messages,args,question,options_str)
        else:
            acc,answer,extract_answer = vllm_complete(ground_truth,model,tokenizer,messages,question,options_str)
        result.append({"doc_index":doc_index,"depth":depth,"token_lengths":token_lengths,'result': acc, "question":question, "options":options,'LLM_original_answer': answer,"LLM_extracted_answer":extract_answer ,"ground_truth":ground_truth})  
        point_time = time.time()
        print(f"{num}/{len(matching_entries)},lengths:{token_lengths},depth:{depth},Answer: {acc*100}% "+f"Time: {round((point_time - start_time)/60, 1)} min Used")
    
        # if (num+1) % 10 == 0:
        #     model_name = args.test_model_name.split('/')[-1]
            
        #     print(termcolor.colored(f"Num saved: {len(result)}", 'green'))
        #     with open(args.test_result_save_dir + f"result_{model_name}.json", 'w') as json_file:  
        #         json.dump(result, json_file)  
        #     visualize(args)
        
        
    
    # if os.path.exists(args.test_result_save_dir + f"result_{model_name}.json"):
    #     with open(args.test_result_save_dir + f"result_{model_name}.json", 'r') as json_file:
    #         data = json.load(json_file)
    #     result = data + result
    print(f"[TAIL] Sucessfully finished testing {model_name} on {args.QA_save_path}.")
    with open(os.path.join(args.test_result_save_dir, f"result_{model_name}.json"), 'w') as json_file:  
        json.dump(result, json_file)  

