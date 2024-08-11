import json
import argparse
from openai import OpenAI 
import os
import tiktoken
from tqdm import tqdm
  
from nltk.tokenize import sent_tokenize 
def combine_to_paragraphs(sentences, max_length=600):  
    paragraphs = []  
    current_paragraph = ""  
      
    for sentence in sentences:  
        if len(current_paragraph) + len(sentence) <= max_length:  
            current_paragraph += sentence 
        else:  
            current_paragraph += sentence 
            paragraphs.append(current_paragraph)  
            current_paragraph = ""
      
    if current_paragraph:  
        paragraphs.append(current_paragraph)  
    return paragraphs  

def generate_QA(context, client, args, attempt=1, max_attempts=5):
    response = client.chat.completions.create(
        response_format={"type": "json_object"},
        model=args.gen_QA_model_name,
        messages=[{
            "role": "system",
            "content": "You are a helpful AI bot that generates simple and detailed multiple choice question and its respective answer based on a context in the format of JSON. Each question should have options A, B, C, D, E and F and only one correct answer. Question should be clear and has no confusion. Answers should be a single character."
        },
        {
            "role": "user",
            "content": "Here's the context: " + context + " Please respond in the format of json, start with question:."
        }]
    )
    QA = response.choices[0].message.content
    QA = json.loads(QA)

    question = QA["question"]
    options = QA["options"]
    options_str = ' '.join([f"{key}: {value}" for key, value in options.items()])

    acc = 0
    ground_truth = QA["answer"][0]

    ans = []

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I will give you a multiple choice question and a corresponding document. Please provide your chain of thoughts." + f"{question} Options: {options_str}"+"Please answer the above question refer to this document only: " + context},]  ,
        n = 5,
        )  
    answers = [choice.message.content for choice in completion.choices]
    for LLM_answer in answers:
        completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Please respond only in a single character in A,B,C,D,E,F: Here is a question: {question} options: {options_str}. Here's the candidate's response: {LLM_answer}. What is the candidate's choice?"}
            ]
        )
        LLM_answer = completion.choices[0].message.content
        if LLM_answer[0] == ground_truth:
            acc += 1
        ans.append(LLM_answer[0])

    if acc == 5 and "context" not in question:
        return QA
    else:
        if attempt < max_attempts:
            return generate_QA(context, client, args, attempt + 1, max_attempts)
        else:
            # Proceed with the next context or handle this case as needed
            print(f"QA quality check: Failed after {max_attempts} attempts. Proceeding with the next context.")
            return None  # or handle this case as needed

def process_paragraphs(all_paragraphs_tokens, tokenizer, client, args, sublist_index):

    context = tokenizer.decode(all_paragraphs_tokens[sublist_index])
    QA = generate_QA(context, client, args)

    if QA:
        return QA,context
    else:
        return process_paragraphs(all_paragraphs_tokens, tokenizer, client, args, sublist_index + 1)


def find_sublist_index_by_position(nested_list, position):  
    current_position = 0  
    for i, sublist in enumerate(nested_list):  
        for _ in sublist:  
            if current_position == position:  
                return i  
            current_position += 1  
    return -1  

def gen_benchmark(args,client):
    with open(args.raw_document_path, 'r', encoding='utf-8') as file:  
        data_all = json.load(file) 

    sentences = sent_tokenize(data_all[0]["text"])
    paragraphs = combine_to_paragraphs(sentences)
    
    doc_index = 0
    tokenizer = tiktoken.encoding_for_model(args.gen_QA_model_name)

    json_objects = [] 
    all_paragraphs_tokens = []
    all_paragraphs_chars = []
    for context in paragraphs:
        all_paragraphs_tokens.append(tokenizer.encode(context))
        all_paragraphs_chars.append(context)
    
    total_tokens = sum([len(sublist) for sublist in all_paragraphs_tokens])
    total_chars = sum([len(sublist) for sublist in all_paragraphs_chars])
    # print(f"total chars: {int(total_chars/1000)}K")
    print(f"Total tokens in raw document: {total_tokens}") 
    if total_tokens < max(args.document_length):
        raise ValueError("Total tokens in your raw document are less than the maximum document length specified! \nPlease input a longer document or decrease the maximum document length.")
    for depth in tqdm(args.depth_list, desc="Generating QA for each depths"):
        needle_point = int(total_tokens * (depth / 100))
        sublist_index = find_sublist_index_by_position(all_paragraphs_tokens, needle_point)
        QA,context = process_paragraphs(all_paragraphs_tokens, tokenizer, client, args, sublist_index)
        for doc_len in args.document_length:
            ratio = doc_len/total_tokens
            new_doc_encode = all_paragraphs_tokens[int(sublist_index*(1-ratio)):int(sublist_index+ratio*(len(all_paragraphs_tokens)-sublist_index))]
            new_doc = []
            while sum([len(sublist) for sublist in new_doc_encode]) > doc_len:
                new_doc_encode = new_doc_encode[:-1]
            # print(ratio,sum([len(sublist) for sublist in new_doc_encode]))

            for para in new_doc_encode:
                new_doc.append(tokenizer.decode(para))
            QA_json = {}  
            QA_json["QA"] = QA 
            QA_json["doc_index"] = doc_index
            QA_json['depth'] = int(depth)  
            QA_json['context'] = context  
            QA_json["document_length"] = doc_len 
            QA_json["whole_document"] = new_doc  
            
            json_objects.append(QA_json)  
    doc_index += 1

    with open(args.QA_save_path, 'w', encoding='utf-8') as f:  
        json.dump(json_objects, f, ensure_ascii=False, indent=4)  
