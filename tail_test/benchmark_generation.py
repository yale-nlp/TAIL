import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse
from openai import OpenAI 
import os
import tiktoken
from tqdm import tqdm
import numpy as np
import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize 
import tiktoken
tokenizer = tiktoken.encoding_for_model("gpt-4o")

def count_tokens(text, tokenizer):
    return len(tokenizer.encode(text))

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

def RAG_filter(text, paragraphs, paragraph_embeds, context, question, options_str, ground_truth, client, model="text-embedding-3-large", top_n=3):
    question_embed = client.embeddings.create(input=[text], model=model).data[0].embedding

    similarities = [np.dot(question_embed, p_embed) / (np.linalg.norm(question_embed) * np.linalg.norm(p_embed)) for p_embed in paragraph_embeds]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_paragraphs = [paragraphs[i] for i in top_indices]
    filtered_paragraphs = [p.replace(context, "") for p in top_paragraphs]
    combined_paragraphs = " ".join(filtered_paragraphs)
    ans = []
    acc = 0
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "I will give you a multiple choice question and a corresponding document. Please provide your chain of thoughts. If the document doesn't contain direct evidence to the question, simply respond the question is unanswerable." + f"{question} Options: {options_str}"+"Please answer the above question refer to this document only: " + combined_paragraphs},]  ,
        )  
    #print(completion.choices[0].message.content)
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Please respond only in a single character in A,B,C,D,E,F: Here is a question: {question} options: {options_str}. Here's the candidate's response: {completion.choices[0].message.content}. What is the candidate's choice? If the candidate's response is unanswerable, please respond with 'U'."}])
    LLM_answer = completion.choices[0].message.content
    if LLM_answer[0] == ground_truth:
        acc += 1
    ans.append(LLM_answer[0])
   
    return acc == 0


def generate_QA(context, all_paragraphs_chars, p_embed, client, args, attempt=1, max_attempts=5):
    example = '''{
            'question': 'What is required from license holders to obtain an export permit for cannabis under the Cannabis Act?',
            'options': {
                'A': "Written consent from the importing country's government",
                'B': 'Detailed information about the substance to be exported and the importer',
                'C': 'Approval from the federal Minister of Health',
                'D': 'A permit application fee and proof of business registration',
                'E': 'A detailed environmental impact report for the shipment',
                'F': 'Consent from local law enforcement'
            },
            'answer': 'B'
        }'''

    response = client.chat.completions.create(
        response_format={"type": "json_object"},
        model=args.gen_QA_model_name,
        messages=[{
            "role": "system",
            "content": "You are a helpful AI bot that generates simple and detailed multiple choice question and its respective answer based on a context in the format of JSON. Each question should have options A, B, C, D, E and F and only one correct answer. Question should be clear and has no confusion. This context is from a long document and I want you to try your best to make the question shouldn't be solve by other parts of the document. Answers should be a single character."
        },
        {
            "role": "user",
            "content": "Here's the context: " + context + " Please respond in the format of json, here's an example: " + example
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
        ) 
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": f"Please respond only in a single character in A,B,C,D,E,F: Here is a question: {question} options: {options_str}. Here's the candidate's response: {completion.choices[0].message.content}. What is the candidate's choice?"}
        ]
    )
    LLM_answer = completion.choices[0].message.content
    if LLM_answer[0] == ground_truth:
        acc += 1
    ans.append(LLM_answer[0])
    if acc == 1 and "context" not in question:
        if RAG_filter(context, all_paragraphs_chars, p_embed, context, question, options_str, ground_truth, client):
            question = QA["question"]
            options = QA["options"]
            ground_truth = QA["answer"][0]
            print("Question: ", question)
            print("Options: ", options)
            print("Ground Truth: ", ground_truth)
            return QA
            
        else:
            return generate_QA(context,all_paragraphs_chars, p_embed, client, args, attempt + 1, max_attempts)
    else:
        if attempt < max_attempts:
            return generate_QA(context, all_paragraphs_chars, p_embed,client,  args, attempt + 1, max_attempts)
        else:
            return None  

def process_paragraphs(all_paragraphs_tokens, all_paragraphs_chars, p_embed, tokenizer, client, args, sublist_index):

    context = tokenizer.decode(all_paragraphs_tokens[sublist_index])
    QA = generate_QA(context, all_paragraphs_chars, p_embed, client, args)

    if QA:
        return QA,context
    else:
        return process_paragraphs(all_paragraphs_tokens, all_paragraphs_chars, p_embed, tokenizer, client, args, sublist_index + 1)


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
    doc_index = 0
    for i in range(len(data_all)):
        tokenizer = tiktoken.encoding_for_model("gpt-4o")
        context = data_all[i]["text"]
        if len(tokenizer.encode(context)) > max(args.document_length)+1000:
            context = tokenizer.decode(tokenizer.encode(context)[:max(args.document_length)+1000])
        sentences = sent_tokenize(context)
        paragraphs = combine_to_paragraphs(sentences)
        json_objects = [] 
        all_paragraphs_tokens = []
        all_paragraphs_chars = []
        for context in paragraphs:
            all_paragraphs_tokens.append(tokenizer.encode(context))
            all_paragraphs_chars.append(context)
        # print(len(all_paragraphs_tokens))
        total_tokens = sum([len(sublist) for sublist in all_paragraphs_tokens])
        print("Total tokens: ", total_tokens)
        print("[TAIL] Start emebedding source document...")
        paragraph_embeds = [client.embeddings.create(input=[p], model="text-embedding-3-large").data[0].embedding for p in all_paragraphs_chars]
        print("[TAIL] Embedding finished!")
        print("[TAIL] Start generating QA...")
        if total_tokens < max(args.document_length):
            raise ValueError("Total tokens in your raw document are less than the maximum document length specified! \nPlease input a longer document or decrease the maximum document length.")
        for depth in args.depth_list:
            needle_point = int(total_tokens * (depth / 100))
            sublist_index = find_sublist_index_by_position(all_paragraphs_tokens, needle_point)
            
            QA,context = process_paragraphs(all_paragraphs_tokens, all_paragraphs_chars, paragraph_embeds, tokenizer, client, args, sublist_index)
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
                print(f"QA for depth {depth}% generated!")
                
        doc_index += 1

        if os.path.exists(args.QA_save_path):
            with open(args.QA_save_path, 'r') as f:  
                data = json.load(f)
            json_objects = data + json_objects
        with open(args.QA_save_path, 'w', encoding='utf-8') as f:  
            json.dump(json_objects, f, ensure_ascii=False, indent=4)  
        print(f"[TAIL] Succefully saved QA to {args.QA_save_path}!")
    
