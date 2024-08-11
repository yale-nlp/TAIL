import json
from openai import OpenAI  
import numpy as np
import os
import pandas as pd

def get_embedding(text, model="text-embedding-3-large"):
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def find_top_paragraphs(question_embed, paragraph_embeds, paragraphs, top_n=5):
    similarities = [cosine_similarity(question_embed, p_embed) for p_embed in paragraph_embeds]
    top_indices = np.argsort(similarities)[::-1][:top_n]
    return [(paragraphs[i]) for i in top_indices]

def main(args):
    with open(args.QA_save_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        RAG_result = []
        Retrieved_Para = []
        url_emb_list = {}
        Context = []
        LLM_result_1 = []
        LLM_result_2 = []
        Paras = []
        for item in data:
            context.append(item["context"])
            QA = item['QA']
            depth = item["depth"]
            paragraphs = item['whole_document']
            url = item['url']
            print(f"Checking Question: {question}")
            true_ans = item["answer"]
                       
            question_embeds = get_embedding(question)
            paragraph_embeds = []
            
            # No need to embed the same document multiple times
            if url not in url_emb_list:  
                for para in paragraphs:
                    paragraph_embeds.append(get_embedding(para))
                url_emb_list[url] = paragraph_embeds 
            else:
                paragraph_embeds = url_emb_list[url]
            
            top_paragraphs = find_top_paragraphs(question_embeds, paragraph_embeds, paragraphs)
            possible_para = len(top_paragraphs)

            # remove original question from top_paragraphs
            if item["Context"] in top_paragraphs:
                top_paragraphs.remove(item["Context"])
                possible_para -= 1


            options = item['options']  
            options_str = ' '.join([f"{key}: {value}" for key, value in options.items()])
            
            # 连接一个列表里的所有字符串
            q = ' '.join([str(elem) for elem in top_paragraphs])
            Paras.append(q)
            flag = True
            # for q in top_paragraphs:
            #     # print("===================")
            completion = client.chat.completions.create(  
                model="gpt-3.5-turbo",  
                messages=[  
                    {"role": "system", "content": "I will give you a content and a quesiton. Give me your answer based on the content and list your chain of thoughts.\
                        If the content is not enough, please just point out no enough information."},  
                    {"role": "user", "content": f"Here is the content: {q}"},
                    {"role": "user", "content": f"Here is the question: {question} Options: {options_str}"}  
                ]  
            )  
            LLM_answer_COT = completion.choices[0].message.content
            LLM_result_1.append(LLM_answer_COT)
            # print(LLM_answer_COT)
            if "no enough information" or "Not enough information" in LLM_answer_COT:
                #print("Incorrect")
                possible_para -= 1
                flag = False
                LLM_result_2.append("None")
                
            completion2 = client.chat.completions.create(  
                model="gpt-3.5-turbo",  
                messages=[  
                    {"role": "system", "content": "I will give you a person's answer and chain of thoughts on a quesion based on a content. \
                                                Also I will provide the true answer to this quesiton. \
                        If the person thinks no enough information, please judge it as incorrect. \
                                                Please verify whether its answer and its chain of thoughts is right and don't explain. \
                                                "},  
                    {"role": "user", "content": f"Here is the question: {question} Options: {options_str}"} ,
                    {"role": "user", "content": f"Here is the content: {q} and true answer: {true_ans}"},
                    {"role": "user", "content": f"Here is the person's answer and chain of thoughts: {LLM_answer_COT}"}  
                ]  
            )  
            
            LLM_judgement = completion2.choices[0].message.content
            LLM_result_2.append(LLM_judgement)
            if "incorrect" in completion2.choices[0].message.content:
                possible_para -= 1
                flag = False
            
            if flag == False:
                RAG_result.append("PASS")
            else:
                RAG_result.append("FAIL")
                print(f"{question} in {url} at {depth} is abiguis!")

            result = []
            # for i in range(len(top_paragraphs)):
            #     dict = {"text": top_paragraphs[i], "LLM_Judge_1": LLM_result_1[i],"LLM_Judge_2": LLM_result_2[i]}
            #     result.append(dict)
            # Retrieved_Para.append(result)

        print(RAG_result)
        print(Retrieved_Para)

        out = []
        print(len(RAG_result))
        print(len(Paras))
        print(len(LLM_result_1))
        for i in range(len(RAG_result)):
            # dict = {"answer": RAG_result[i],"context": Context[i], "retrieved": Retrieved_Para[i]}
            dict = {"answer": RAG_result[i],"context": Context[i], "Paras": Paras[i], "LLM_Judge_1": LLM_result_1[i], "LLM_Judge_2": LLM_result_2[i]}
            out.append(dict)
        print(out)
        df_system = pd.DataFrame(
            {
                "output": out
            }
        )

        # Create an id column to match the base dataset.
        df_system["id"] = df_system.index
        df_system["correct"] = df_system["output"].apply(lambda x: 1 if x["answer"] == "PASS" else 0)
        project.upload_system(df_system, name="Ziva.trangra", id_column="id", output_column="output")

