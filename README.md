

<p align="center">
<img src="img/logo.png" alt="img" width="40%">
<h4 align="center">
Automatic, Easy and Realistic tool for LLM Evaluation
</h4>
</p>

<p align="center">
| <a href="https://yale-nlp.github.io/TAIL/"><b>Documentation</b></a> | <a href="https://aclanthology.org/2024.emnlp-demo.21/"><b>Paper</b></a> | 
</p>

<p align="center">
<img src="img/outline.png" alt="img" width="65%">
</p>

## Introduction
TAIL is an automatic toolkit for creating realistic evaluation
benchmarks and assessing the performance of long-context LLMs. 
With TAIL, users
can customize the building of a long-context,
document-grounded QA benchmark and obtain
visualized performance metrics of evaluated
models.
## Quickstart 
1. install the package from PyPi:
    ```
    # (Recommended) Create a new conda environment.
    conda create -n tail python=3.10 -y
    conda activate tail

    # Install tailtest
    pip install tailtest
    ```
    set yout OPENAI_API_KEY:
    ```
    export OPENAI_API_KEY="..."
    ```
2. Prepare a source document you want to use to generate benchmark and organize in the format of json.
    `[{"text": "Content of your document"}]`

3. Benchmark Generation:

    ```
    tail-cli.build --raw_document_path "/Users/frank/Desktop/code/TAIL/data/law_10.json" --QA_save_path "/Users/frank/Desktop/code/TAIL/data/QA_law.json" --document_length 8000 16000 32000 48000 64000 80000 96000 112000 128000 --depth_list 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95
    ```

4. Model Evaluation & Testing:

    ```
    tail-cli.eval --QA_save_path "/data/QA.json" --test_model_name "gpt-4o" --use_api --test_depth_list 25 75 --test_doc_length 8000 32000 --test_result_save_dir "/data/result/"
    ```

## Citation

```
@inproceedings{gu-etal-2024-tail,
    title = "{TAIL}: A Toolkit for Automatic and Realistic Long-Context Large Language Model Evaluation",
    author = "Gu, Gefei  and
      Zhao, Yilun  and
      Ning, Ruoxi  and
      Zheng, Yanan  and
      Cohan, Arman",
    editor = "Hernandez Farias, Delia Irazu  and
      Hope, Tom  and
      Li, Manling",
    booktitle = "Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2024",
    address = "Miami, Florida, USA",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.emnlp-demo.21",
    pages = "198--208",
    abstract = "As long-context large language models (LLMs) are attracting increasing attention for their ability to handle context windows exceeding 128k tokens, the need for effective evaluation methods for these models becomes critical.Existing evaluation methods, however, fall short: needle-in-a-haystack (NIAH) and its variants are overly simplistic, while creating realistic benchmarks is prohibitively expensive due to extensive human annotation requirements. To bridge this gap, we propose TAIL, an automatic toolkit for creating realistic evaluation benchmarks and assessing the performance of long-context LLMs.With TAIL, users can customize the building of a long-context, document-grounded QA benchmark and obtain visualized performance metrics of evaluated models.TAIL has the advantage of requiring minimal human annotation and generating natural questions based on user-provided long-context documents. We apply TAIL to construct a benchmark encompassing multiple expert domains, such as finance, law, patent, and scientific literature. We then evaluate four state-of-the-art long-context LLMs using this benchmark. Results show that all LLMs experience varyingdegrees of performance degradation as contextlengths increase.",
}
```
