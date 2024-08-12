# TAIL: A Toolkit for Automatic and Realistic Long-Context Large Language Model Evaluation

![img](img/outline.png)
## Introduction
TAIL can generate high-quality benchmark on any source documents. First, users should prepare a long-context document as input and provide expect lengths and depth for their questions. 


## Usage

We provide a detailed documentation at [https://yale-nlp.github.io/TAIL/](https://yale-nlp.github.io/TAIL/). 

## Quickstart 

1. Prepare a source document you want to use to generate benchmark and organize in the format of json.
    [{"text": "Content of your document."}]

2. Benchmark Generation:

    ```
    usage: tail-cli.build [--raw_document_path <path>] [--document_length <value>] 
        [--depth_list <value>] [--QA_save_path <path>] 
        [--gen_QA_model_name <name>]

    Options:
    --raw_document_path <path>    Path to the document your prepared.
    --document_length <value>     Expected token lengths for benchmarking. Multiple values can be provided.
    --depth_list <value>          Expected depths for your questions. Multiple values can be provided.
    --QA_save_path <path>         Path to save the generated QA dataset.
    --gen_QA_model_name <name>    Model name for generating the QA (default: gpt-4o).
    ```

3. Model Evaluation & Testing:

    ```
    usage: tail-cli.eval [--QA_save_path <path>] [--test_model_name <name>] 
                      [--test_depth_list <value>] [--test_doc_length <value>] 
                      [--test_result_save_dir <path>]

    Options:
    --QA_save_path <path>          Path to the saved QA dataset.
    --test_model_name <name>       Test model name (default: gpt-4o).
    --test_depth_list <value>      Depths you want to test. Multiple values can be provided (default: 30 70).
    --test_doc_length <value>      Token lengths you want to test. Multiple values can be provided (default: 8000).
    --test_result_save_dir <path>  Path to save the test results and visualizations.  
    ```

## Citation
