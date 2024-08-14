# User Guide

### Installation

##### Install the package from PyPi:
```
# (Recommended) Create a new conda environment.
conda create -n tail python=3.10 -y
conda activate tail

# Install tailtest
pip install tailtest
```

##### Build from source
```
git clone https://github.com/vllm-project/vllm.git
pip install -e .  
```
Since TAIL relies on [vLLM](https://docs.vllm.ai/en/latest/) for off-line inference, which only runs on Linux and needs to build from source if you are not using CUDA 11.8 or 12.1. So if you encounter problems with vLLM and only want to use TAIL for benchmark generation or test LLMs using API calls, you can clone the repository and modify it accordingly.

###  Long-context Document Preparation

To build a benchmark, firstly users need to prepare a long-context document as input. TAIL allows users to build benchmarks on any given document, such as patents, papers or financial reports. The input text should be constructed to meet the maximum length requirement of the models being evaluated.

For instance, if users want to generate a benchmark with 128k tokens to evaluate LLMs, input texts that are at least 128k tokens long are needed. If the texts users have prepared aren’t long enough to meet the above requirement, we suggest combining multiple shorter texts that are similar to each other. For example

Users should prepare the input document in JSON file, in the format of `[{"text: YOUR_LONG_TEXT_HERE}]`
### Benchmark Generation

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

The next step is to set the `document_length` and `depth` for your benchmark. `document_length` means how long the test document in the benchmark will be, while `depth` indicates how deep a question's evidence locates within the test document. For example, setting `document_length` to 8000 and  `depth` to 50 means generating a QA and test document of 8000 tokens, where the evidence for the question is located around the middle of the test document.

Provide path for your long document and path to save your benchmark, specify `document_length` and `depth`,and then run `tail-cli.build` to start benchmark generation. Here's an example: 
```
tail-cli.build --raw_document_path "/data/raw.json" --QA_save_path "/data/QA.json" --document_length 8000 32000 64000 --depth_list 25 50 75
```
### Model Evaluation & Visualization

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

TAIL supports evaluation for both commercial LLMs using OpenAI API interface and off-line inference using vLLM. 

* For commericial LLMs that compatibale with OpenAI interface, first set your ‘OPENAI_API_KEY’ Environment Variable, you can find a guide [here](https://help.openai.com/en/articles/5112595-best-practices-for-api-key-safety) and set up your base_url if needed. Then simply pass your model name using `--test_model_name "gpt-4o"`.

* For open-source LLMs, pass the your model name in command line using its name or local dir. The list of supported models can be found at [supported models](https://docs.vllm.ai/en/stable/models/supported_models.html#supported-models). e.g.  `--test_model_name "meta-llama/Meta-Llama-3.1-70B"`

Specific the context lengths and depths your want to test. For example, `--test_depth_list 30 80 --test_doc_length 64000 128000` means you want to test the question targetting at depth 30% and 70% of 64k tokens and 128k tokens documents. Be aware that you need to first generate QA for this specific depth and context length, otherwise TAIL will raise an error warning you that it can't find this QA in the QA file you provided. 

Input the test model's name and path to the saved benchmark, provide document_length and depth you want to test, TAIL will automatically run the evaluation and save results in JSON format in `test_result_save_dir`.


### Results visualization

TAIL will automatically visualize the results after the evaluation is done. A line plot and a heatmap showing model's performance across context lengths and depths will be stored in the `result_save_dir` users provide. Here are example visuilaztions for GLM-4-9B-128K: 


<center>
<div style="display: flex; align-items: center;">
    <img src="glm_all.png" alt="TAIL" style="width: 300px; height: 300px; object-fit: cover; margin-right: 30px;">
    <img src="line.jpg" alt="TAIL" style="width: 270px; height: 300px; object-fit: cover; margin-left: 40px;">
</div>
</center>