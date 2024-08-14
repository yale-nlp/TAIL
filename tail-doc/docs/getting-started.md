# Getting Started
---
## Installation
Install the package from PyPi:
```
# (Recommended) Create a new conda environment.
conda create -n tail python=3.10 -y
conda activate tail

# Install tailtest
pip install tailtest
```
Set your OPENAI_API_KEY as an environment variable.
```
export OPENAI_API_KEY="..."
```

For more details, see [Installation Guide](/userguide).
## Prepare a long document

TAIL generates QAs for benchmark generation based on the document users inputs. Users need to prepare the input document in a JSON file, in the format of `[{"text: YOUR_LONG_TEXT}]` (YOUR_LONG_TEXT is a long string). We prepare a example input document file in `/data/example_input.json`, if you don't have time to collect your own document, you can use it to generate benchmarks.

## Generate your own benchmark

The next step is to set the `document_length` and `depth` for your benchmark. `document_length` means how long the test document in the benchmark will be, while `depth` indicates how deep a question's evidence locates within the test document. For example, setting `document_length` to 8000 and  `depth` to 50 means generating a QA and test document of 8000 tokens, where the evidence for the question is located around the middle of the test document.

Provide path for your long document and path to save your benchmark, specify `document_length` and `depth`,and then run `tail-cli.build` to start benchmark generation! Here's an example: 

```
tail-cli.build --raw_document_path "/data/raw.json" --QA_save_path "/data/QA.json" --document_length 8000 32000 64000 --depth_list 25 50 75
```

## Test LLMs on your benchmark

After generation your benchmark, it's time to evaluate LLMs on it. Input the test model's name and path to the saved benchmark, provide document_length and depth you want to test, TAIL will automatically run the evaluation and store visualizations in `test_result_save_dir`.

```
tail-cli.eval --QA_save_path "/data/QA.json" --test_model_name "gpt-4o" --test_depth_list 25 75 --test_doc_length 8000 32000 --test_result_save_dir /data/result/
```