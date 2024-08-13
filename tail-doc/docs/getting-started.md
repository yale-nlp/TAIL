# Getting Started
---
### Installation
Install the package from PyPi:
```
# (Recommended) Create a new conda environment.
conda create -n tail python=3.10 -y
conda activate tail

# Install tailtest
pip install tailtest
```


### Generate your own benchmark

```
tail-cli.build --
```

### Test LLMs on your benchmark

```
tail-cli.eval --
```