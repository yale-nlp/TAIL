## Welcome to TAIL!

<center>
<img src="img/TAIL.png" alt="TAIL" width="400">
</center>
<center>
Automatic, Easy and Realistic tool for LLM Evaluation

</center>
<center>
<a href="getting-started/" class="btn btn-primary">Getting Started</a>
<a href="userguide/" class="btn btn-primary">User Guide</a>
</center>

<div class="pt-2 pb-4 px-4 my-4 bg-body-tertiary rounded-3">
<h3 class="text-center">Features</h3>

<div class="row">
  <div class="col-sm-6">
    <div class="card mb-4">
      <div class="card-body">
        <h3 class="card-title">Easy to customize</h3>
        <p class="card-text">
            TAIL allows you to generate test examples of any context length, and questions at any depth you want.
        </p>
      </div>
    </div>
  </div>
  <div class="col-sm-6">
    <div class="card mb-4">
      <div class="card-body">
        <h3 class="card-title">Realistic and Natural QA</h3>
        <p class="card-text">
            Unlike the 
            <a href="https://github.com/gkamradt/LLMTest_NeedleInAHaystack">needle-in-a-haystack</a> test, TAIL generate questions based on infomations from your own document, instead of inserting a piece of new infomation, which 
        </p>
      </div>
    </div>
  </div>
</div>

<div class="row">
  <div class="col-sm-6">
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">Multi check to ensure quality</h3>
        <p class="card-text">
            TAIL has multiple quality checking modules to ensure high quality QAs. 
        </p>
      </div>
    </div>
  </div>
  <div class="col-sm-6">
    <div class="card">
      <div class="card-body">
        <h3 class="card-title">Ready-to-use evaluation module</h3>
        <p class="card-text">
            TAIL integrates a out-of-the-box evaluation module that enables users to easily evaluate commercial LLMs via API calls and open-source LLMs via <a href="https://docs.vllm.ai/en/latest/">vLLM</a> on their generated benchmarks.
        </p>
      </div>
    </div>
  </div>
</div>
</div>
TAIL is a toolkit for automatically creating realistic evaluation
benchmarks and assessing the performance
of long-context LLMs. With TAIL, users
can customize the generation of natural and reliable QAs at specific depths to construct a long-context,
document-grounded QA benchmark and obtain
visualized performance metrics of evaluated
models. TAIL has the advantage of requiring minimal human annotation and generating natural questions based on user-provided
long-context documents. 