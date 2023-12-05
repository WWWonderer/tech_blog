---
title: "Finetuning T5"
date: 2023-12-03
categories: "deep_learning"
---

**Overview**

T5, or the Text-To-Text Transfer Transformer, is a popular all-purpose NLP model proposed by Google in 2020. It is important to play with it due to its polyvalence in textual processing. However, simple and clean codebase for model training are often hard to come by. Luckily, I found [this repository][source1] (which is in turn based on [this repository][source2]) on T5 training. 

**Environment**

The training environment is Compute Canada. It is a high-performance computing cluster providing high-end hardware such as NVIDIA A100 GPUs. Python 3.10 is used along with the packages found in `requirements.txt`. Any missing package must be installed during runtime if applicable.

**Dataset**

The training dataset can be accessed [here][dataset_link]. It is a dataset formed by selecting causal questions among different popular question answering datasets such as NQ, MSMARCO, etc. The original datasets' formats are also edited to be uniform with columns `| id | question | question_processed | context | context_processed | answer | answer_processed |`, so the same metric script can be used to evaluate all of them.

**Training**

Training uses HuggingFace API, with the general process being:

* instantiate HuggingFace model
* instantiate HuggingFace tokenizer
* instantiate HuggingFace trainer with training arguments

For more detailed code breakdown of the above you can refer to the previous [RoBERTa training post][roberta_post] which is similar, or to go to the linked repositories in the overview section and look at source code directly. However, there are certain functions of interest in the preprocessing stage:

* The *build_input()* function concatenates the question and the context for extractive QA. A simple newline is enough. Notice how the batch being a list utilizes the batched processing capability of `dataset.map()`, speeding up the process time.

    {% highlight python %}
    # concatenate question+context with \\n as a separator
    def build_input(batch):
        input_ = [
            (question + " \\n " + context if context is not None else question)
            for question, context in zip(
                batch["question_processed"], batch["context_processed"]
            )
        ]
        batch["input"] = input_
        return batch
    {% endhighlight %}

* Pay attention to the `encoded_inputs["labels"]` part in *tokenize_function_train*. Notice how `pad_token_id` is encoded by `-100`. This is a convention in HuggingFace models used to signify to the optimizer to not calculate loss for padding, which is consistent with Pytorch's [cross-entropy loss][crossentropy] implementation where `-100` is the default ignore index.

    {% highlight python %}
    def train_unifiedqa(args: argparse.ArgumentParser):
    set_seed(args.seed)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer)
    model = T5ForConditionalGeneration.from_pretrained(args.checkpoint)

    def tokenize_function_train(batches):
        encoded_inputs = tokenizer(
            batches["input"],
            max_length=args.source_length, # default = 2048
            padding="max_length",
            truncation=True,
        )
        encoded_answers = tokenizer(
            batches["answer"],
            max_length=args.target_length, # default = 2048
            padding="max_length",
            truncation=True,
        )
        encoded_inputs["labels"] = [
            [(a if a != tokenizer.pad_token_id else -100) for a in ans] # -100 signifies to optimizer to not calculate loss
            for ans in encoded_answers["input_ids"]
        ]
        return encoded_inputs
    {% endhighlight %}

* The training then can be started with a `Seq2SeqTrainer` and `Seq2SeqTrainingArguments`. 

**Training with accelerate**

[Accelerate][accelerate] is a library to enable straightforward distributed and mixed-precision training with PyTorch. The HuggingFace `Trainer` class is by default supporting accelerate, but you need to create a yaml configuration file and launch the script via `accelerate launch` instead of simply `python`. To create the configuration file, do:

```
accelerate config --config_file CONFIG_FILE(str) 
```
This will initiate a few interactive questions such as `In which compute environment are you running?` which you can respond to in the command line. The generated file will be saved to the location of the `CONFIG_FILE` argument. An example of finished configuration file is:

{% highlight yaml%}
compute_environment: LOCAL_MACHINE
debug: true
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: fp16
num_machines: 1
num_processes: 2
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
{% endhighlight %}

You can then launch your script with accelerate using:

```
accelerate launch --config_file CONFIG_FILE(str) <script.py>
```

In Compute Canada, after allocating your resources with `sbatch`, accelerate is able to take care of the distributed training automatically without you having to explicitly set up classes such as `DistributedDataParallel`. To train T5 for 3 epochs on the full SQUAD2.0 dataset using accelerate and 2 GPUs, the training time is cut down significantly from around 50h to only around 20h. 

**Evaluation**

For evaluation, we can predict through trainer, or we can explicitly instantiate the finetuned model, and loop through the test data using a `DataLoader` as so:

{% highlight python %}
data = ... # load data and preprocess the same way as during training
model = T5ForConditionalGeneration.from_pretrained(<path to model>).to(<device>) 
tokenizer = T5Tokenizer.from_pretrained(<path to tokenizer>)
loader = DataLoader(data, batch_size=<batch size>)
predictions = []
for batch in tqdm(loader):
    batch_predictions = run_model(batch['input'], model, tokenizer)
    predictions.extend(batch_predictions)
{% endhighlight %}

where `run_model` is defined as:

{% highlight python %}
def run_model(batch, model, tokenizer):
    encoded_inputs = tokenizer(batch, max_length=<max_length>, padding='max_length', truncation=True, return_tensors='pt').to(<device>)
    res = model.generate(**encoded_inputs, max_length=<max_length>)
    return tokenizer.batch_decode(res, skip_special_tokens=True)
{% endhighlight %}

Then, having the `predictions` and the true `answers`, we can define a metric function to perform different metrics (such as em, f1) and output them in a specified format (such as json). A sample prediction is:

{% highlight json %}
{
    "checkpoint": "allenai/unifiedqa-v2-t5-base-1363200",
    "metrics": {
        "f1": 0.06755228987622684,
        "em": 0.0,
        "rougeL_precision": 0.2904461279461279,
        "rougeL_recall": 0.03622366819767625,
        "rougeL_f1": 0.06117838431353354,
    }
}
{% endhighlight %}

[source1]: https://github.com/andreaschandra/causalqa
[source2]: https://github.com/webis-de/coling22-benchmark-for-causal-question-answering
[dataset_link]: https://zenodo.org/records/7186761#.Y3DxncdBy5c
[roberta_post]: https://wwwonderer.github.io/tech_blog/deep_learning/2023/11/25/finetuning-RoBERTa-on-SQUAD2.0.html
[crossentropy]: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
[accelerate]: https://huggingface.co/docs/accelerate/index