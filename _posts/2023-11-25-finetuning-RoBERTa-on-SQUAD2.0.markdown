---
title: "Finetuning RoBERTa on SQUAD2.0"
date: 2023-11-25
categories: "deep_learning"
---

**Environment**

I used Compute Canada's Narval cluster as the environment for this experiment. Compute Canada is a high-performance computing cluster offering high-end GPU access (NVIDIA A100). It is accessed through the login nodes, but the actual training is performed on the compute nodes that do not have internet access. Thus, both the model and the datasets have to be downloaded. From scratch, the general process is as follow: **1.** ssh to a login node **2.** create a virtual environment and install the necessary packages **3.** download the dataset and the model **4.** preprocess data **5.** schedule training on a compute node (and optional postprocessing). **6.** evaluate model with metrics. All nodes (including compute nodes) have access to a shared filesystem, thus what is downloaded from the login nodes can be accessed by the compute nodes.

**Setup virtual environment**

Compute Canada uses Python virtual environments instead of Anaconda. But beforehand, we need to load the appropriate software modules such as Python. The following steps generally suffice to create a functional environment able to train HuggingFace models, the latter is the de facto hub for open source deep learning at the time of the writing:

* load necessary modules (here `arrow` is needed to manipulate large datasets)

    ```
    module load gcc arrow python/3.10
    ```

* create virtual environment and install packages
    ```
    python -m venv venv
    source venv/bin/activate
    pip install transformers, datasets, ...
    ```

During runtime, there might be errors caused by missing packages. These errors can typically be fixed easily by simply installing the missing package with `pip install`. Conflicting requirements is another story, but luckily they don't happen too often.

**Downloading model and dataset**

With HuggingFace API, downloading model and dataset is very easy, since both the model and datasets are typically uploaded to the HuggingFace Hub ([model hub link][model_hub], [dataset hub link][dataset_hub]), and follow the same streamlined syntax convention. Tokenizers associated with models can also be found in the model hub and use the same syntax as models. For RoBERTa and SQUAD2.0, we can do:

* loading and saving model (from online hub)
    {% highlight python %}
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer

    model = AutoModelForQuestionAnswering.from_pretrained('roberta-base')
    model.save_pretrained(<path_to_saved_model>)
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    tokenizer.save_pretrained(<path_to_saved_tokenizer>)
    {% endhighlight %}

* loading and saving dataset (from online hub)
    {% highlight python %}
    from datasets import load_dataset
    ds = load_dataset("squad_v2")
    ds.save_to_disk(<path_to_saved_dataset>)
    {% endhighlight %}

To load the models locally during training, we can use the corresponding load functions as so:

* loading model locally:
    {% highlight python %}
    model = AutoModelForQuestionAnswering.from_pretrained(<path_to_saved_model>)
    tokenizer = AutoTokenizer.from_pretrained(<path_to_saved_tokenizer>)
    {% endhighlight %}

* loading dataset locally:
    {% highlight python %}
    ds = dataset.load_from_disk(<path_to_saved_dataset>)
    {% endhighlight %}

Notice that the function `from_pretrained()` is used both to load model remotely and locally, it is because the function is overloaded to do so, thus providing a clean API. 

**Preprocessing**

Preprocessing the datasets needs to be done according to model specifications. For [RobertaForQuestionAnswering][roberta_qa_model] (which is what we really instantiate with auto model), we can see that the model input needs arguments `start_positions` and `end_positions`, which the original SQUAD2.0 dataset does not provide. Thus we have to build them ourselves. Moreover, as SQUAD2.0 contains unanswerable questions, we need to consider that as well. For answerable questions, we determine the start and end positions by matching the index of the start/end tokens of the answer within the context. For unanswerable questions, we can simply return (0, 0). The following preprocess function does the job:

{% highlight python %}
def pre_process_roberta_squad2(samples, tokenizer):
    inputs = tokenizer(
        [q.strip() for q in samples['question']], # strip needed to prevent truncation error
        samples['context'],
        max_length= 384,
        truncation='only_second',
        return_offsets_mapping=True,
        padding='max_length'
        )
    
    # code below used to add start and end token positions of the answer within context
    offset_mapping = inputs["offset_mapping"]
    answers = samples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        # If the question is unanswerable, label it (0, 0)
        if len(answer["text"]) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue
        
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the answer within context
        idx = 0
        while sequence_ids[idx] != 1: idx += 1
        context_start = idx
        while sequence_ids[idx] == 1: idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        # otherwise it's the start and end token positions
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char: idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char: idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs
{% endhighlight %}

This preprocess function can be mapped to each data of the dataset, either individually or in batches using the `map()` function:

{% highlight python %}
ds.map(pre_process_roberta_squad2, batched=True, batch_size=32)
{% endhighlight %}

Batched mapping is recommended as it is typically faster on multi-core CPUs. 

**Training**

With the processed dataset and the model in place, we can now train the model. Training can be done in several ways, one way is to loop through the dataset, define the loss function and perform backpropagation ourselves, this way offers more control but requires more code, especially when using parallel training on multiple hardware resources. Another way is by using the [Trainer][trainer] API of HuggingFace, this way offers less control but requires less code, and the trainer is able to handle many of the internal complexities of training. The trainer class requires training arguments, an example is as follow:

{% highlight python %}
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir=<path_to_output_dir>,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir=<path_to_logging_dir>
)
{% endhighlight %}

Then, we can pass model, dataset and the training arguments to the trainer class and start training:

{% highlight python %}
model = AutoModelForQuestionAnswering.from_pretrained(<path_to_saved_model>)
tokenizer = AutoTokenizer.from_pretrained(<path_to_saved_tokenizer>)
ds = load_from_disk(<path_to_saved_dataset>)
ds.map(pre_process_roberta_squad2, batched=True, batch_size=32)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"]
)
trainer.train()
trainer.save_model(<path_to_saved_trained_model>)
{% endhighlight %}

By default, the trainer will save a model checkpoint every 500 data, so even if you don't explicitly save the model or if the training is interrupted, you can still load the (trained) model of the latest checkpoint.

To schedule training on a compute node, one has to submit the training script containing everything from loading module up to training code to a compute node through the SLURM scheduler. You also need to specify the resources needed to be allocated. The more resources, the longer the wait time. A typical script is:

```
#!/bin/bash
#SBATCH --job-name=roberta-squad2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=127000M
#SBATCH --time=03:00:00
#SBATCH --account=<account_name>
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=<email_address>
#SBATCH --output=<output_name>

module load gcc arrow python/3.10
cd <path_to_project_folder>
source venv/bin/activate

python <path_to_training_script.py>
```

**Evaluation**

After training is done, we probably want to evaluate the performance of our model. This sometimes requires some postprocessing as in the case of RobertaForQuestionAnswering and SQUAD2.0. As we are in extractive question answering, the model outputs the logprobabilities of each token being the start/end token corresponding to the answer. If we want to make it human readable, we need to first convert it to text. The following function does the trick:

{% highlight python %}
def postprocess_qa_predictions(predictions, samples):
    processed_predictions = []

    start_logits, end_logits = predictions
    for idx, (start, end) in tqdm(enumerate(zip(start_logits, end_logits))):
        
        offset_mapping = samples[idx]["offset_mapping"]

        start_idx = np.argmax(start)
        end_idx = np.argmax(end)

        if end_idx > start_idx:
            start_char_pos = offset_mapping[start_idx][0]
            end_char_pos = offset_mapping[end_idx][1]

            context = samples[idx]["context"]
            answer = context[start_char_pos:end_char_pos]

        else:
            answer = ""

        processed_predictions.append({
            "qid": samples["id"][idx],
            "question": samples["question"][idx],
            "answer": answer
        })

    return processed_predictions

    processed_predictions = postprocess_qa_predictions(prediction_output.predictions, tokenized_dataset)

    # Save to a JSON file in the official SQUAD format
    squad_format_predictions = {item["qid"] : item["answer"] for item in processed_predictions}        
    with open(<path_to_prediction_json>, "w") as f:
        json.dump(squad_format_predictions, f)

    return processed_predictions
{% endhighlight %}

We can then apply this function to the model predictions as so:

{% highlight python %}
# it is important to put the model to GPU for faster inference if GPU is available, as during training the trainer handles this automatically
model = AutoModelForQuestionAnswering.from_pretrained(<path_to_saved_trained_model>).to("cuda")
tokenizer = AutoTokenizer.from_pretrained(<path_to_saved_tokenizer>)
ds = load_from_disk(<path_to_saved_test_dataset>)
# preprocess and tokenize test dataset the same way as during training
ds.map(pre_process_roberta_squad2, batched=True, batch_size=32)
# instantiate trainer with trained model
trainer = Trainer(model=model)
prediction_output = trainer.predict(test_dataset=ds)
processed_predictions = postprocess_qa_predictions(prediction_output.predictions, ds)
{% endhighlight %}

With predictions in text format conforming to the official [Evaluation Script v2.0][eval_script], we can simply use the evaluation script to get metrics for our model by doing:

```
<path_to_eval_script.py> <path_to_true_answers.json> <path_to_predictions.json> -o <path_to_output_metrics.json>
```

For this experiment, I got: 
{% highlight json %}
{"exact": 73.69662258906763, "f1": 76.73523207875002, "total": 11873, "HasAns_exact": 62.93859649122807, "HasAns_f1": 69.02452943167981, "HasAns_total": 5928, "NoAns_exact": 84.42388561816652, "NoAns_f1": 84.42388561816652, "NoAns_total": 5945}
{% endhighlight %}

Which is not amazing but passable. A performance like this indicates that the training code is working and can be a basis for further optimizations.

[model_hub]:https://huggingface.co/models
[dataset_hub]: https://huggingface.co/datasets 
[roberta_qa_model]: https://huggingface.co/docs/transformers/v4.35.2/en/model_doc/roberta#transformers.RobertaForQuestionAnswering
[trainer]: https://huggingface.co/docs/transformers/main_classes/trainer
[eval_script]: https://rajpurkar.github.io/SQuAD-explorer/




