---
title: "Replicating GPT with nanoGPT"
date: 2024-03-21
categories: "deep_learning"
---

**Why doing this work?**

Large languages models (LLMs) have taken the world by storm, we now have a myriad of models such as GPT-4, Llama 2, Claude, Mistral, Qwen, Grok, etc. all exhibiting astounding capabilities. The more performant models have trillions of parameters and are unwieldy to train or even making inference without a large computing cluster, a resource unavailable to most developers. Thus they act as a blackbox and are offered only as a service, no one but the researchers in the world's most resourceful companies and labs really have an idea of their inner workings. However, if we strip the more sophisticated techniques applies to these large models such as the myriad of scaling techniques and optimizations, large dataset collection and postprocessing, instruction finetuning and reinforcement learning with human feedback (RLHF), and reduce the model size by several orders of magnitude, we can also create a somewhat promising prototype, the GPT, the source of it all. In fact, the root of these models can be traced in the very near past first with the seminal [transformer paper][transformer_paper] in 2017, then its immediate application in the original [GPT paper][GPT_paper] in 2018, [GPT-2 paper][GPT-2_paper] in 2019 and [GPT-3 paper][GPT-3_paper] in 2020 mostly by scaling up, and finally the [InstructGPT paper][InstructGPT_paper] (precursor to ChatGPT) in 2022, introducing RLHF on LLM. After that, OpenAI became CloseAI and hid all the technical details of GPT-4. We can also argue that OpenAI was never sincere in 'opening' their complete research on LLMs, even the [GPT-2 open source code][GPT-2_code] that they released only contains the model code and not the training code or parameters, nor the training dataset. Luckily, the open source community created their own OpenWebText dataset trying to replicate the unreleased WebText dataset used by OpenAI, and the legendary Andrej Karpathy singlehandedly released a [working codebase][nanogpt_repo] to replicate the whole creation of GPT-2. In this post, I will focus on replicating the smallest GPT (124M) model from scratch using Andrej's code on Compute Canada (CC), a high performance computing cluster (HPC) that I'm lucky to have access to as a research master's student. The model size is determined by the number of layers, the number of attention heads and the embedding size while all models use the same architecture (stacked transformer decoders), thus I believe testing on a 'small' model is sufficient. Even this 'small' model requires abundant resources (8 Nvidia A100 GPUs for 4 to 5 days), but at least we can have a peek of the basic principles of this AI revolution right in front of us. The post focuses on the training part and not the model part, as the latter deserves a separate post, and is already extremely well explained in many online posts, most notably by [The Annotated Transformer][the_annotated_transformer] post of Harvard NLP group. The training however is infeasible for most developers without resources and has to be adapted to the specific environment used (CC), and this post can serve as a report of the replication process for people without the resources and for the future me. 

**Dataset**

Everything starts with the dataset, the replication training uses the [OpenWebText][OpenWebText_original] dataset downloaded via the [HuggingFace interface][OpenWebText_huggingface]. The dataset consists roughly of filtered Reddit posts, an example would be: 

```Port-au-Prince, Haiti (CNN) -- Earthquake victims, writhing in pain and grasping at life, watched doctors and nurses walk away from a field hospital Friday night after a Belgian medical team evacuated the area, saying it was concerned about security. <A bunch of more sentence deleted for brevity...> CNN's Justine Redman, Danielle Dellorto and John Bonifield contributed to this report.```

The dataset consists of roughly 38GB of text data from around 8 million documents. Downloading the dataset requires around 55GB cache, and since the default cache location might not have sufficient space, the cache location might need to be moved. Specifically, the environment variables `HF_DATASETS_DIR` and `TIKTOKEN_CACHE_DIR` had to be rewritten to a place with enough disk space. The former is used to store the raw dataset and the latter is used to store the tokenized dataset using the GPT-2 tokenizer.

The dataset is downloaded via the [`data/openwebtext/prepare.py`][prepare_script] script, where it first downloads the dataset through HuggingFace's datasets api, and then tokenizes it using the byte-pair encoding (BPE) tokenizer of GPT-2. Finally, all tokenized texts from train and validation sets are respectively concatenated into 2 gigantic arrays and stored on disk. The resulting training array `train.bin` is around 17GB and validation array `val.bin` is around 8.5MB in size.

**Dataloading**

As the arrays are gigantic and cannot fit in typical memory, Andrej used `np.memmap` to create a memory-map to the array stored on disk. This numpy feature allows us to manipulate the array as if it is stored entirely in memory. A peek of the first 10 elements of the training array can be done with:

{% highlight python %}
m = np.memmap('train.bin', dtype=np.uint16, mode='r')
print("m[:10]: ", m[:10])
{% endhighlight %}

and we get `m[:10]:  [ 8585   262  1772    25 25334  8120    17 43959 44947   318]`, which can be detokenized using GPT-2 tokenizer as the tokens `About the author: OBACK2KENYA is`. The whole training array has length `9035582489` and serves as a gigantic text mine. Andrej created a "poor man's data Loader" in the [train script][train_script] as follow:

{% highlight python %}
# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
{% endhighlight %}

Essentially, we are randomly picking consecutive tokens of a fixed length in this huge text mine following a uniform distribution. As the texts are concatenated from different sources about potentially different topics, this can inevitably create segments where seemingly unrelated texts are stitched together which can potentially hurt our next word prediction task of GPT training. On the other hand, adding such noises can also potentially increase model robustness and more importantly, as we will be training in a distributed fashion, this mechanism bypasses the need for data synchronization among the different training GPUs entirely. It is actually quite genius in this regard. 

**Distributed training**

Single machine training would take too long even for the smallest GPT, thus distributed training is needed and we do so using Pytorch's distributed data parallel (DDP) feature. In Andrej's code, the environment is for LambdaLabs where the environment variables are set by DDP's own `torchrun`, it took me quite some time to figure out how to adapt to the CC environment as the latter uses the Slurm workload manager to setup distributed machines. The difference in code can be seen side by side below:

<div style="display: flex; justify-content: space-between;">
  <pre style="background: #f0f0f0; width: 48%; padding: 10px;">
<code style="color: blue;">ORIGINAL</code>
<code style="color: red;">- init_process_group(backend=backend)</code>
<code style="color: red;">- ddp_rank = int(os.environ['RANK'])</code>
<code style="color: red;">- ddp_local_rank = int(os.environ['LOCAL_RANK'])</code>
<code style="color: red;">- ddp_world_size = int(os.environ['WORLD_SIZE'])</code>
<code style="color: red;">- device = f'cuda:{ddp_local_rank}'</code>
<code>  torch.cuda.set_device(device)</code>
<code>  master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.</code>
<code>  seed_offset = ddp_rank # each process gets a different seed</code>
<code style="color: gray;">  # world_size number of processes will be training simultaneously, so we can scale</code>
<code style="color: gray;">  # down the desired gradient accumulation iterations per process proportionally</code>
<code>  assert gradient_accumulation_steps % ddp_world_size == 0</code>
<code>  gradient_accumulation_steps //= ddp_world_size</code>
  </pre>
  <pre style="background: #f0f0f0; width: 48%; padding: 10px;">
<code style="color: blue;">ADAPTED</code>
<code style="color: green;">+ ngpus_per_node = torch.cuda.device_count()</code>
<code style="color: green;">+ ddp_local_rank = int(os.environ.get("SLURM_LOCALID"))</code>
<code style="color: green;">+ ddp_rank = int(os.environ.get("SLURM_NODEID")) * ngpus_per_node + ddp_local_rank</code>
<code style="color: green;">+ ddp_world_size = int(os.environ.get("SLURM_NTASKS_PER_NODE")) * int(os.environ.get("SLURM_JOB_NUM_NODES"))</code>
<code style="color: green;">+ device = ddp_local_rank</code>
<code>  torch.cuda.set_device(device)</code>
<code style="color: green;">+ init_process_group(init_method=f"tcp://{os.environ.get('MAIN_NODE')}:3456", world_size=ddp_world_size, rank=ddp_rank, backend=backend)</code>
<code>  master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.</code>
<code>  seed_offset = ddp_rank # each process gets a different seed</code>
<code style="color: gray;">  # world_size number of processes will be training simultaneously, so we can scale</code>
<code style="color: gray;">  # down the desired gradient accumulation iterations per process proportionally</code>
<code>  assert gradient_accumulation_steps % ddp_world_size == 0</code>
<code>  gradient_accumulation_steps //= ddp_world_size</code>
  </pre>
</div>

The key difference here is that with `torchrun`, the required environment variables such as `RANK`, `LOCAL_RANK` and `WORLD_SIZE` are input to the system on each machine: 

```
Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
```

while with a SLURM system, we can use the `srun` command to simultaneously launch a distributed program on different machines. The corresponding names of the environment variables also change, for example, `SLURM_LOCALID` will correspond to `RANK`. In the `init_process_group(...)` line, launching by `torchrun` does not need additional parameters such as `init_method` or `world_size`, since they are already specified on the command line of each machine, while launching by `srun` requires these parameters in order to coordinate the distributed training automatically. The full `srun` script using 8 NVIDIA a100 GPUs on the Narval cluster is as follow: 

```
#!/bin/bash
#SBATCH --account=<account name>
#SBATCH --nodes=2
#SBATCH --gpus-per-node=a100:4
#SBATCH --tasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=498G
#SBATCH --time=5-0

export MAIN_NODE=$(hostname)

module load gcc arrow python

cd <path to nanoGPT repository>
source venv/bin/activate

srun python train.py
```
Here, the address of the landing compute node will be registered via the environment variable `MAIN_NODE` as the main process, and `init_process_group` thus knows which machine to use to register outputs. As the number of requested resources is high (Narval only has 64 compute nodes with 4 a100 GPUs and I was requesting 2 of them), the queue time is significant and can wait up to 2 days before launching. During training, we can `ssh` into the compute nodes and enter `nvidia-smi` command to monitor GPU resource utilization. All 8 GPUs had 100% usage rate when I checked. On the other hand, Andrej conveniently provided a benchmarking function [`estimate_mfu`][estimate_mfu] as a model utility, where it estimates the percentage of GPU utilization in terms of floating point operations per second (FLOPS). At 8 GPUs and using default training parameters, the mfu measure stabilizes at around 34%, with each iteration taking around 480ms. I also tried briefly with 2 GPUs, in which the mfu measure was around 39% and each iteration taking around 1680ms. It is reasonable to assume that more GPUs can induce heavier overhead in terms of cross-GPU and cross-node communications, but my practical experience suggests that such overhead is not significant (yet), and more GPUs are worth it. It may become a bottleneck with a higher number of GPUs, but I didn't have the resources to test. During my initial training I also encountered a memory issue where all my initial 512GB(256GB/node) memory were used up and `oom_kill` was forced on the main process. I could not identify the cause of the error as there was no apparent memory leak, but I could resume training from a saved checkpoint using `srun python train.py --init_from=resume`, this time, I requested the whole node's memory (498GB/node) but only 361.94GB was used. 

**Training result sampling**

I modified the original code to also save a checkpoint at iterations 10000, 20000 and at each 50000th iteration. The resulting models are all around 1.4GB in size. Let's see what they can produce at different stages of training, using the prompt `Joe Biden and Donald Trump` as a starting point to generate a sample of 50 tokens using near greedy decoding. As the trained model samples from a multinomial distribution based on the softmax output of the final logits of the model divided by the temperature, we can increase the sharpness of the distribution (increasing sampling greediness) by changing the temperature to a small value (`0.01`). The results are as follow:

---
<h4>iteration 10000:</h4>

Joe Biden and Donald Trump Jr. have been in the White House since the election.

The president-elect has been in the White House since the election.

The president-elect has been in the White House since the election.

The president-elect

---
<h4>iteration 20000:</h4>
Joe Biden and Donald Trump.

The president-elect has been a vocal critic of the Obama administration’s war on drugs, and has called for a ban on all drugs in the U.S.

“I’m not going to let

---
<h4>iteration 50000:</h4>
Joe Biden and Donald Trump.

The Republican nominee has been a frequent critic of the Obama administration’s policies, and has called for a “full and complete shutdown of Muslims entering the United States.”

Trump has also called for a “

---
<h4>iteration 100000:</h4>
Joe Biden and Donald Trump.

The former president and former vice president are set to meet with the president and his family at the White House on Tuesday.

The two men will be joined by the president and his family for a private dinner at the White House.

---
<h4>iteration 150000:</h4>
Joe Biden and Donald Trump are both running for president.

The two men have been in touch with the media, and have been in touch with the public.

The two men have been in touch with the media, and have been in touch with the public.

---
<h4>iteration 200000:</h4>
Joe Biden and Donald Trump, who are both running for president, have been the most vocal in their opposition to the Trans-Pacific Partnership, which would have created a new trade deal with the United States.

The TPP would have created a new trade deal with the United

---
<h4>iteration 250000:</h4>
Joe Biden and Donald Trump.

The former vice president and 2016 presidential candidate has been a vocal supporter of the Republican Party’s presidential nominee, Donald Trump.

Biden has been a vocal supporter of the Republican Party’s presidential nominee, Donald Trump

---
<h4>iteration 300000:</h4>
Joe Biden and Donald Trump.

The two men are both former presidential candidates, and both have been in the running for the White House for a while.

Biden, who is running for the White House in 2016, has been a vocal critic of Trump�

---
<h4>iteration 350000:</h4>
Joe Biden and Donald Trump.

The two men have been in the spotlight since the election, with the former vice president calling Trump a “moron” and the latter calling him “a moron.”

The two men have been in

---
<h4>iteration 400000:</h4>
Joe Biden and Donald Trump.

The two men have been in the news for their controversial comments about women, but they also have been in the news for their support of the Trans-Pacific Partnership (TPP).

The TPP is a trade deal that would give the

---
<h4>iteration 450000:</h4>
Joe Biden and Donald Trump.

The former vice president, who is expected to be the Democratic nominee for president in 2016, has been a vocal critic of the president’s policies.

“I think he’s a very, very dangerous man

---
<h4>iteration 500000:</h4>
Joe Biden and Donald Trump.

The former vice president, who is now a presidential candidate, has been a vocal critic of the president’s policies, including his decision to pull the United States out of the Paris climate accord.

“I think it

---
<h4>iteration 550000:</h4>
Joe Biden and Donald Trump.

The two men have been in the news for a while now, but they’ve been in the news for a while now, and they’ve been in the news for a while, and they’ve been in

---
<h4>iteration 600000:</h4>
Joe Biden and Donald Trump.

The former vice president, who is running for the Democratic nomination, said he was “very proud” of the speech.

“I think it’s a great speech,” Biden said. “

---
Although this experiment is anecdotal, we can still get some observations about the GPT model training:

1. It does not take long for the model to learn the syntax and grammar to a reasonable level, as the model at lower iterations can still generate coherent sentences.

2. Dataset quality affects the quality of the model a lot. In this example, the most common continuation for `Joe Biden and Donald Trump` is the point `.`, and it might be because there aren't as many sentences starting with `Joe Biden and Donald Trump ...` as sentences ending with `... Joe Biden and Donald Trump.` in the training set. 

3. There are potential risks of cyclic generation such as at iterations 10000, 150000 and 550000, where certain sentences or words are repeated. This phenomenon [can still be observed in ChatGPT][ChatGPT_repeating] even after instruction finetuning and RLHF. 






[transformer_paper]: https://arxiv.org/abs/1706.03762
[GPT_paper]: https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf
[GPT-2_paper]: https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
[GPT-3_paper]: https://arxiv.org/abs/2005.14165
[InstructGPT_paper]: https://arxiv.org/abs/2203.02155
[GPT-2_code]: https://github.com/openai/gpt-2
[nanogpt_repo]: https://github.com/karpathy/nanoGPT
[the_annotated_transformer]: https://nlp.seas.harvard.edu/annotated-transformer/
[OpenWebText_original]: https://skylion007.github.io/OpenWebTextCorpus/
[OpenWebText_huggingface]: https://huggingface.co/datasets/Skylion007/openwebtext
[prepare_script]: https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py
[train_script]: https://github.com/karpathy/nanoGPT/blob/master/train.py
[estimate_mfu]: https://github.com/karpathy/nanoGPT/blob/325be85d9be8c81b436728a420e85796c57dba7e/model.py#L289
[chatGPT_repeating]: https://www.thedailybeast.com/openais-chatgpt-went-completely-off-the-rails-for-hours