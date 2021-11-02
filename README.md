# Tevatron
Tevatron is a simple and efficient toolkit for training and running dense retrievers with deep language models. The toolkit has a modularized design for easy research; a set of command line tools are also provided for fast development and testing. A set of easy-to-use interfaces to Huggingfac's state-of-the-art pre-trained transformers ensures Tevatron's superior performance.

*Tevatron is currently under initial development stage. We will be actively adding new features and API changes may happen. Suggestions, feature requests and PRs are welcomed.*

## Table of Contents
  - [Features](#features)
  - [Installation](#installation)
  - [Datasets](#datasets)
  - [Custom Dataset](#custom-data)
  - [Training (Pytorch)](#training-pytorch)
  - [Training (Pytorch) (Gradient Cache)](#training-pytorch-gradient-cache)
  - [Encoding (Pytorch)](#encoding-pytorch)
  - [Retrieval](#retrieval)
  - [Training (Jax)](#training-jax)
  - [Encoding (Jax)](#encoding-jax)
  - [Evaluation](#evaluation)
  - [Examples](#examples)
  - [Contacts](#contacts)
  

## Features
- Command line interface for dense retriever training/encoding and dense index search.
- Flexible and extendable Pytorch retriever models. 
- Highly efficient Trainer, a subclass of  Huggingface Trainer, that naively support training performance features like mixed precision and distributed data parallel.
- Fast and memory-efficient train/inference data access based on memory mapping with Apache Arrow through Huggingface datasets.
- Jax/Flax training/encoding on TPU

## Installation
First install neural network and similarity search backends, namely Pytorch (or Jax) and FAISS. Check out the official installation guides for [Pytorch](https://pytorch.org/get-started/locally/#start-locally), [Jax]() and [FAISS](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md).

Then install Tevatron with pip,
```bash
pip install tevatron
```

Or typically for develoment/research, clone this repo and install as editable,
```
git https://github.com/texttron/tevatron
cd tevatron
pip install --editable .
```

> Note: The current code base has been tested with, `torch==1.8.2`, `faiss-cpu==1.7.1`, `transformers==4.9.2`, `datasets==1.11.0`

## Datasets
Tevatron self-contained following preprocessed datasets for dense retrieval (via [HuggingFace](https://huggingface.co/Tevatron)). These datasets will be downloaded and tokenized automically during training and encoding by setting `--dataset_name <dataset name>`. See below for details.

- NQ: `Tevatron/wikipedia-nq`
- TriviaQA: `Tevatron/wikipedia-trivia`- 
- WebQuestions: `Tevatron/wikipedia-wq`
- CuratedTREC: `Tevatron/wikipedia-curated`
- SQuAD: `Tevatron/wikipedia-curated`
- MS MARCO: `Tevatron/msmarco-passage`
- SciFact: `Tevatron/scifact`

### Custom data
Tevatron also accept custom dataset. The datasets need to be crafted in the format below:
- Training: `jsonl` file with each line is a training instance,
```
{'query': TEXT_TYPE, 'positives': List[TEXT_TYPE], 'negatives': List[TEXT_TYPE]}
```
- Encoding: `jsonl` file with each line is a piece of text to be encoded,
```
{text_id: "xxx", 'text': TEXT_TYPE}
```
> Here `TEXT_TYPE` can be either raw string or pre-tokenized ids, i.e. `List[int]`. Pre-tokenization has lower data processing latency during training to reduce/eliminate GPU wait.  
 **Note**: the current `reducer` requires text_id of passages/contexts to be convertible to integer, e.g. integer or string of integer.

## Training (Pytorch)
Here we use Natural Questions as example.

To train a simple dense retriever, call the `tevatron.driver.train` module.

Here, we train on a machine with 4xV100 GPU, if the GPU resources are limited for you, please train with gradient cache.
```bash
python -m torch.distributed.launch --nproc_per_node=4 -m tevatron.driver.train \
  --output_dir model_nq \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 32 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
  --negatives_x_device
```

Here we are using our self-contained datasets to train. To use custom dataset, replace `--dataset_name Tevatron/wikipedia-nq` by  `--train_dir <train data dir>`.

>Here we picked `bert-base-uncased` BERT weight from Huggingface Hub and turned on AMP with `--fp16` to speed up training. Several command flags are provided in addition to configure the learned model, e.g. `--add_pooler` which adds an linear projection. A full list command line arguments can be found in `tevatron.arguments`.

## Training (Pytorch) (Gradient Cache)
If the GPU resource is limited, please train with gradient cache (i.e. `--grad_cache`).

```bash
python -m tevatron.driver.train \
  --output_dir model_nq \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 40 \
  --learning_rate 1e-5 \
  --num_train_epochs 40
```

Here we are using our self-contained datasets to train. To use custom dataset, replace `--dataset_name Tevatron/wikipedia-nq` by  `--train_dir <train data dir>`.

>Here we picked `bert-base-uncased` BERT weight from Huggingface Hub and turned on AMP with `--fp16` to speed up training. Several command flags are provided in addition to configure the learned model, e.g. `--add_pooler` which adds an linear projection. A full list command line arguments can be found in `tevatron.arguments`.

<!-- ## Training (Research)
Check out the [run.py](examples/run.py) in examples directory for a fully configurable train/test loop. Typically you will do,
```
from tevatron.modeling import DenseModel
from tevatron.trainer import DenseTrainer as Trainer

...
model = DenseModel.build(
        model_args,
        data_args,
        training_args,
        config=config,
        cache_dir=model_args.cache_dir,
    )
trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )
...
trainer.train()
``` -->


## Encoding (Pytorch)

### Corpus
To encode, call the `tevatron.driver.encode` module. 
For large corpus, split the corpus into shards to parallelize.
In this example, we splits the corpus into 20 shards.

```bash
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path corpus${s}.pt \
  --encode_num_shard 20 \
  --encode_shard_index ${s}
done
```

Here we are using our self-contained datasets to train. To use custom dataset, simply replace `--dataset_name Tevatron/wikipedia-nq-corpus` by  `--encode_in_path <file to encode>`.

<!-- ```
for s in shard1 shar2 shard3
do
python -m tevatron.driver.encode \  
  --output_dir=$OUTDIR \  
  --tokenizer_name $TOK \  
  --config_name $CONFIG \  
  --model_name_or_path $MODEL_DIR \  
  --fp16 \  
  --per_device_eval_batch_size 128 \  
  --encode_in_path $CORPUS_DIR/$s.json \
  --encoded_save_path $ENCODE_DIR/$s.pt
done
``` -->

### Query
```
python -m tevatron.driver.encode \
  --output_dir=temp \
  --model_name_or_path model_nq \
  --fp16 \
  --per_device_eval_batch_size 156 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path query.pt \
  --encode_is_qry
```
Here we are using our self-contained datasets to encode. To use custom dataset, simply replace `--dataset_name Tevatron/wikipedia-nq/test` by  `--encode_in_path <file to encode>`.

## Retrieval
Call the `tevatron.faiss_retriever` module,
```
python -m tevatron.faiss_retriever \  
--query_reps query.pt \  
--passage_reps 'corpus*.pt' \  
--depth  \
--batch_size -1 \
--save_text \
--save_ranking_to rank.txt
```

>Encoded corpus or corpus shards are loaded based on glob pattern matching of argument `--passage_reps`. Argument `--batch_size` controls number of queries passed to the FAISS index each search call and `-1` will pass all queries in one call. Larger batches typically run faster (due to better memory access patterns and hardware utilization.) Setting flag `--save_text` will save the ranking to a txt file with each line being `qid pid score`.

Alternatively paralleize search over the shards.
```bash
INTERMEDIATE_DIR=intermediate
mkdir ${INTERMEDIATE_DIR}
for s in $(seq -f "%02g" 0 19)
do
python -m tevatron.faiss_retriever \  
--query_reps query.pt \  
--passage_reps ${s}.pt \  
--depth 100 \  
--save_ranking_to ${INTERMEDIATE_DIR}/${s}
done

```
Then combine the results using the reducer module,
```bash
python -m tevatron.faiss_retriever.reducer \
--score_dir ${INTERMEDIATE_DIR} \
--query query.pt \
--save_ranking_to rank.txt
```
> Note: currently, `reducer` requires doc/query id being integer.

## Training (Jax)
Tevatron also provides Jax pipeline, which supports training and encoding on TPU.

The following pipeline was tested on TPU VM with v3-8 tpu.

```
python -m tevatron.driver.jax_train \
  --output_dir dpr_nq_jax \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 16 \
  --learning_rate 1e-5 \
  --num_train_epochs 40 \
  --dataloader_num_workers 2
```

## Encoding (Jax)

### Corpus

```
python -m tevatron.driver.jax_encode \
  --output_dir=temp \
  --model_name_or_path dpr_nq_jax \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/wikipedia-nq-corpus \
  --encoded_save_path corpus.pt \
  --dataset_proc_num 64 \
  --p_max_len 156
```

### Query
```
python -m tevatron.driver.jax_encode \
  --output_dir=temp \
  --model_name_or_path dpr_nq_jax \
  --per_device_eval_batch_size 128 \
  --dataset_name Tevatron/wikipedia-nq/test \
  --encoded_save_path query.pt \
  --dataset_proc_num 12 \
  --q_max_len 32 \
  --encode_is_qry
```

## Evaluation
As different datasets have different evaluation pipeline, please see our example documents for details.

## Examples
We currently provide following examples:
- [DPR](https://github.com/texttron/tevatron/tree/main/examples/dpr)
- [MS MARCO passage ranking](https://github.com/texttron/tevatron/tree/main/examples/msmarco-passage-ranking)
- [coCondenser-msmarco](https://github.com/texttron/tevatron/tree/main/examples/coCondenser-marco)
- [SciFact](https://github.com/texttron/tevatron/tree/main/examples/scifact)
- [MrTyDi](https://github.com/texttron/tevatron/tree/main/examples/mrtydi)

## Contacts
If you have a toolkit specific question, feel free to open an issue. 

You can also reach out to us for general comments/suggestions/questions through email.
- Luyu Gao luyug@cs.cmu.edu
- Xueguang Ma x93ma@uwaterloo.ca
