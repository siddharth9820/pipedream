from datasets import list_datasets, list_metrics, load_dataset, load_metric
from pprint import pprint
import sys
import transformers 
import argparse

TOKENIZERS = {
    "GPT2Fast" : "transformers.GPT2TokenizerFast.from_pretrained('gpt2')" 
} # add more tokenizers here as per need

def get_dataset(name, tokenizer, split='train[:20%]', cache_dir=None, num_workers=8, bptt_len=1024):
    """Get a PyTorch Dataset object for popular LM datasets supported by huggingface
    Args:
        name (str): Name of dataset, eg:- wikitext-2, wikitext-103, openwebtext
        tokenizer (transformers.Tokenizer): a huggingface tokenizer object
        split (str, optional):  Split of the dataset (train, test, val). Defaults to 'train[:20%]'.
        cache_dir (str, optional): The directory where the dataset is stored. Defaults to None (None implies -> ~/.cache/huggingface). On clusters dont leave this as None, change it to the filesystem for heavy I/O.
        num_workers (int, optional): number of processes for preprocessing. Defaults to 8.
        bptt_len (int, optional): Back-propagation through time length i.e. number of words in each training example. Defaults to 1024.
    Returns:
        torch.utils.data.Dataset : A PyTorch Dataset object that can be used with a PyTorch dataloader
    """    
    if name == "wikitext-103":
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split, cache_dir=cache_dir)
    elif name == "wikitext-2":
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split, cache_dir=cache_dir)
    else:
        dataset = load_dataset(name, split=split, cache_dir=cache_dir)
    
    encoded_dataset = dataset.map(lambda example : tokenizer(example['text']), batched=True, num_proc=num_workers, load_from_cache_file=True)
    print(encoded_dataset.column_names)

    block_size = bptt_len
    
    def chunk_examples(examples):
        concat = []
        for input_ids in examples['input_ids']:
            if input_ids:
                concat.extend(input_ids + [tokenizer.eos_token_id])
        chunks = [concat[i:i+block_size] for i in range(0, len(concat), block_size)]
        src = []
        trg = []
        for chunk in chunks:
            if len(chunk) >= block_size:
                src.append(chunk[:-1])
                trg.append(chunk[1:])
        return {"src" : src, "trg" : trg}

    lm_dataset = encoded_dataset.map(
        chunk_examples,
        batched=True,
        num_proc=num_workers,
        load_from_cache_file=True,
        remove_columns=encoded_dataset.column_names,
        batch_size = 2000
    )
    
    lm_dataset.set_format(type='torch', columns=['src', 'trg'])
    return lm_dataset
