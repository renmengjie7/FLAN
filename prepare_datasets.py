from flan import task_splits
import tensorflow as tf
import seqio
from multiprocessing import Pool
import json

def process_prompt(prompt, response):
    '''Process a FLAN prompt to remove extra tokens'''
    prompt,response = list(map(lambda x:x.numpy().decode(),[prompt,response]))
    return [prompt.strip(), response.strip()]

def process_single_example(example):
    '''Converts a single prompt to A Decoder only format'''
    inputs_pretokenized = example['inputs_pretokenized']
    targets_pretokenized = example['targets_pretokenized']
    return tf.py_function(
        process_prompt,
        inp=[inputs_pretokenized,targets_pretokenized],
        Tout=[tf.string, tf.string],
    )

def prepare_task(task, split=None):
    '''Saves a single task data'''
    dataset = seqio.get_mixture_or_task(task).get_dataset(
        split=split,
        sequence_length={'inputs':4096,'targets':4096} # Extranous length to capture all data
    )
    dataset = dataset.map(process_single_example)
    
    with open(f'./FLANdata/{task}_{split}.jsonl', 'w') as f:
        for (p, r) in dataset.as_numpy_iterator():
            f.write(
                json.dumps({
                "inputs": p.decode(),
                "targets": r.decode(),
                "task": task,
            }))
            f.write("\n")

if __name__ == '__main__':
    splits = task_splits.generate_superglue_num_tasks_ablation()
    print("SPLITS: ", splits)
    task_split = splits[-1]

    all_tasks = list(task_split.train_tasks)
    all_tasks += task_split.test_tasks
    print("TASKS: ", all_tasks)

    all_tasks.remove("newsroom_10templates") # Manual download
    all_tasks.remove("winogrande_10templates") # Not working
    all_tasks.remove("xsum_10templates") # Manual download but not working
   
    for split in ["train", "validation", "test"]:
        with Pool(36) as p:
            p.starmap(prepare_task, [(t, split) for t in all_tasks])

