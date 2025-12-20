import torch as t
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, OPTForCausalLM, GPTNeoXForCausalLM, AutoModelForCausalLM
import argparse
import pandas as pd
from tqdm import tqdm
import os
import configparser
from nnsight import LanguageModel

DEBUG = False
if DEBUG:
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    tracer_kwargs = {'scan': False, 'validate': False}

config = configparser.ConfigParser()
config.read('config.ini')

def load_model(model_name, device='remote'):
    print(f"Loading model {model_name}...")
    weights_directory = config[model_name]['weights_directory']
    if device == 'remote':
        model = LanguageModel(weights_directory)
    else:
        model = LanguageModel(weights_directory, torch_dtype=t.bfloat16, device_map="auto")
    return model

def load_statements(dataset_name):
    """
    Load statements from csv file, return list of strings.
    """
    dataset = pd.read_csv(f"datasets/{dataset_name}.csv")
    statements = dataset['statement'].tolist()
    labels = dataset['label'].tolist()
    if dataset_name == 'cities':
        prompt = """
            The city of Tokyo is in Japan. This statement is: TRUE
            The city of Hanoi is in Poland. This statement is: FALSE
            The city of Berlin is not in Poland. This statement is: TRUE
            The city of Kyoto is not in Japan. This statement is: FALSE
        """
    elif dataset_name == 'neg_cities':
        prompt = """
            The city of Berlin is not in Poland. This statement is: TRUE
            The city of Kyoto is not in Japan. This statement is: FALSE
        """
    elif dataset_name == 'sp_en_trans':
        prompt = """\
        The Spanish word 'jirafa' means 'giraffe'. This statement is: TRUE
        The Spanish word 'aire' means 'silver'. This statement is: FALSE
        """
    elif dataset_name == 'neg_sp_en_trans':
        prompt = """\
        The Spanish word 'escribir' does not mean 'cat'. This statement is: TRUE
        The Spanish word 'gato' does not mean 'cat'. This statement is: FALSE
        """
    elif dataset_name == 'larger_than':
        prompt = """\
        Twenty-one is larger than twelve. This statement is: TRUE
        One is larger than eight. This statement is: FALSE
        """
    elif dataset_name == 'smaller_than':
        prompt = """\
        Five is smaller than eight. This statement is: TRUE
        Seventy is smaller than fifty. This statement is: FALSE
        """
    statements = [prompt + s + ' This statement is:' for s in statements]
    return statements, labels

def get_responses(statements, model, labels):
    """
    Gets responses of the model for a given statement
    """
    with t.no_grad():
        with model.trace(statements):
            # save the model's output logits
            logits = model.output.logits.save()
    responses = model.tokenizer.batch_decode(logits.argmax(dim=-1)[:, -1])
    return [(l, r) for l, r in zip(labels, responses)]

if __name__ == "__main__":
    """
    read statements from dataset, record activations in given layers, and save to specified files
    """
    parser = argparse.ArgumentParser(description="Generate activations for statements in a dataset")
    parser.add_argument("--model", default="llama-3.2-3B",
                        help="Size of the model to use. Options are 7B or 30B")
    parser.add_argument("--layers", nargs='+', type=int,
                        help="Layers to save embeddings from")
    parser.add_argument("--datasets", nargs='+',
                        help="Names of datasets, without .csv extension")
    parser.add_argument("--output_dir", default="reponses",
                        help="Directory to save activations to")
    parser.add_argument("--noperiod", action="store_true", default=False,
                        help="Set flag if you don't want to add a period to the end of each statement")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.datasets is None:
        args.datasets = ['cities', 'neg_cities', 'larger_than', 'smaller_than', 'sp_en_trans', 'neg_sp_en_trans']
        # args.datasets = ['larger_than']
        # args.datasets = ['likely']

    t.set_grad_enabled(False)
    model = load_model(args.model, args.device)
    rows = []
    for dataset in args.datasets:
        statements, labels = load_statements(dataset)
        if args.noperiod:
            statements = [statement[:-1] for statement in statements]
        save_dir = os.path.join(f"{args.output_dir}", args.model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if args.noperiod:
            save_dir = os.path.join(save_dir, "noperiod")
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for idx in tqdm(range(0, len(statements), 25)):
            responses_with_labels = get_responses(statements[idx:idx + 25], model, labels[idx:idx + 25])
            for label, response in responses_with_labels:
                rows.append({
                    "model": args.model,
                    "dataset": dataset,
                    "label": label,
                    "response": response,
                })
    df = pd.DataFrame(rows)
    df.to_csv(f"{save_dir}/repsonses.csv", index=False)
