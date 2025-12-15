import torch as t
import pandas as pd
import os
from tqdm import tqdm
from utils import collect_acts
from generate_acts import load_model
from probes import LRProbe, MMProbe, CCSProbe
import plotly.express as px
import json
import argparse
import configparser

DEBUG = False
if DEBUG:
    tracer_kwargs = {'scan': True, 'validate': True}
else:
    tracer_kwargs = {'scan': False, 'validate': False}

def intervention_experiment(model, queries, direction, strength, hidden_states, batch_size=5, remote=True):
    """
    model : an nnsight LanguageModel
    queries : a list of statements to be labeled
    direction : a direction in the residual stream of the model
    hidden_states : list of (layer, -1 or 0) pairs, -1 for intervene before the period, 0 for intervene over the period
    subtract : if True, subtract the direction from the hidden states instead of adding it
    batch_size : batch size for forward passes
    remote : run on the NDIF server?
    Add the direction to the specified hidden states and return the resulting probability diff P(TRUE) - P(FALSE)
    and sum P(TRUE) + P(FALSE) averaged over the data
    """

    true_idx, false_idx = model.tokenizer.encode(' TRUE')[-1], model.tokenizer.encode(' FALSE')[-1]
    len_suffix = len(model.tokenizer.encode('This statement is:'))

    p_diffs = []
    tots = []
    for batch_idx in range(0, len(queries), batch_size):
        batch = queries[batch_idx:batch_idx+batch_size]
        for layer, offset in hidden_states:
            with t.no_grad():
                with model.trace(batch, remote=remote, **tracer_kwargs):
                    model.model.layers[layer].output[:,-len_suffix + offset, :] += \
                        direction * strength
                    logits = model.lm_head.output[:, -1, :]
                    probs = logits.softmax(-1)
                    p_diffs.append((probs[:, true_idx] - probs[:, false_idx]).save())
                    tots.append((probs[:, true_idx] + probs[:, false_idx]).save())
    p_diffs = t.cat([p_diff for p_diff in p_diffs])
    tots = t.cat([tot for tot in tots])

    return p_diffs.mean().item(), tots.mean().item()

def get_probe_direction(ProbeClass, model_name, train_datasets, end_layer, noperiod):
    print(f'training {model_name} probe...')
    if ProbeClass == LRProbe or ProbeClass == MMProbe or ProbeClass == 'random':
        acts, labels = [], []
        for dataset in train_datasets:
            acts.append(collect_acts(dataset, model_name, end_layer, noperiod=noperiod).to('cuda:0'))
            labels.append(t.Tensor(pd.read_csv(f'datasets/{dataset}.csv')['label'].tolist()).to('cuda:0'))
        acts, labels = t.cat(acts), t.cat(labels)
        if ProbeClass == LRProbe or ProbeClass == MMProbe:
            probe = ProbeClass.from_data(acts, labels, device='cuda:0')
        elif ProbeClass == 'random':
            probe = MMProbe.from_data(acts, labels, device='cuda:0')
            probe.direction = t.nn.Parameter(t.randn_like(probe.direction))
    elif ProbeClass == CCSProbe:
        acts = collect_acts(train_datasets[0], model_name, end_layer, noperiod=noperiod).to('cuda:0')
        neg_acts = collect_acts(train_datasets[1], model_name, end_layer, noperiod=noperiod).to('cuda:0')
        labels = t.Tensor(pd.read_csv(f'datasets/{train_datasets[0]}.csv')['label'].tolist()).to('cuda:0')
        probe = ProbeClass.from_data(acts, neg_acts, labels=labels, device='cuda:0')

    direction = probe.direction
    probe_direction = direction.clone().cuda()
    true_acts, false_acts = acts[labels==1], acts[labels==0]
    true_mean, false_mean = true_acts.mean(0), false_acts.mean(0)
    direction = direction / direction.norm()
    diff = (true_mean - false_mean) @ direction
    direction = diff * direction
    # direction is from true to false
    direction = direction.cuda()
    return probe_direction, direction

def prepare_data(prompt, dataset, subset='all'):
    """
    prompt : the few shot prompt
    dataset : dataset name
    model : an nnsight LanguageModel
    subset : 'all', 'true', or 'false'
    Returns a list of queries to be run through the model for the patching experiment
    and a list of the index of the last period token in each query.
    """
    df = pd.read_csv(f'datasets/{dataset}.csv')
    if subset == 'all':
        statements = df['statement'].tolist()
    elif subset == 'true':
        statements = df[df['label'] == 1]['statement'].tolist()
    elif subset == 'false':
        statements = df[df['label'] == 0]['statement'].tolist()

    queries = []
    for statement in statements:
        if statement not in prompt:
            queries.append(prompt + statement + ' This statement is:')
    
    return queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1', default='llama-3.2-3B')
    parser.add_argument('--model_2', default='llama-3.2-3B-Instruct')
    parser.add_argument('--probe', default='MMProbe')
    parser.add_argument('--train_datasets', nargs='+', default=['cities'], type=str)
    parser.add_argument('--val_dataset', default = 'sp_en_trans', type=str)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--intervention', default='2', type=str)
    parser.add_argument('--subset', default='false', type=str)
    parser.add_argument('--device', default='cuda', type=str)
    args = parser.parse_args()

    remote = args.device == 'remote'

    # prepare hidden states to intervene over
    config = configparser.ConfigParser()
    config.read('config.ini')
    end_layer = 13
    noperiod = False
    for train_set in [['cities'], ["cities", "neg_cities"], ['larger_than'], ['larger_than', 'smaller_than']]:
        print(f"train set {train_set}")
        probe_direction_1, direction_1 = get_probe_direction(eval(args.probe), args.model_1, train_set, end_layer, noperiod)
        probe_direction_2, direction_2 = get_probe_direction(eval(args.probe), args.model_2, train_set, end_layer, noperiod)

        cos_sim = t.dot(direction_1, direction_2) / (
                    t.norm(direction_1) * t.norm(direction_2)
                )
        raw_probe_cos_sim = t.dot(probe_direction_1, probe_direction_2) / (
                    t.norm(direction_1) * t.norm(direction_2)
                )
        print(f"{train_set}: {cos_sim}")