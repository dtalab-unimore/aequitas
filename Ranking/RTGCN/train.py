#!/usr/bin/env python
import argparse
import json
import os
import random
import sys

import numpy as np
import torch

from Ranking.RTGCN.torchlight.torchlight import import_class
from utils.load_configuation import create_saving_path
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Processor collection')
    parser.add_argument('--seed', type=int, default=42, help='random seed for initialization')
    parser.add_argument('--processor', default='recognition', help='the processor will be used')

    # region register processor yapf: disable
    processors = dict()
    processors['recognition'] = import_class('processor.recognition.REC_Processor')
    #endregion yapf: enable

    # add sub-parser
    subparsers = parser.add_subparsers(dest='processor')
    for k, p in processors.items():
        subparsers.add_parser(k, parents=[p.get_parser()])

    # read arguments
    arg = parser.parse_args()
    set_seed(arg.seed)

    # start
    Processor = processors[arg.processor]
    p = Processor(sys.argv[4:])

    model_save_path, metrics_path = create_saving_path(p.arg, "Ranking", "RTGCN", PROJECT_ROOT)

    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(arg), f, indent=4)

    metrics, preds, labels, dataset = p.start(model_save_path, metrics_path)

    metrics, results_pkl, metrics_json = save_results(
        p.arg,
        metrics,
        preds,
        labels,
        dataset.dates_gt.to_list(),
        dataset.last_seq_date.to_list(),
        [dataset.tickers] * len(preds),
        metrics_path
    )

    print('Test metrics:', metrics)


