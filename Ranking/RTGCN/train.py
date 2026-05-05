#!/usr/bin/env python
import json
import os
import sys

from Ranking.RTGCN.processor.recognition import REC_Processor
from utils.load_configuation import create_saving_path
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))


if __name__ == '__main__':

    p = REC_Processor(sys.argv[1:])

    model_save_path, metrics_path = create_saving_path(p.arg, "Ranking", "RTGCN", PROJECT_ROOT)

    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(p.arg), f, indent=4)

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


