import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

from Regression.HIST.train import create_loaders, pprint, get_model, test_epoch, get_metrics
from utils.load_configuation import _log, load_configuration
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

def run_test(
        task_type,
        test_model,
        universe,
        config_name,
        sl_value,
        pl_value,
        start_test_date,
        end_date,
        log_fn=None,
):
    _log(log_fn, "Preparazione del test...")

    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value, start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    _, _, test_loader, stock2concept_matrix, stock_index, test_last_dates, pred_dates = create_loaders(args)
    stock_index = {v: k for k, v in stock_index.items()}

    if args.model_name == 'HIST':
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)

    pprint('create model...')
    if args.model_name == 'SFM':
        model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
    elif args.model_name == 'ALSTM':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
    elif args.model_name == 'Transformer':
        model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
    elif args.model_name == 'HIST':
        model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers, K = args.K)
    else:
        model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)

    model.to(device)
    chk = torch.load(f'{model_save_path}/model.pth', weights_only=False)
    model.load_state_dict(chk)

    test_preds, test_labels, test_stock_indexes, _, _, mask = test_epoch(model, test_loader, args, stock2concept_matrix, prefix='Test')

    tickers = [[stock_index.get(num, num) for num in sublist] for sublist in test_stock_indexes]

    best_metrics = get_metrics(test_preds, test_labels)
    pred_dates = [date for i, date in enumerate(pred_dates) if mask[i]]
    test_last_dates = [date for i, date in enumerate(test_last_dates) if mask[i]]
    test_labels = [np.expand_dims(lab, axis=1) for lab in test_labels]
    test_preds = [np.expand_dims(pred, axis=1) for pred in test_preds]

    metrics, results_pkl, metrics_json = save_results(
        args,
        best_metrics,
        test_preds,
        test_labels,
        pred_dates,
        test_last_dates,
        tickers,
        metrics_path
    )

    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": str(device),
    }

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Regression')
    parser.add_argument("--model_name", type=str, default='HIST')
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--config_name", type=str, default='HIST')
    parser.add_argument("--sl_value", type=int, default=1)
    parser.add_argument("--pl_value", type=int, default=5)
    parser.add_argument("--start_test_date", type=str, default='2023-01-01')
    parser.add_argument("--end_date", type=str, default='2023-12-31')
    args = parser.parse_args()

    out = run_test(
        task_type=args.task_type,
        test_model=args.model_name,
        universe=args.universe,
        config_name=args.config_name,
        sl_value=args.sl_value,
        pl_value=args.pl_value,
        start_test_date=args.start_test_date,
        end_date=args.end_date,
    )
    print(json.dumps(out, indent=4))