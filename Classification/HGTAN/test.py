import os
from argparse import ArgumentParser

import torch

from .load_data import load_data, extract_dates, load_matrices
from .model.models import HGTAN
from .tool import eval_model, get_metrics, prepare_data
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result
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
    _log(log_fn, "Preparing test...")

    args, model_save_path, metrics_path = load_configuration(
        task_type, test_model, universe, config_name, sl_value, pl_value, start_test_date, end_date, PROJECT_ROOT, log_fn
    )

    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'

    args.d_word_vec = args.d_model

    print('Extracting data...')
    eod_data, ground_truth, valid_index, test_index, dates, tickers = load_data(args)
    pred_dates, last_dates = extract_dates(test_index, dates, args)
    adj, H = load_matrices(tickers, args)

    len_data = eod_data.shape[1] - args.seq_len - args.pred_len + 1
    test_loader = prepare_data(eod_data, ground_truth, test_index, len_data, args, flag='test')

    model = HGTAN(
        rnn_unit=args.rnn_unit,
        n_hid=args.hidden,
        n_class=args.n_class,
        feature=args.feature,
        tgt_emb_prj_weight_sharing=args.proj_share_weight,
        d_k=args.d_k,
        d_v=args.d_v,
        d_model=args.d_model,
        d_word_vec=args.d_word_vec,
        n_head=args.n_head,
        dropout=args.dropout
    ).to(device)

    preds, labels = eval_model(model, test_loader, adj, H, device, args, model_save_path)
    metrics = get_metrics(preds, labels)

    metrics, results_pkl, metrics_json = save_results(
        args,
        metrics,
        preds,
        labels,
        pred_dates.dt.strftime("%Y-%m-%d").to_list(),
        last_dates.dt.strftime("%Y-%m-%d").to_list(),
        [tickers] * len(preds),
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
    parser.add_argument("--task_type", type=str, default='Classification')
    parser.add_argument("--model_name", type=str, default='HGTAN')
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--config_name", type=str, default='HGTAN')
    parser.add_argument("--sl_value", type=int, default=20)
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
    emit_result(out)