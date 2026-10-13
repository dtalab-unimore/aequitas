import argparse
import copy
import json
import os
import random
import warnings
from time import time

import numpy as np
from sklearn.utils import shuffle

warnings.filterwarnings("ignore", category=FutureWarning)

from .evaluator import evaluate
from .load_dataset import load_dataset
from .model import AdvALSTM
from utils.load_configuation import create_saving_path
from utils.logger import emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

WEIGHTS_FILE = "model.weights.npz"


def load_data(args):
    """Train / valid / test arrays (float32) and the test sample identifiers."""
    (tra_pv, tra_wd, tra_gt, val_pv, val_wd, val_gt, tes_pv, tes_wd, tes_gt,
     test_tickers, test_last_dates, test_dates) = load_dataset(
        args.data_path, args.universe, args.start_date, args.end_date, args.end_train_date,
        args.end_valid_date, args.seq_len, args.pred_len, start_test_date=args.start_test_date,
    )
    as_float = lambda a: np.asarray(a, dtype=np.float32)
    return (as_float(tra_pv), as_float(tra_gt), as_float(val_pv), as_float(val_gt),
            as_float(tes_pv), as_float(tes_gt), test_tickers, test_last_dates, test_dates)


def build_model(args, fea_dim):
    return AdvALSTM(
        fea_dim=fea_dim, seq=args.seq_len, unit=args.unit,
        alp=args.alpha_l2, bet=args.beta_adv, eps=args.epsilon_adv, lr=args.learning_rate,
        att=bool(args.att), hinge=bool(args.hinge_lose), adv=bool(args.adv), seed=args.seed,
    )


def get_batch(tra_pv, tra_gt, batch_size, sta_ind):
    if sta_ind + batch_size < tra_pv.shape[0]:
        end_ind = sta_ind + batch_size
    else:
        sta_ind = tra_pv.shape[0] - batch_size
        end_ind = tra_pv.shape[0]
    return tra_pv[sta_ind:end_ind], tra_gt[sta_ind:end_ind]


def run_train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    model_save_path, metrics_path = create_saving_path(args, "Classification", "Adv-ALSTM", PROJECT_ROOT)
    with open(f'{model_save_path}/config_params.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    (tra_pv, tra_gt, val_pv, val_gt, tes_pv, tes_gt,
     test_tickers, test_last_dates, test_dates) = load_data(args)
    print(f"Samples: train {len(tra_pv)}, valid {len(val_pv)}, test {len(tes_pv)}", flush=True)

    model = build_model(args, tra_pv.shape[2])
    weights_path = os.path.join(model_save_path, WEIGHTS_FILE)
    hinge = bool(args.hinge_lose)

    bat_count = -(-tra_pv.shape[0] // args.batch_size)
    best_valid_perf = {'Accuracy': 0, 'MCC': -2}
    best_test_perf, best_test_pred = None, None

    for epoch in range(args.epoch):
        t1 = time()
        tra_obj = tra_loss = l2 = tra_adv = 0.0
        for j in range(bat_count):
            pv_b, gt_b = get_batch(tra_pv, tra_gt, args.batch_size, j * args.batch_size)
            obj, loss, cur_l2, adv = model.train_step(pv_b, gt_b)
            tra_obj += float(obj); tra_loss += float(loss); l2 += float(cur_l2); tra_adv += float(adv)

        tra_acc = 0.0
        for j in range(bat_count):
            pv_b, gt_b = get_batch(tra_pv, tra_gt, args.batch_size, j * args.batch_size)
            tra_acc += evaluate(model.predict(pv_b).numpy(), gt_b, hinge)['Accuracy']

        val_pre = model.predict(val_pv).numpy()
        cur_valid_perf = evaluate(val_pre, val_gt, hinge)
        tes_pre = model.predict(tes_pv).numpy()
        cur_test_perf = evaluate(tes_pre, tes_gt, hinge)

        print(f"Epoch {epoch + 1}/{args.epoch}: obj={tra_obj / bat_count:.4f} loss={tra_loss / bat_count:.4f} "
              f"adv={tra_adv / bat_count:.4f} train_acc={tra_acc / bat_count:.4f} "
              f"valid_acc={cur_valid_perf['Accuracy']:.4f} test_acc={cur_test_perf['Accuracy']:.4f} "
              f"({time() - t1:.1f}s)", flush=True)

        if cur_valid_perf['Accuracy'] > best_valid_perf['Accuracy'] or best_test_pred is None:
            best_valid_perf = copy.copy(cur_valid_perf)
            best_test_perf = copy.copy(cur_test_perf)
            best_test_pred = copy.copy(tes_pre)
            model.save(weights_path)

        tra_pv, tra_gt = shuffle(tra_pv, tra_gt, random_state=0)

    with open(f'{metrics_path}/val_metrics_sl{args.seq_len}_pl{args.pred_len}.json', 'w') as f:
        json.dump({k: float(v) for k, v in best_valid_perf.items()}, f, indent=4)

    metrics, results_pkl, metrics_json = save_results(
        args, best_test_perf, best_test_pred, tes_gt, test_dates, test_last_dates, test_tickers, metrics_path
    )
    return {"model_path": weights_path, "metrics": metrics,
            "results_path": results_pkl, "metrics_path": metrics_json}


def build_parser():
    parser = argparse.ArgumentParser(description='Train Adv-ALSTM')
    parser.add_argument('-p', '--data_path', help='path of pv data', type=str, default='./Evaluation/data')
    parser.add_argument('--universe', help='universe of stocks', type=str, default='sp500')
    parser.add_argument('-m', '--model_name', type=str, default='Adv-ALSTM', help='name of the configuration')
    parser.add_argument('-l', '--seq_len', '--seq', dest='seq_len', help='length of history', type=int, default=15)
    parser.add_argument('-n', '--pred_len', help='length of prediction', type=int, default=5)
    parser.add_argument('-u', '--unit', help='number of hidden units in lstm', type=int, default=16)
    parser.add_argument('-l2', '--alpha_l2', type=float, default=1e-3, help='alpha for l2 regularizer')
    parser.add_argument('-la', '--beta_adv', type=float, default=5e-2, help='beta for adverarial loss')
    parser.add_argument('-le', '--epsilon_adv', type=float, default=1e-3, help='epsilon to control the scale of noise')
    parser.add_argument('-b', '--batch_size', help='batch size', type=int, default=1024)
    parser.add_argument('-e', '--epoch', help='epoch', type=int, default=150)
    parser.add_argument('-r', '--learning_rate', help='learning rate', type=float, default=1e-2)
    parser.add_argument('-f', '--seed', type=int, default=42, help='random seed')
    parser.add_argument('-a', '--att', type=int, default=1, help='use attention model')
    parser.add_argument('-v', '--adv', type=int, default=1, help='adversarial training')
    parser.add_argument('-hi', '--hinge_lose', type=int, default=1, help='use hinge lose')

    parser.add_argument('--start_date', type=str, default='2018-01-01')
    parser.add_argument('--end_train_date', type=str, default='2021-12-31')
    parser.add_argument('--start_valid_date', type=str, default='2022-01-01')
    parser.add_argument('--end_valid_date', type=str, default='2022-12-31')
    parser.add_argument('--start_test_date', type=str, default='2023-01-01')
    parser.add_argument('--end_date', type=str, default='2023-12-31')
    return parser


if __name__ == '__main__':
    output = run_train(build_parser().parse_args())
    emit_result(output)
