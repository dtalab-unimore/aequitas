import os
from argparse import ArgumentParser

from Ranking.RTGCN.processor.recognition import REC_Processor
from utils.load_configuation import load_configuration
from utils.logger import _log, emit_result
from utils.save_results import save_results

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

def namespace_to_cli_args(namespace, exclude=None, bool_value_args=None, bool_flags=None):
    exclude = set(exclude or [])
    bool_value_args = set(bool_value_args or [])
    bool_flags = set(bool_flags or [])

    cli_args = []

    for key, value in vars(namespace).items():
        if key in exclude or value is None:
            continue

        option = f"--{key}"

        if isinstance(value, bool):
            if key in bool_flags:
                if value:
                    cli_args.append(option)
            else:
                cli_args.extend([option, "1" if value else "0"])

        elif isinstance(value, (list, tuple)):
            if len(value) > 0:
                cli_args.append(option)
                cli_args.extend(map(str, value))

        else:
            cli_args.extend([option, str(value)])

    return cli_args

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

    args.weights = 1
    list_args = namespace_to_cli_args(
        args,
        exclude={"weights"} if args.weights is None else None,
        bool_flags={"debug"}
    )

    p = REC_Processor(list_args)

    metrics, preds, labels, dataset = p.test(metrics_path, flag='test')

    metrics, results_pkl, metrics_json = save_results(
        p.arg,
        metrics,
        preds,
        labels,
        dataset.dates_gt.to_list(),
        dataset.last_seq_date.to_list(),
        [dataset.tickers] * len(preds),
        metrics_path,
        log_fn=log_fn
    )

    return {
        "metrics": metrics,
        "results_path": results_pkl,
        "metrics_path": metrics_json,
        "device": args.device,
    }




if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--task_type", type=str, default='Ranking')
    parser.add_argument("--model_name", type=str, default='RTGCN')
    parser.add_argument("--universe", type=str, default='sx5e')
    parser.add_argument("--config_name", type=str, default='RT-GCN')
    parser.add_argument("--sl_value", type=int, default=4)
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