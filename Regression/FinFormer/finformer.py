import copy
import json

import numpy as np
import pandas as pd
import torch
import math
from sklearn.metrics import r2_score
from torch import nn
from tqdm import tqdm

from Regression.FinFormer.model import Finformer
from Regression.FinFormer.utils import edgeIndexTransform, pprint
from audtorch.metrics.functional import concordance_cc

from utils.save_results import save_results


def metric_fn_2(preds):
    preds = preds[~np.isnan(preds['label'])]

    ic = preds['label'].corr(preds['score'])
    rank_ic = preds['label'].corr(preds['score'], method='spearman')

    return ic, rank_ic


class Finformer_Model:
    def __init__(self, d_feat, hidden_size, num_layers, temporal_dropout, snum_head, device):
        super().__init__()
        self.d_feat = d_feat
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.temporal_dropout = temporal_dropout
        self.device = device
        self.finformer = Finformer(d_feat=d_feat, hidden_size=hidden_size, temporal_dropout=temporal_dropout,
                                   snum_head=snum_head).to(device)

        for n in self.finformer.modules():
            if isinstance(n, nn.Linear):
                n.weight = nn.init.xavier_normal_(n.weight, gain=1.)

    def get_metrics(self, preds, labels):
        preds_flat = np.concatenate(preds, axis=0)
        labels_flat = np.concatenate(labels, axis=0)

        metrics = {
            'MSE': np.mean((preds_flat - labels_flat) ** 2),
            'MAE': np.mean(np.abs(preds_flat - labels_flat)),
            'RMSE': np.sqrt(np.mean((preds_flat - labels_flat) ** 2)),
            'R2': r2_score(preds_flat, labels_flat)
        }

        return metrics

    def trainer(self, train_loader, optimizer, adj_matrix, ticker_list):
        self.finformer.train()

        loss_record = []
        batch_num = 0

        for data in tqdm(train_loader):
            feature, label, instruments, daily_index = data['data'], data['label'], data['instruments'], data['daily_index']
            feature = feature.squeeze(0).to(self.device)
            label = label.squeeze(0).to(self.device)

            mask = [str(elem) in instruments for elem in ticker_list]
            current_adj_matrix = adj_matrix[mask, :][:, mask]
            graph = edgeIndexTransform(current_adj_matrix)

            batch_num += 1
            pred, mask = self.finformer(feature.float(), graph.to(self.device))
            optimizer.zero_grad()
            loss = -concordance_cc(pred, label)
            loss.backward()
            optimizer.step()
            loss_record.append(loss)

    def tester(self, test_loader, adj_matrix, ticker_list):
        self.finformer.eval()
        losses = []
        preds = []
        predictions = []
        labels = []
        tickers = []
        batch_num = 0
        itr = 0

        for data in test_loader:
            with torch.no_grad():
                feature, label, instruments, daily_index = data['data'], data['label'], data['instruments'], data['daily_index']
                feature = feature.squeeze(0).to(self.device)
                label = label.squeeze(0).to(self.device)

                mask = [str(elem) in instruments for elem in ticker_list]
                current_adj_matrix = adj_matrix[mask, :][:, mask]
                graph = edgeIndexTransform(current_adj_matrix)

                batch_num += 1
                pred, mask = self.finformer(feature.float(), graph.to(self.device))
                loss = -concordance_cc(pred, label)

                preds.append(pd.DataFrame({'score': pred.cpu().numpy(), 'label': label.cpu().numpy()}))

                labels.append(label.cpu().numpy())
                predictions.append(pred.cpu().numpy())
                tickers.append(instruments)

                itr += 1
                losses.append(loss.item())

        # evaluate
        preds = pd.concat(preds, axis=0)
        ic, rank_ic = metric_fn_2(preds)

        return np.mean(losses), ic, rank_ic, predictions, labels, tickers

    def fit(self, args, train_loader, valid_loader, test_loader, adj_matrix, ticker_list, metrics_path, model_save_path):

        optimizer = torch.optim.Adam(self.finformer.parameters(), lr=args.lr)
        stop_round = 0
        early_stop = 10
        best_score = -math.inf
        best_epoch = -1
        best_param = 0

        for epoch in tqdm(range(args.n_epochs)):
            pprint('Running Epoch:', epoch)
            pprint('training...')
            self.trainer(train_loader, optimizer, adj_matrix, ticker_list)

            valid_loss, valid_score, valid_rank_ic, val_preds, val_labels, _ = self.tester(valid_loader, adj_matrix, ticker_list)
            test_loss, test_score, test_rank_ic, preds, labels, tickers = self.tester(test_loader, adj_matrix, ticker_list)

            pprint('valid_loss {:.6f}, test_loss {:.6f}'.format(valid_loss, test_loss))
            pprint('valid_score {:.6f}, test_score {:.6f}'.format(valid_score, test_score))
            pprint('valid_rank_ic {:.6f}, test_rank_ic {:.6f}'.format(valid_rank_ic, test_rank_ic))

            if valid_score > best_score:
                best_score = valid_score
                best_param = copy.deepcopy(self.finformer.state_dict())
                best_epoch = epoch
                print('This is epoch {:.1f}, Saving FF with score {:.3f}...'.format(epoch, best_score))
                stop_round = 0

                val_metrics = self.get_metrics(val_preds, val_labels)
                val_metrics = {k: float(v) for k, v in val_metrics.items()}

                with open(f'{metrics_path}/val_metrics_sl1_pl{args.pred_len}.json', 'w') as f:
                    json.dump(val_metrics, f, indent=4)

                test_metrics = self.get_metrics(preds, labels)

                preds = [np.expand_dims(i, axis=1) for i in preds]
                labels = [np.expand_dims(i, axis=1) for i in labels]

                metrics, results_pkl, metrics_json = save_results(
                    args,
                    test_metrics,
                    preds,
                    labels,
                    test_loader.dataset.valid_dates[args.pred_len:],
                    test_loader.dataset.valid_dates[:-args.pred_len],
                    tickers,
                    metrics_path
                )

                print('Test Metrics:\n', metrics)

            else:
                stop_round += 1
                if stop_round >= early_stop:
                    pprint('early stop')
                    break

        pprint('best score:', best_score, '@', best_epoch)
        torch.save(best_param, f'{model_save_path}/model.pth')




