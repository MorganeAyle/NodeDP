import copy

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from torch.autograd import Variable


from src.models import create_model


def to_numpy(x):
    if isinstance(x, Variable):
        x = x.data
    return x.cpu().numpy() if x.is_cuda else x.numpy()


class Evaluator:
    def __init__(self, model_args, feats, class_arr, loss, early_stopping_after):
        self.feats = torch.from_numpy(feats.astype(np.float32))
        self.labels = torch.from_numpy(class_arr.astype(np.float32))
        self.early_stopping_after = early_stopping_after

        # Loss type
        if loss == 'sigmoid':
            self.sigmoid_loss = True
        else:
            assert loss == 'softmax'
            self.sigmoid_loss = False

        # Create model
        in_channels = self.feats.shape[1]
        out_channels = self.labels.shape[1]
        self.model = create_model(in_channels, out_channels, model_args)

        self.metrics = None
        self.best_metric = None
        self.count = 0
        self.best_model = None
        self.eps = 0
        self.best_eps = 0

    def predict(self, preds):
        return nn.Sigmoid()(preds) if self.sigmoid_loss else F.softmax(preds, dim=1)

    def eval_step(self, nodes, adj, roots=None):
        self.model.eval()
        with torch.no_grad():
            preds = self.model(self.feats[nodes], adj)
        return self.predict(preds), self.labels[nodes]

    def calc_metrics(self, preds, labels):
        y_pred = to_numpy(preds)
        y_true = to_numpy(labels)

        if not self.sigmoid_loss:
            y_true = np.argmax(y_true, axis=1)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred[y_pred > 0.5] = 1
            y_pred[y_pred <= 0.5] = 0

        ret = {
            "F1 Micro": metrics.f1_score(y_true, y_pred, average="micro"),
            "F1 Macro": metrics.f1_score(y_true, y_pred, average="macro")
        }
        if not self.sigmoid_loss:
            ret["Accuracy"] = metrics.accuracy_score(y_true, y_pred)

        self.metrics = ret
        return ret

    @property
    def early_stopping(self) -> bool:
        """
        Keeps track of the best metric for the corresponding loss and returns True if early stopping needed.
        """
        if not self.sigmoid_loss:
            metric = self.metrics["Accuracy"]
        else:
            metric = self.metrics["F1 Macro"]

        if self.best_metric is None:
            self.best_metric = metric
            self.best_model = copy.deepcopy(self.model)
            self.best_eps = self.eps
            return False
        elif metric > self.best_metric:
            self.best_metric = metric
            self.best_model = copy.deepcopy(self.model)
            self.count = 0
            self.best_eps = self.eps
            return False
        else:
            self.count += 1
            if self.count >= self.early_stopping_after:
                return True
            else:
                return False
