import time

import torch
import torch.nn as nn

import numpy as np

from RunningStats import TorchRunningMeanStd


class RandomEncoder(nn.Module):
    def __init__(self, feature_dims):
        super(RandomEncoder, self).__init__()

        self.ent_stats = TorchRunningMeanStd()
        self.k = 3

        self.fc0 = nn.Linear(feature_dims, 192)

        self.net = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        for param in self.net.parameters():
            param.data.normal_(0, 0.1)
            param.requires_grad = False

    def forward(self, x):
        x = self.fc0(x)

        x = x.reshape(-1, 3, 8, 8).squeeze()

        return self.net(x)

    def normalize_entropies(self, entropies):
        self.ent_stats.update(entropies)

        normalized_entropy = entropies / self.ent_stats.std

        return torch.log(normalized_entropy + 1.0)

    def compute_intrinsic_rewards(self, source_y_t, target_y_t, average_entropy=False):
        """
        Computes intrinsic rewards across the whole buffer.

        Implementation from: https://github.com/younggyoseo/RE3/blob/master/a2c_re3/torch-ac/torch_ac/algos/base.py#L303
        """
        with torch.no_grad():
            dists = []

            for i in range(len(target_y_t) // 10000 + 1):
                start = i * 10000
                end = (i + 1) * 10000

                dist = torch.norm(
                    source_y_t[:, None, :] - target_y_t[None, start:end, :],
                    dim=-1, p=2
                )

                dists.append(dist)

            dists = torch.cat(dists)
            knn_dists = 0.0

            if not average_entropy:
                knn_dists = torch.kthvalue(dists, k=self.k + 1, dim=1).values
            else:
                for k in range(5):
                    knn_dists += torch.kthvalue(dists, k=k + 1, dim=1).values
                knn_dists /= 5

            state_entropy = knn_dists

            return self.normalize_entropies(state_entropy.to('cpu'))
