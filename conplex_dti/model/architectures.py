from types import SimpleNamespace

import os
import pickle as pk
from functools import lru_cache

import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..featurizer.protein import FOLDSEEK_MISSING_IDX
from ..utils import get_logger

logg = get_logger()

#################################
# Latent Space Distance Metrics #
#################################


class Cosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2)


class SquaredCosine(nn.Module):
    def forward(self, x1, x2):
        return nn.CosineSimilarity()(x1, x2) ** 2


class Euclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0)


class SquaredEuclidean(nn.Module):
    def forward(self, x1, x2):
        return torch.cdist(x1, x2, p=2.0) ** 2


DISTANCE_METRICS = {
    "Cosine": Cosine,
    "SquaredCosine": SquaredCosine,
    "Euclidean": Euclidean,
    "SquaredEuclidean": SquaredEuclidean,
}

ACTIVATIONS = {"ReLU": nn.ReLU, "GELU": nn.GELU, "ELU": nn.ELU, "Sigmoid": nn.Sigmoid}

#######################
# Model Architectures #
#######################


class LogisticActivation(nn.Module):
    """
    Implementation of Generalized Sigmoid
    Applies the element-wise function:
    :math:`\sigma(x) = \frac{1}{1 + \exp(-k(x-x_0))}`
    :param x0: The value of the sigmoid midpoint
    :type x0: float
    :param k: The slope of the sigmoid - trainable -  :math:`k \geq 0`
    :type k: float
    :param train: Whether :math:`k` is a trainable parameter
    :type train: bool
    """

    def __init__(self, x0=0, k=1, train=False):
        super().__init__()
        self.x0 = x0
        self.k = nn.Parameter(torch.FloatTensor([float(k)]), requires_grad=False)
        self.k.requiresGrad = train

    def forward(self, x):
        """
        Applies the function to the input elementwise
        :param x: :math:`(N \times *)` where :math:`*` means, any number of additional dimensions
        :type x: torch.Tensor
        :return: :math:`(N \times *)`, same shape as the input
        :rtype: torch.Tensor
        """
        o = torch.clamp(
            1 / (1 + torch.exp(-self.k * (x - self.x0))), min=0, max=1
        ).squeeze()
        return o

    def clip(self):
        """
        Restricts sigmoid slope :math:`k` to be greater than or equal to 0, if :math:`k` is trained.
        :meta private:
        """
        self.k.data.clamp_(min=0)


#######################
# Model Architectures #
#######################


class SimpleCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), self.latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class ResidualBlock(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model)
        self.fc2 = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        self.alpha = nn.Parameter(torch.tensor(0.0)) # Learnable alpha initialized to 0

    def forward(self, x):
        residual = x
        out = self.activation(self.fc1(x))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        return residual + self.alpha * out


class ResidualCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        num_blocks=3,
        dropout=0.1,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]
        self.num_blocks = num_blocks

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension),
            self.latent_activation(),
            *[ResidualBlock(latent_dimension, dropout) for _ in range(num_blocks)]
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension),
            self.latent_activation(),
            *[ResidualBlock(latent_dimension, dropout) for _ in range(num_blocks)]
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class SimpleCoembeddingSigmoid(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCoembedding_FoldSeek(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=1024,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class SimpleCoembedding_FoldSeekX(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        classify=True,
        foldseek_embedding_dimension=512,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.foldseek_embedding_dimension = foldseek_embedding_dimension
        self.do_classify = classify

        self.foldseek_index_embedding = nn.Embedding(
            22,
            self.foldseek_embedding_dimension,
            padding_idx=FOLDSEEK_MISSING_IDX,
        )

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self._target_projector = nn.Sequential(
            nn.Linear(
                (self.target_shape + self.foldseek_embedding_dimension),
                latent_dimension,
            ),
            latent_activation(),
        )
        nn.init.xavier_normal_(self._target_projector[0].weight)

        # self.projector_dropout = nn.Dropout(p=0.2)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def _split_foldseek_target_embedding(self, target_embedding):
        """
        Expect that first dimension of target_embedding is batch dimension, second dimension is [target_shape | protein_length]

        FS indexes from 1-21, 0 is padding
        target is D + N_pool
            first D is PLM embedding
            next N_pool is FS index + pool
            nn.Embedding ignores elements with padding_idx = 0

            N --embedding--> N x D_fs --mean pool--> D_fs
            target is (D | D_fs) --linear--> latent
        """
        if target_embedding.shape[1] == self.target_shape:
            return target_embedding

        plm_embedding = target_embedding[:, : self.target_shape]
        foldseek_indices = target_embedding[:, self.target_shape :].long()
        foldseek_embedding = self.foldseek_index_embedding(foldseek_indices).mean(dim=1)

        full_target_embedding = torch.cat([plm_embedding, foldseek_embedding], dim=1)
        return full_target_embedding

    def target_projector(self, target):
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)
        return target_projection

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_fs_emb = self._split_foldseek_target_embedding(target)
        target_projection = self._target_projector(target_fs_emb)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        relu_f = torch.nn.ReLU()
        return relu_f(inner_prod).squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class GoldmanCPI(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=100,
        latent_activation=nn.ReLU,
        latent_distance="Cosine",
        model_dropout=0.2,
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify

        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.drug_projector[0].weight)

        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension), latent_activation()
        )
        nn.init.xavier_normal_(self.target_projector[0].weight)

        self.last_layers = nn.Sequential(
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, latent_dimension, bias=True),
            nn.Dropout(p=model_dropout),
            nn.ReLU(),
            nn.Linear(latent_dimension, 1, bias=True),
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)
        output = torch.einsum("bd,bd->bd", drug_projection, target_projection)
        distance = self.last_layers(output)
        return distance

    def classify(self, drug, target):
        distance = self.regress(drug, target)
        sigmoid_f = torch.nn.Sigmoid()
        return sigmoid_f(distance).squeeze()


class SimpleCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class AffinityCoembedInner(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.mol_projector[0].weight)

        print(self.mol_projector[0].weight)

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )
        nn.init.xavier_uniform(self.prot_projector[0].weight)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        print(mol_proj)
        print(prot_proj)
        y = torch.bmm(
            mol_proj.view(-1, 1, self.latent_size),
            prot_proj.view(-1, self.latent_size, 1),
        ).squeeze()
        return y


class CosineBatchNorm(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric="Cosine",
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, self.latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, self.latent_size),
            latent_activation(),
        )

        self.mol_norm = nn.BatchNorm1d(self.latent_size)
        self.prot_norm = nn.BatchNorm1d(self.latent_size)

        self.dist_metric = distance_metric
        self.activator = DISTANCE_METRICS[self.dist_metric]()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_norm(self.mol_projector(mol_emb))
        prot_proj = self.prot_norm(self.prot_projector(prot_emb))

        return self.activator(mol_proj, prot_proj)


class LSTMCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        lstm_layers=3,
        lstm_dim=256,
        latent_size=256,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.rnn = nn.LSTM(
            self.prot_emb_size,
            lstm_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(2 * lstm_layers * lstm_dim, latent_size), nn.ReLU()
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)

        outp, (h_out, _) = self.rnn(prot_emb)
        prot_hidden = h_out.permute(1, 0, 2).reshape(outp.shape[0], -1)
        prot_proj = self.prot_projector(prot_hidden)

        return self.activator(mol_proj, prot_proj)


class DeepCosine(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        hidden_size=4096,
        latent_activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, hidden_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
            nn.Linear(hidden_size, latent_size),
            torch.nn.Dropout(p=0.5, inplace=False),
            latent_activation(),
        )

        self.activator = nn.CosineSimilarity()

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)

        return self.activator(mol_proj, prot_proj)


class SimpleConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        hidden_dim_1=512,
        hidden_dim_2=256,
        activation=nn.ReLU,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.fc1 = nn.Sequential(
            nn.Linear(mol_emb_size + prot_emb_size, hidden_dim_1), activation()
        )
        self.fc2 = nn.Sequential(nn.Linear(hidden_dim_1, hidden_dim_2), activation())
        self.fc3 = nn.Sequential(nn.Linear(hidden_dim_2, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc3(self.fc2(self.fc1(cat_emb))).squeeze()


class SeparateConcat(nn.Module):
    def __init__(
        self,
        mol_emb_size=2048,
        prot_emb_size=100,
        latent_size=1024,
        latent_activation=nn.ReLU,
        distance_metric=None,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), latent_activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), latent_activation()
        )

        self.fc = nn.Sequential(nn.Linear(2 * latent_size, 1), nn.Sigmoid())

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


class AffinityEmbedConcat(nn.Module):
    def __init__(
        self, mol_emb_size, prot_emb_size, latent_size=1024, activation=nn.ReLU
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.latent_size = latent_size

        self.mol_projector = nn.Sequential(
            nn.Linear(self.mol_emb_size, latent_size), activation()
        )

        self.prot_projector = nn.Sequential(
            nn.Linear(self.prot_emb_size, latent_size), activation()
        )

        self.fc = nn.Linear(2 * latent_size, 1)

    def forward(self, mol_emb, prot_emb):
        mol_proj = self.mol_projector(mol_emb)
        prot_proj = self.prot_projector(prot_emb)
        cat_emb = torch.cat([mol_proj, prot_proj], axis=1)
        return self.fc(cat_emb).squeeze()


SimplePLMModel = AffinityEmbedConcat


class AffinityConcatLinear(nn.Module):
    def __init__(
        self,
        mol_emb_size,
        prot_emb_size,
    ):
        super().__init__()
        self.mol_emb_size = mol_emb_size
        self.prot_emb_size = prot_emb_size
        self.fc = nn.Linear(mol_emb_size + prot_emb_size, 1)

    def forward(self, mol_emb, prot_emb):
        cat_emb = torch.cat([mol_emb, prot_emb], axis=1)
        return self.fc(cat_emb).squeeze()
SimpleCoembeddingNoSigmoid = SimpleCoembedding

class CrossAttentionCoembedding(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
        num_heads=4,
        dropout=0.1,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.do_classify = classify
        self.num_heads = num_heads
        self.dropout = dropout

        # Project inputs to latent dimension
        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, latent_dimension),
            ACTIVATIONS[latent_activation](),
            nn.Dropout(dropout)
        )
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, latent_dimension),
            ACTIVATIONS[latent_activation](),
            nn.Dropout(dropout)
        )

        # Cross-Attention Layer
        # Embeddings will be (Batch, Latent), we need (Seq, Batch, Latent) for MultiheadAttention
        # Since we have single embeddings per drug/target, Seq length is 1.
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=latent_dimension,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Final prediction head
        # We'll concatenate the attended drug and target embeddings
        self.mlp = nn.Sequential(
            nn.Linear(2 * latent_dimension, latent_dimension),
            ACTIVATIONS[latent_activation](),
            nn.Dropout(dropout),
            nn.Linear(latent_dimension, 1)
        )

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        # Project to latent space
        drug_emb = self.drug_projector(drug)      # (B, L)
        target_emb = self.target_projector(target) # (B, L)

        # Reshape for MultiheadAttention: (B, S, E) where S=1
        drug_seq = drug_emb.unsqueeze(1)       # (B, 1, L)
        target_seq = target_emb.unsqueeze(1)   # (B, 1, L)

        # Cross Attention: Drug attends to Target
        # query=drug, key=target, value=target
        attn_output, _ = self.cross_attention(
            query=drug_seq,
            key=target_seq,
            value=target_seq
        )
        
        # attn_output is (B, 1, L)
        drug_attended = attn_output.squeeze(1) # (B, L)

        # Concatenate
        combined = torch.cat([drug_attended, target_emb], dim=1) # (B, 2L)

        # Predict
        prediction = self.mlp(combined) # (B, 1)

        if self.do_classify:
            return torch.sigmoid(prediction).squeeze()
        else:
            return prediction.squeeze()

    def regress(self, drug, target):
        return self.forward(drug, target)

    def classify(self, drug, target):
        return self.forward(drug, target)


class DuoLayerPerceptron(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        # *** NEW PARAMETER FOR HIDDEN LAYER DIMENSION ***
        hidden_dimension=1024,
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        self.hidden_dimension = hidden_dimension # Store new parameter
        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # *** MODIFIED DRUG PROJECTOR ***
        # Adds a new Linear layer and activation (hidden layer)
        self.drug_projector = nn.Sequential(
            nn.Linear(self.drug_shape, self.hidden_dimension),
            self.latent_activation(),
            nn.Linear(self.hidden_dimension, latent_dimension),
            self.latent_activation()
        )
        # Initialize weights for the two Linear layers
        nn.init.xavier_normal_(self.drug_projector[0].weight)
        nn.init.xavier_normal_(self.drug_projector[2].weight)

        # *** MODIFIED TARGET PROJECTOR ***
        # Adds a new Linear layer and activation (hidden layer)
        self.target_projector = nn.Sequential(
            nn.Linear(self.target_shape, self.hidden_dimension),
            self.latent_activation(),
            nn.Linear(self.hidden_dimension, latent_dimension),
            self.latent_activation()
        )
        # Initialize weights for the two Linear layers
        nn.init.xavier_normal_(self.target_projector[0].weight)
        nn.init.xavier_normal_(self.target_projector[2].weight)

        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class QuintupleLayerPerceptron(nn.Module):
    def __init__(
        self,
        drug_shape=2048,
        target_shape=1024,
        latent_dimension=1024,
        # *** NEW PARAMETERS FOR ALL FOUR HIDDEN LAYER DIMENSIONS ***
        hidden_dim1=2048, # The original 'hidden_dimension'
        hidden_dim2=1024, # New intermediate layer 2
        hidden_dim3=512,  # New intermediate layer 3
        hidden_dim4=256,  # New intermediate layer 4 (pre-latent)
        latent_activation="ReLU",
        latent_distance="Cosine",
        classify=True,
    ):
        super().__init__()
        self.drug_shape = drug_shape
        self.target_shape = target_shape
        self.latent_dimension = latent_dimension
        
        # Store all four hidden dimensions
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.hidden_dim3 = hidden_dim3
        self.hidden_dim4 = hidden_dim4 

        self.do_classify = classify
        self.latent_activation = ACTIVATIONS[latent_activation]

        # The structure is: Input -> H1 -> H2 -> H3 -> H4 -> Latent Output (5 total layers/4 hidden)

        # *** MODIFIED DRUG PROJECTOR (5 Layers Total) ***
        self.drug_projector = nn.Sequential(
            # Layer 1: Input (drug_shape) -> Hidden 1
            nn.Linear(self.drug_shape, self.hidden_dim1),
            self.latent_activation(),
            # Layer 2: Hidden 1 -> Hidden 2
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            self.latent_activation(),
            # Layer 3: Hidden 2 -> Hidden 3
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            self.latent_activation(),
            # Layer 4: Hidden 3 -> Hidden 4
            nn.Linear(self.hidden_dim3, self.hidden_dim4),
            self.latent_activation(),
            # Layer 5: Hidden 4 -> Latent Output
            nn.Linear(self.hidden_dim4, latent_dimension),
            self.latent_activation()
        )
        
        # Initialize weights for all five Linear layers (at indices 0, 2, 4, 6, 8)
        for i in [0, 2, 4, 6, 8]:
            nn.init.xavier_normal_(self.drug_projector[i].weight)


        # *** MODIFIED TARGET PROJECTOR (5 Layers Total) ***
        self.target_projector = nn.Sequential(
            # Layer 1: Input (target_shape) -> Hidden 1
            nn.Linear(self.target_shape, self.hidden_dim1),
            self.latent_activation(),
            # Layer 2: Hidden 1 -> Hidden 2
            nn.Linear(self.hidden_dim1, self.hidden_dim2),
            self.latent_activation(),
            # Layer 3: Hidden 2 -> Hidden 3
            nn.Linear(self.hidden_dim2, self.hidden_dim3),
            self.latent_activation(),
            # Layer 4: Hidden 3 -> Hidden 4
            nn.Linear(self.hidden_dim3, self.hidden_dim4),
            self.latent_activation(),
            # Layer 5: Hidden 4 -> Latent Output
            nn.Linear(self.hidden_dim4, latent_dimension),
            self.latent_activation()
        )
        
        # Initialize weights for all five Linear layers (at indices 0, 2, 4, 6, 8)
        for i in [0, 2, 4, 6, 8]:
            nn.init.xavier_normal_(self.target_projector[i].weight)


        if self.do_classify:
            self.distance_metric = latent_distance
            self.activator = DISTANCE_METRICS[self.distance_metric]()

    def forward(self, drug, target):
        if self.do_classify:
            return self.classify(drug, target)
        else:
            return self.regress(drug, target)

    def regress(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        # Calculates dot product (Inner product for regression)
        inner_prod = torch.bmm(
            drug_projection.view(-1, 1, self.latent_dimension),
            target_projection.view(-1, self.latent_dimension, 1),
        ).squeeze()
        return inner_prod.squeeze()

    def classify(self, drug, target):
        drug_projection = self.drug_projector(drug)
        target_projection = self.target_projector(target)

        # Uses the specified distance metric (e.g., Cosine Similarity)
        distance = self.activator(drug_projection, target_projection)
        return distance.squeeze()


class CNNBlock(nn.Module):
    """
    A single 1D Convolutional Block: Conv1D -> Activation -> MaxPool1D
    """
    def __init__(self, in_channels, out_channels, kernel_size, activation_fn):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=(kernel_size - 1) // 2)
        self.activation = activation_fn()
        # We can use MaxPool to downsample, or just let the convolutional layers do the work.
        # Here, we use a simple MaxPool with stride 2 for effective downsampling.
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) 
        
        # Initialize weights
        nn.init.kaiming_normal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        return x

class CNNDimensionalityReducer(nn.Module):
    def __init__(
        self,
        # *** Input vector lengths. These must be reshapeable to [Batch, Channels, Length] ***
        drug_input_length=2048, 
        target_input_length=1024,
        latent_dimension=1024,
        activation_type="ReLU",
        # --- CNN Hyperparameters ---
        initial_channels=1, # Assuming 1 channel (raw features)
        num_filters=64,
        kernel_size=5,
        num_conv_blocks=3
    ):
        super().__init__()
        self.latent_dimension = latent_dimension
        self.activation_fn = ACTIVATIONS[activation_type]
        self.num_conv_blocks = num_conv_blocks
        self.num_filters = num_filters
        self.initial_channels = initial_channels
        
        # Store input lengths for reshaping
        self.drug_input_length = drug_input_length
        self.target_input_length = target_input_length

        # --- Drug Feature Extractor ---
        drug_conv_layers = []
        in_c = initial_channels
        for i in range(num_conv_blocks):
            out_c = num_filters * (2 ** i) # Double the filters in each block
            drug_conv_layers.append(CNNBlock(in_c, out_c, kernel_size, self.activation_fn))
            in_c = out_c
        self.drug_feature_extractor = nn.Sequential(*drug_conv_layers)
        
        # --- Target Feature Extractor ---
        target_conv_layers = []
        in_c = initial_channels
        for i in range(num_conv_blocks):
            out_c = num_filters * (2 ** i) 
            target_conv_layers.append(CNNBlock(in_c, out_c, kernel_size, self.activation_fn))
            in_c = out_c
        self.target_feature_extractor = nn.Sequential(*target_conv_layers)

        # Calculate the final flattened size after the convolutions and pooling
        # This is non-trivial due to the dynamic pooling. We will calculate it in the forward pass
        # OR use Adaptive Pooling to fix the output size before the final Linear layer.
        
        # --- Final Projection Layer (Fixed size from Adaptive Pooling) ---
        
        # We'll use Adaptive Max Pooling to ensure a fixed size output of 1 element 
        # per channel, regardless of input length or previous pooling layers.
        # Final channels will be num_filters * (2 ** (num_conv_blocks - 1))
        final_channels = num_filters * (2 ** (num_conv_blocks - 1))
        
        # Linear layer to map the flattened features (final_channels * 1) to latent_dimension
        self.drug_projector = nn.Linear(final_channels, latent_dimension)
        self.target_projector = nn.Linear(final_channels, latent_dimension)
        
        nn.init.xavier_normal_(self.drug_projector.weight)
        nn.init.xavier_normal_(self.target_projector.weight)

    def _process_input(self, x, length):
        """Helper to reshape input vector to [Batch, Channels, Length] for Conv1D"""
        if x.dim() == 2:
            # Reshape [Batch, Length] -> [Batch, Channels, Length]
            return x.view(x.size(0), self.initial_channels, length)
        return x

    def forward(self, drug, target):
        # 1. Reshape inputs for 1D CNN processing
        drug = self._process_input(drug, self.drug_input_length)
        target = self._process_input(target, self.target_input_length)

        # 2. Extract features using the convolutional blocks
        drug_features = self.drug_feature_extractor(drug)
        target_features = self.target_feature_extractor(target)

        # 3. Apply Global Max Pooling to get a fixed-size vector
        # This takes the max value across the sequence length for each channel
        # [Batch, Channels, Length_out] -> [Batch, Channels, 1]
        drug_global_features = adaptive_max_pool1d(drug_features, 1)
        target_global_features = adaptive_max_pool1d(target_features, 1)

        # 4. Flatten the features: [Batch, Channels, 1] -> [Batch, Channels]
        drug_flat = drug_global_features.squeeze(-1)
        target_flat = target_global_features.squeeze(-1)

        # 5. Project the global features to the final latent dimension
        drug_projection = self.drug_projector(drug_flat)
        target_projection = self.target_projector(target_flat)

        return drug_projection, target_projection
