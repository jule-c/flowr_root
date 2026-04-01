import torch

from flowr.constants import ATOM_ENCODER as atom_encoder


def get_distributions(args, dataset_info, datamodule):
    histogram = dataset_info["n_nodes"]
    nodes_dist = DistributionNodes(histogram)
    return nodes_dist


class DistributionNodes:
    def __init__(self, histogram):
        """Compute the distribution of the number of nodes in the dataset, and sample from this distribution.
        historgram: dict. The keys are num_nodes, the values are counts
        """

        if type(histogram) is dict:
            max_n_nodes = max(histogram.keys())
            prob = torch.zeros(max_n_nodes + 1)
            for num_nodes, count in histogram.items():
                prob[num_nodes] = count
        else:
            prob = histogram

        self.prob = prob / prob.sum()
        self.m = torch.distributions.Categorical(prob)

    def sample_n(self, n_samples, device):
        idx = self.m.sample((n_samples,))
        return idx.to(device)

    def log_prob(self, batch_n_nodes):
        assert len(batch_n_nodes.size()) == 1
        probas = self.prob[batch_n_nodes.to(self.prob.device)]
        log_p = torch.log(probas + 1e-10)
        return log_p.to(batch_n_nodes.device)


class AbstractDatasetInfos:
    def complete_infos(self, statistics, atom_encoder):
        self.atom_decoder = [key for key in atom_encoder.keys()]
        self.num_atom_types = len(self.atom_decoder)

        # Train + val + test for n_nodes
        train_n_nodes = statistics["train"].num_nodes
        val_n_nodes = statistics["val"].num_nodes
        test_n_nodes = statistics["test"].num_nodes
        max_n_nodes = max(
            max(train_n_nodes.keys()), max(val_n_nodes.keys()), max(test_n_nodes.keys())
        )
        n_nodes = torch.zeros(max_n_nodes + 1, dtype=torch.long)
        for c in [train_n_nodes, val_n_nodes, test_n_nodes]:
            for key, value in c.items():
                n_nodes[key] += value

        self.n_nodes = n_nodes / n_nodes.sum()
        self.atom_types = statistics["train"].atom_types
        self.edge_types = statistics["train"].bond_types
        self.charges_types = statistics["train"].charge_types
        self.charges_marginals = (self.charges_types * self.atom_types[:, None]).sum(
            dim=0
        )
        self.valency_distribution = statistics["train"].valencies
        self.max_n_nodes = len(n_nodes) - 1
        self.nodes_dist = DistributionNodes(n_nodes)

        if hasattr(statistics["train"], "is_aromatic"):
            self.is_aromatic = statistics["train"].is_aromatic
        if hasattr(statistics["train"], "is_in_ring"):
            self.is_in_ring = statistics["train"].is_in_ring
        if hasattr(statistics["train"], "hybridization"):
            self.hybridization = statistics["train"].hybridization
        if hasattr(statistics["train"], "numHs"):
            self.numHs = statistics["train"].numHs
        if hasattr(statistics["train"], "is_h_donor"):
            self.is_h_donor = statistics["train"].is_h_donor
        if hasattr(statistics["train"], "is_h_acceptor"):
            self.is_h_acceptor = statistics["train"].is_h_acceptor


class GeneralInfos(AbstractDatasetInfos):
    def __init__(self, statistics, vocab, cfg):
        self.remove_h = cfg.remove_hs
        self.need_to_strip = (
            False  # to indicate whether we need to ignore one output from the model
        )
        self.statistics = statistics
        self.atom_encoder = atom_encoder
        self.name = "plinder"
        self.vocab = vocab
        self.num_bond_classes = 5
        self.num_charge_classes = 6
        self.charge_offset = 2
        self.collapse_charges = torch.Tensor([-2, -1, 0, 1, 2, 3]).int()

        super().complete_infos(self.statistics, self.atom_encoder)
