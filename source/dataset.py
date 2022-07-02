from sklearn.utils import shuffle


class DataSet:
    def __init__(self, As, Ss, coarse_nodes_list, baseline_P_list, sparsity_patterns):
        self.As = As
        self.Ss = Ss
        self.coarse_nodes_list = coarse_nodes_list
        self.baseline_P_list = baseline_P_list
        self.sparsity_patterns = sparsity_patterns

    def shuffle(self):
        As, Ss, coarse_nodes_list, baseline_P_list, sparsity_patterns = shuffle(self.As,
                                                             self.Ss,
                                                             self.coarse_nodes_list,
                                                             self.baseline_P_list,
                                                             self.sparsity_patterns)
        return DataSet(As, Ss, coarse_nodes_list, baseline_P_list, sparsity_patterns)

    def __getitem__(self, item):
        return DataSet(
            self.As[item],
            self.Ss[item],
            self.coarse_nodes_list[item],
            self.baseline_P_list[item],
            self.sparsity_patterns[item]
        )

    def __add__(self, other):
        return DataSet(
            self.As + other.As,
            self.Ss + other.Ss,
            self.coarse_nodes_list + other.coarse_nodes_list,
            self.baseline_P_list + other.baseline_P_list,
            self.sparsity_patterns + other.sparsity_patterns
        )
