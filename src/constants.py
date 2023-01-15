
MAX_BATCH_SIZE = 100

SENSITIVITY_ONE = [
    "pre_drw",
    "drw",
    "pre_drw_w_restarts",
    "uniform"
]

NON_DP_METHODS = [
    "rw"
]

DP_METHODS = [
    "pre_drw",
    "drw",
    "baseline",
    "pre_drw_w_restarts",
    "uniform"
]

RDP_ACCOUNTANT = [
    "baseline",
    "rdp_poisson_autodp",
    "rdp_uniform_autodp"
]

AUTODP_ACCOUNTANT = [
    "rdp_poisson_autodp",
    "rdp_uniform_autodp"
]

BOUND_DEGREE_METHODS = [
    "drw",
    "baseline"
]

POISSON_ACCOUNTANT = [
    "rdp_poisson_autodp"
]

TRANSDUCTIVE_DATASETS = [
    "cora",
    "citeseer",
    "pubmed"
]
