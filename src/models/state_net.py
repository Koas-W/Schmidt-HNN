"""State-Net: q -> {v_k(q)}.

Shared trunk + multiple heads: one head for V (rank R_V), one head per lower-triangular slot of L (rank R_L);
the output is a packed tensor sliced by head_ranks.
"""
from .blocks import FeatureTrunk


class StateNet(FeatureTrunk):
    def __init__(self, n_dof: int, width: int, depth: int,
                 head_ranks: list[int], block: str = "swiglu"):
        super().__init__(n_dof, width, depth, head_ranks, block)
