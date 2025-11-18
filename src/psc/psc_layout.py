from dataclasses import dataclass


@dataclass
class PSCDecisionLayout:
    state_dim: int
    control_dim: int
    num_nodes: int

    @property
    def state_block(self) -> int:
        return self.state_dim * self.num_nodes

    @property
    def control_block(self) -> int:
        return self.control_dim * self.num_nodes

    @property
    def decision_dim(self) -> int:
        return self.state_block + self.control_block
