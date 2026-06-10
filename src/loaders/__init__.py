"""Microgrid Digital Twin — data loaders.

Single-dataset focus: UCI Electrical Grid Stability Simulated Data
(Schafer et al., Eur. Phys. J. Special Topics 225, 569 (2016)).
"""

from .load_uci_grid import load_uci_grid

__all__ = ["load_uci_grid"]
