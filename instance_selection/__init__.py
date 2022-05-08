"""Instance Selection.

The package contains some of the most widely used instance selection algorithms
in the literature.
"""

__version__ = "2.0"
__author__ = "Daniel Puente Ramírez"

from ._CNN import CNN
from ._DROP3 import DROP3
from ._ENN import ENN
from ._ICF import ICF
from ._LocalSets import LSBo, LSSm
from ._MSS import MSS
from ._RNN import RNN

__all__ = ["ENN", "CNN", "RNN", "MSS", "ICF", "DROP3", "LSSm", "LSBo"]
