"""Instance Selection Utils.

Methods that are useful for multiple algorithms implemented in the instance
selection library.
"""
from ._transformer import transform, transform_original_complete, \
    delete_multiple_element

__all__ = [
    "transform",
    "transform_original_complete",
    "delete_multiple_element"
]
