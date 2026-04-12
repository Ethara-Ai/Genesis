import numpy as np
import torch


def brief(x):
    pass


def __repr_name__(x):
    """
    Only used for non-genesis object by `brief()`.
    To convert <class 'classname'> into <classname>.
    """
    if isinstance(x, type):
        raw_class_name = str(x)
    else:
        raw_class_name = str(x.__class__)
    full_name = f"<{' '.join(raw_class_name.split(' ')[1:])[1:-2]}>"
    return full_name
