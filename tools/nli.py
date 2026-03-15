"""
NLI
"""
from enum import Enum


class NLIClassifications(Enum):
    """
    The names of the NLI classification.
    """
    CONTRADICTION = 'contradiction'
    ENTAILMENT = 'entailment'
    NEUTRAL = 'neutral'
