"""
Evolution module for the ev0x project.

This module contains components related to the evolutionary model selection
and meta-intelligence systems that allow the ev0x framework to adaptively
improve over time by learning from model behaviors.
"""

from src.evolution.selection import AdaptiveModelSelector
from src.evolution.meta_intelligence import MetaIntelligence, ModelProfile

__all__ = ['AdaptiveModelSelector', 'MetaIntelligence', 'ModelProfile']

