"""
Data module for the ev0x project.

This module provides access to datasets and data processing utilities
for training and evaluating AI models within the ev0x framework.
"""

from src.data.datasets import DatasetManager, FTSODataset, GitHubDataset, TrendsDataset

__all__ = ['DatasetManager', 'FTSODataset', 'GitHubDataset', 'TrendsDataset']

