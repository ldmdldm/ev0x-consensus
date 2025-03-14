"""Dataset access implementation for the ev0x project."""

import logging
import pandas as pd
from typing import Dict, List, Optional, Any, TypeVar, Union, Callable
from pandas import DataFrame
from abc import ABC, abstractmethod
from google.cloud import bigquery

# Initialize logger
logger = logging.getLogger(__name__)


class BaseDataset(ABC):
    """Base class for all datasets."""

    @abstractmethod
    def get_data(self, **kwargs: Any) -> DataFrame:
        """Get data from the dataset."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the dataset."""


class FTSODataset(BaseDataset):
    """Dataset access for FTSO data."""

    def __init__(self, dataset_type: str = "block-latency"):
        """
        Initialize FTSO dataset client.

        Args:
            dataset_type: Type of FTSO data ('block-latency' or 'anchor-feeds')
        """
        self.client = bigquery.Client()
        self.dataset_type = dataset_type

        # Define dataset IDs
        self.block_latency_dataset = "public.ftso_block_latency"
        self.anchor_feeds_dataset = "public.ftso_anchor_feeds"

    def get_data(self, limit: int = 1000, offset: int = 0, filters: Optional[Dict[str, Any]] = None) -> DataFrame:
        """
        Get FTSO data from BigQuery.

        Args:
            limit: Maximum number of rows to return
            offset: Number of rows to skip
            filters: Dictionary of column/value pairs to filter by

        Returns:
            Pandas DataFrame with the requested data
        """
        # Select appropriate dataset
        dataset_id = self.block_latency_dataset if self.dataset_type == "block-latency" else self.anchor_feeds_dataset

        # Build query
        query = f"SELECT * FROM `{dataset_id}`"

        # Add filters if provided
        if filters:
            where_clauses = []
            for col, val in filters.items():
                if isinstance(val, str):
                    where_clauses.append(f"{col} = '{val}'")
                else:
                    where_clauses.append(f"{col} = {val}")

            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)

        # Add limit and offset
        query += f" LIMIT {limit} OFFSET {offset}"

        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"Error fetching FTSO data: {e}")
            return DataFrame()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the FTSO dataset."""
        dataset_id = self.block_latency_dataset if self.dataset_type == "block-latency" else self.anchor_feeds_dataset

        try:
            # Get table schema
            table = self.client.get_table(dataset_id)

            return {
                "name": f"FTSO {self.dataset_type}",
                "description": f"FTSO {self.dataset_type} dataset from BigQuery",
                "columns": [field.name for field in table.schema],
                "num_rows": table.num_rows,
                "size_bytes": table.num_bytes,
                "latency": "2s" if self.dataset_type == "block-latency" else "90s"
            }
        except Exception as e:
            logger.error(f"Error fetching FTSO metadata: {e}")
            return {"error": str(e)}


class DatasetManager:
    """Unified interface for accessing all datasets."""

    def __init__(self):
        """Initialize DatasetManager with all available datasets."""
        self.datasets = {
            "ftso_block_latency": FTSODataset(dataset_type="block-latency"),
            "ftso_anchor_feeds": FTSODataset(dataset_type="anchor-feeds"),
            "github": GitHubDataset(),
            "trends": TrendsDataset()
        }

    def get_dataset(self, dataset_name: str) -> Optional[BaseDataset]:
        """
        Get a dataset by name.

        Args:
            dataset_name: Name of the dataset to retrieve

        Returns:
            Dataset instance or None if not found
        """
        if dataset_name not in self.datasets:
            logger.error(f"Dataset {dataset_name} not found")
            return None
        return self.datasets[dataset_name]

    def list_datasets(self) -> List[str]:
        """
        List all available datasets.

        Returns:
            List of dataset names
        """
        return list(self.datasets.keys())

    def get_metadata_for_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all datasets.

        Returns:
            Dictionary mapping dataset names to their metadata
        """
        result = {}
        for name, dataset in self.datasets.items():
            result[name] = dataset.get_metadata()
        return result

    def query_across_datasets(self, query_func: Callable[[BaseDataset], DataFrame]) -> Dict[str, DataFrame]:
        """
        Execute a query function across all datasets.

        Args:
            query_func: Function that takes a dataset and returns a DataFrame

        Returns:
            Dictionary mapping dataset names to query results
        """
        results = {}
        for name, dataset in self.datasets.items():
            try:
                results[name] = query_func(dataset)
            except Exception as e:
                logger.error(f"Error querying dataset {name}: {e}")
                results[name] = DataFrame()
        return results


class TrendsDataset(BaseDataset):
    """Dataset access for Google Trends data."""

    def __init__(self):
        """Initialize Google Trends dataset client."""
        self.client = bigquery.Client()
        self.dataset_id = "public.google_trends"

    def get_data(self, keyword: Optional[str] = None, date_from: Optional[str] = None,
                 date_to: Optional[str] = None, limit: int = 1000) -> DataFrame:
        """
        Get Google Trends data.

        Args:
            keyword: Keyword to filter by
            date_from: Start date in YYYY-MM-DD format
            date_to: End date in YYYY-MM-DD format
            limit: Maximum number of rows to return

        Returns:
            Pandas DataFrame with Google Trends data
        """
        query = f"SELECT * FROM `{self.dataset_id}`"

        # Add filters
        where_clauses = []
        if keyword:
            where_clauses.append(f"keyword = '{keyword}'")
        if date_from:
            where_clauses.append(f"date >= '{date_from}'")
        if date_to:
            where_clauses.append(f"date <= '{date_to}'")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        # Add limit
        query += f" LIMIT {limit}"

        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"Error fetching Google Trends data: {e}")
            return DataFrame()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the Google Trends dataset."""
        try:
            # Get table schema
            table = self.client.get_table(self.dataset_id)

            return {
                "name": "Google Trends",
                "description": "Historical trend data from Google Trends",
                "columns": [field.name for field in table.schema],
                "num_rows": table.num_rows,
                "size_bytes": table.num_bytes,
                "latency": "24h"
            }
        except Exception as e:
            logger.error(f"Error fetching Google Trends metadata: {e}")
            return {"error": str(e)}


class GitHubDataset(BaseDataset):
    """Dataset access for GitHub data."""

    def __init__(self):
        """Initialize GitHub dataset client."""
        self.client = bigquery.Client()
        self.dataset_id = "public.github_activity"

    def get_data(self, repo: Optional[str] = None, event_type: Optional[str] = None,
                 limit: int = 1000) -> DataFrame:
        """
        Get GitHub activity data.

        Args:
            repo: Repository name to filter by (e.g., 'flare-foundation/flare')
            event_type: Event type to filter by (e.g., 'PushEvent', 'IssueEvent')
            limit: Maximum number of rows to return

        Returns:
            Pandas DataFrame with GitHub activity data
        """
        query = f"SELECT * FROM `{self.dataset_id}`"

        # Add filters
        where_clauses = []
        if repo:
            where_clauses.append(f"repo_name = '{repo}'")
        if event_type:
            where_clauses.append(f"type = '{event_type}'")

        if where_clauses:
            query += " WHERE " + " AND ".join(where_clauses)

        # Add limit
        query += f" LIMIT {limit}"

        try:
            return self.client.query(query).to_dataframe()
        except Exception as e:
            logger.error(f"Error fetching GitHub data: {e}")
            return DataFrame()

    def get_metadata(self) -> Dict[str, Any]:
        """Get metadata about the GitHub dataset."""
        try:
            # Get table schema
            table = self.client.get_table(self.dataset_id)

            return {
                "name": "GitHub Activity",
                "description": "GitHub activity data from public repositories",
                "columns": [field.name for field in table.schema],
                "num_rows": table.num_rows,
                "size_bytes": table.num_bytes,
                "latency": "30s"
            }
        except Exception as e:
            logger.error(f"Error fetching GitHub metadata: {e}")
            return {"error": str(e)}


class DatasetLoader:
    """
    Wrapper around DatasetManager to provide the interface expected by main.py.
    This class maintains API compatibility while delegating functionality to DatasetManager.
    """

    def __init__(self):
        """Initialize DatasetLoader with a DatasetManager instance."""
        self.manager = DatasetManager()

    def get_dataset(self, dataset_name: str) -> Optional[BaseDataset]:
        """
        Get a dataset by name.

        Args:
            dataset_name: Name of the dataset to retrieve

        Returns:
            Dataset instance or None if not found
        """
        return self.manager.get_dataset(dataset_name)

    def list_datasets(self) -> List[str]:
        """
        List all available datasets.

        Returns:
            List of dataset names
        """
        return self.manager.list_datasets()

    def get_metadata_for_all(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata for all datasets.

        Returns:
            Dictionary mapping dataset names to their metadata
        """
        return self.manager.get_metadata_for_all()

    def query_across_datasets(self, query_func: Callable[[BaseDataset], DataFrame]) -> Dict[str, DataFrame]:
        """
        Execute a query function across all datasets.

        Args:
            query_func: Function that takes a dataset and returns a DataFrame

        Returns:
            Dictionary mapping dataset names to query results
        """
        return self.manager.query_across_datasets(query_func)

    def load_dataset(self, dataset_name: str, **kwargs: Any) -> DataFrame:
        """
        Load data from a specific dataset with optional parameters.
        This method provides a simplified interface for accessing dataset data.

        Args:
            dataset_name: Name of the dataset to load
            **kwargs: Additional parameters to pass to the dataset's get_data method

        Returns:
            Pandas DataFrame with the dataset contents or empty DataFrame if dataset not found
        """
        dataset = self.get_dataset(dataset_name)
        if dataset:
            return dataset.get_data(**kwargs)
        logger.error(f"Dataset {dataset_name} not found when attempting to load data")
        return DataFrame()
