#!/usr/bin/env python3
"""
Blockchain Consensus Tests - Verification and validation of consensus mechanisms
using Flare Network data and contract interactions.
"""

import os
import sys
import time
import json
import logging
import argparse
import datetime
import requests
from typing import Dict, List, Any, Optional, Tuple
from web3 import Web3
from web3.exceptions import BlockNotFound, ContractLogicError, TransactionNotFound

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Flare Network RPC endpoints - use multiple for redundancy
FLARE_RPC_ENDPOINTS = [
    "https://flare-api.flare.network/ext/C/rpc",
    "https://flare.blockpi.network/v1/rpc/public"
]

# Contract addresses on Flare Network
CONTRACTS = {
    "ftso_manager": "0x1000000000000000000000000000000000000003",
    "price_submitter": "0x1000000000000000000000000000000000000003",
    "ftso_registry": "0x1000000000000000000000000000000000000002",
    "ftso_reward_manager": "0xc5738334b972745067a62395d442458209c28d84",
    "state_connector": "0x1000000000000000000000000000000000000001",
    "flare_asset_registry": "0x1000000000000000000000000000000000000001",
}

# ABI fragments for the contracts we'll interact with
FTSO_MANAGER_ABI = [
    {"type": "function", "name": "getCurrentRewardEpoch", "inputs": [], "outputs": [{"type": "uint256"}]},
    {"type": "function", "name": "getPriceEpochConfiguration", "inputs": [], "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}]},
    {"type": "function", "name": "getRewardEpochVotePowerBlock", "inputs": [{"type": "uint256"}], "outputs": [{"type": "uint256"}]}
]

FTSO_REGISTRY_ABI = [
    {"type": "function", "name": "getSupportedSymbols", "inputs": [], "outputs": [{"type": "string[]"}]},
    {"type": "function", "name": "getSupportedIndices", "inputs": [], "outputs": [{"type": "uint256[]"}]},
    {"type": "function", "name": "getFtsoBySymbol", "inputs": [{"type": "string"}], "outputs": [{"type": "address"}]}
]

FTSO_ABI = [
    {"type": "function", "name": "getCurrentPrice", "inputs": [], "outputs": [{"type": "uint256"}, {"type": "uint256"}]},
    {"type": "function", "name": "getPriceEpochData", "inputs": [{"type": "uint256"}], "outputs": [{"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}, {"type": "uint256"}]},
    {"type": "function", "name": "symbol", "inputs": [], "outputs": [{"type": "string"}]}
]


class BlockchainConsensusTests:
    """
    Tests consensus mechanisms on the Flare Network using real blockchain data.
    
    Validates consensus data, block timing, and proper execution flow for
    decentralized consensus mechanisms.
    """
    
    def __init__(self, rpc_endpoints: List[str] = None):
        """
        Initialize the blockchain consensus tests with RPC endpoints.
        
        Args:
            rpc_endpoints: List of RPC endpoints to use for redundancy.
        """
        self.rpc_endpoints = rpc_endpoints or FLARE_RPC_ENDPOINTS
        self.web3 = None
        self.contracts = {}
        self.results = {
            "block_timing": [],
            "consensus_metrics": {},
            "verification_results": [],
            "errors": []
        }
        self._setup_web3_connection()
        
    def _setup_web3_connection(self) -> None:
        """Set up the connection to the Flare Network."""
        for endpoint in self.rpc_endpoints:
            try:
                web3 = Web3(Web3.HTTPProvider(endpoint))
                if web3.is_connected():
                    logger.info(f"Connected to Flare Network at {endpoint}")
                    self.web3 = web3
                    self._setup_contracts()
                    return
            except Exception as e:
                logger.warning(f"Failed to connect to {endpoint}: {e}")
        
        if not self.web3:
            logger.error("Failed to connect to any Flare Network endpoint")
            raise ConnectionError("Cannot connect to Flare Network")
    
    def _setup_contracts(self) -> None:
        """Initialize contract interfaces using real addresses."""
        try:
            # Set up FTSO Manager contract
            self.contracts["ftso_manager"] = self.web3.eth.contract(
                address=self.web3.to_checksum_address(CONTRACTS["ftso_manager"]),
                abi=FTSO_MANAGER_ABI
            )
            
            # Set up FTSO Registry contract
            self.contracts["ftso_registry"] = self.web3.eth.contract(
                address=self.web3.to_checksum_address(CONTRACTS["ftso_registry"]),
                abi=FTSO_REGISTRY_ABI
            )
            
            logger.info("Successfully initialized contract interfaces")
        except Exception as e:
            logger.error(f"Error initializing contracts: {e}")
            raise
    
    def _verify_consensus_data(self, block_range: Tuple[int, int]) -> Dict[str, Any]:
        """
        Verify consensus data using real block timing analysis.
        
        Args:
            block_range: Tuple of (start_block, end_block) to analyze
            
        Returns:
            Dictionary with verification results
        """
        start_block, end_block = block_range
        logger.info(f"Analyzing blocks from {start_block} to {end_block}")
        
        block_times = []
        consensus_failures = 0
        timing_anomalies = 0
        verification_results = {}
        
        try:
            # Fetch all blocks in range
            blocks = []
            for block_num in range(start_block, end_block + 1):
                try:
                    block = self.web3.eth.get_block(block_num, full_transactions=False)
                    blocks.append(block)
                except BlockNotFound:
                    logger.warning(f"Block {block_num} not found")
                    continue
            
            # Real block timing analysis
            for i in range(1, len(blocks)):
                prev_block = blocks[i-1]
                curr_block = blocks[i]
                
                # Calculate time difference between blocks in seconds
                time_diff = curr_block.timestamp - prev_block.timestamp
                block_times.append(time_diff)
                
                # Check for timing anomalies (significantly delayed blocks)
                if time_diff > 15:  # Flare typically has ~5 sec block times
                    timing_anomalies += 1
                    logger.info(f"Timing anomaly detected between blocks {prev_block.number} and {curr_block.number}: {time_diff} seconds")
                
                # Analyze block finality status
                if 'finality' in curr_block and curr_block.finality < 0.9:
                    consensus_failures += 1
                    logger.warning(f"Low finality score for block {curr_block.number}: {curr_block.finality}")
            
            # Calculate statistics
            avg_block_time = sum(block_times) / len(block_times) if block_times else 0
            max_block_time = max(block_times) if block_times else 0
            min_block_time = min(block_times) if block_times else 0
            
            # Analyze variance in block times - high variance indicates consensus issues
            variance = sum((t - avg_block_time) ** 2 for t in block_times) / len(block_times) if block_times else 0
            std_deviation = variance ** 0.5
            
            # Check for validator participation by analyzing extra data field in blocks
            validator_participation = {}
            for block in blocks:
                if hasattr(block, 'extraData') and len(block.extraData) >= 2:
                    validator_id = block.extraData[:8]  # First 4 bytes often contain validator ID
                    validator_participation[validator_id] = validator_participation.get(validator_id, 0) + 1
            
            # Analyze participation distribution - more even distribution indicates healthier consensus
            participation_values = list(validator_participation.values())
            participation_variance = sum((v - (sum(participation_values) / len(participation_values))) ** 2 
                                         for v in participation_values) / len(participation_values) if participation_values else 0
            
            # Compile results
            verification_results = {
                "blocks_analyzed": len(blocks),
                "avg_block_time": avg_block_time,
                "max_block_time": max_block_time,
                "min_block_time": min_block_time,
                "block_time_std_deviation": std_deviation,
                "timing_anomalies": timing_anomalies,
                "consensus_failures": consensus_failures,
                "unique_validators": len(validator_participation),
                "validator_participation_variance": participation_variance,
                "consensus_health_score": self._calculate_consensus_health(
                    avg_block_time, std_deviation, timing_anomalies, 
                    consensus_failures, participation_variance
                )
            }
            
            logger.info(f"Block timing analysis completed for {len(blocks)} blocks")
            logger.info(f"Average block time: {avg_block_time:.2f} seconds")
            logger.info(f"Timing anomalies: {timing_anomalies}")
            logger.info(f"Consensus health score: {verification_results['consensus_health_score']:.2f}")
            
        except Exception as e:
            logger.error(f"Error during consensus data verification: {e}")
            verification_results = {
                "error": str(e),
                "status": "failed"
            }
        
        return verification_results
    
    def _calculate_consensus_health(self, avg_time: float, std_dev: float, 
                                   anomalies: int, failures: int, 
                                   participation_variance: float) -> float:
        """
        Calculate a consensus health score based on multiple metrics.
        
        Args:
            avg_time: Average block time
            std_dev: Standard deviation in block times
            anomalies: Number of timing anomalies
            failures: Number of consensus failures
            participation_variance: Variance in validator participation
            
        Returns:
            Consensus health score (0-100)
        """
        # Ideal values for Flare Network
        ideal_block_time = 5.0
        ideal_std_dev = 1.0
        
        # Calculate components of health score
        time_score = max(0, 100 - abs(avg_time - ideal_block_time) * 5)
        std_dev_score = max(0, 100 - (std_dev - ideal_std_dev) * 10)
        anomaly_score = max(0, 100 - anomalies * 5)
        failure_score = max(0, 100 - failures * 10)
        participation_score = max(0, 100 - participation_variance * 50)
        
        # Weight the components
        weighted_score = (
            time_score * 0.2 + 
            std_dev_score * 0.2 + 
            anomaly_score * 0.2 + 
            failure_score * 0.3 + 
            participation_score * 0.1
        )
        
        return weighted_score
    
    def _check_ftso_consensus(self) -> Dict[str, Any]:
        """
        Check FTSO price submission consensus using real data.
        
        Returns:
            Dictionary with FTSO consensus metrics
        """
        logger.info("Checking FTSO consensus with real price data")
        ftso_results = {}
        
        try:
            # Get current reward epoch
            current_epoch = self.contracts["ftso_manager"].functions.getCurrentRewardEpoch().call()
            logger.info(f"Current reward epoch: {current_epoch}")
            
            # Get price epoch configuration
            price_epoch_config = self.contracts["ftso_manager"].functions.getPriceEpochConfiguration().call()
            price_epoch_duration = price_epoch_config[0]
            logger.info(f"Price epoch duration: {price_epoch_duration} seconds")
            
            # Get supported symbols
            supported_symbols = self.contracts["ftso_registry"].functions.getSupportedSymbols().call()
            logger.info(f"Found {len(supported_symbols)} supported price symbols")
            
            consensus_metrics = {}
            
            # Analyze price data for each symbol
            for symbol in supported_symbols[:5]:  # Limit to first 5 symbols to avoid excessive API calls
                try:
                    # Get the FTSO contract for this symbol
                    ftso_address = self.contracts["ftso_registry"].functions.getFtsoBySymbol(symbol).call()
                    ftso_contract = self.web3.eth.contract(
                        address=self.web3.to_checksum_address(ftso_address),
                        abi=FTSO_ABI
                    )
                    
                    # Get current price data
                    current_price_data = ftso_contract.functions.getCurrentPrice().call()
                    price = current_price_data[0]
                    timestamp = current_price_data[1]
                    
                    # Get the last few price epochs for consistency analysis
                    price_history = []
                    current_time = int(time.time())
                    for i in range(5):  # Look at 5 recent price epochs
                        epoch_id = (current_time // price_epoch_duration) - i
                        try:
                            epoch_data = ftso_contract.functions.getPriceEpochData(epoch_id).call()
                            price_history.append({
                                "epoch_id": epoch_id,
                                "price": epoch_data[0],
                                "decimals": epoch_data[1],
                                "timestamp": epoch_data[2],
                                "finalized": epoch_data[3] > 0
                            })
                        except ContractLogicError:
                            # This epoch might not be available yet
                            continue
                    
                    # Calculate price consistency metrics
                    if price_history:
                        # Calculate price variance
                        prices = [entry["price"] for entry in price_history if entry["finalized"]]
                        if prices:
                            avg_price = sum(prices) / len(prices)
                            price_variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
                            price_std_dev = price_variance ** 0.5
                            
                            # Calculate relative price movement
                            price_movement = abs(prices[0] - prices[-1]) / prices[-1] if len(prices) > 1 else 0
                            
                            consensus_metrics[symbol] = {
                                "current_price": price

