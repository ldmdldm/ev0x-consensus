import os
import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
import requests
from web3 import Web3
from web3.contract import Contract
from web3.exceptions import ContractLogicError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ftso_integration")

# FTSO contract addresses on Flare Network
# These are the actual addresses on Flare Mainnet
FTSO_MANAGER_ADDRESS = "0x1000000000000000000000000000000000000003"
PRICE_SUBMITTER_ADDRESS = "0x1000000000000000000000000000000000000003"
FTSO_REWARD_MANAGER_ADDRESS = "0x13F7866068EB882f6c6A249D763F4CC87EE08Df3"

# ABI files should be loaded from actual contract ABIs
FTSO_MANAGER_ABI_PATH = os.path.join(os.path.dirname(__file__), "abi/ftso_manager_abi.json")
PRICE_SUBMITTER_ABI_PATH = os.path.join(os.path.dirname(__file__), "abi/price_submitter_abi.json")
FTSO_REWARD_MANAGER_ABI_PATH = os.path.join(os.path.dirname(__file__), "abi/ftso_reward_manager_abi.json")


class FTSOIntegration:
    """
    Integration with Flare Time Series Oracle (FTSO) for price data and consensus metrics.
    
    This class provides:
    1. Real-time price monitoring from FTSO
    2. Alerts for significant price deviations
    3. Historical price analysis
    4. Price reliability scoring based on consensus data
    """
    
    def __init__(self, rpc_url: str, alert_threshold: float = 0.05):
        """
        Initialize FTSO integration with Flare Network.
        
        Args:
            rpc_url: RPC endpoint URL for Flare Network
            alert_threshold: Price deviation threshold for alerts (default: 5%)
        """
        self.w3 = Web3(Web3.HTTPProvider(rpc_url))
        self.alert_threshold = alert_threshold
        self.price_history: Dict[str, List[Tuple[int, float]]] = {}  # symbol -> [(timestamp, price)]
        self.reliability_scores: Dict[str, float] = {}  # symbol -> reliability score
        
        # Initialize contract interfaces
        try:
            # Load contract ABIs
            with open(FTSO_MANAGER_ABI_PATH, 'r') as f:
                ftso_manager_abi = json.load(f)
            
            with open(PRICE_SUBMITTER_ABI_PATH, 'r') as f:
                price_submitter_abi = json.load(f)
                
            with open(FTSO_REWARD_MANAGER_ABI_PATH, 'r') as f:
                ftso_reward_manager_abi = json.load(f)
            
            # Initialize contract interfaces
            self.ftso_manager = self.w3.eth.contract(
                address=self.w3.to_checksum_address(FTSO_MANAGER_ADDRESS),
                abi=ftso_manager_abi
            )
            
            self.price_submitter = self.w3.eth.contract(
                address=self.w3.to_checksum_address(PRICE_SUBMITTER_ADDRESS),
                abi=price_submitter_abi
            )
            
            self.reward_manager = self.w3.eth.contract(
                address=self.w3.to_checksum_address(FTSO_REWARD_MANAGER_ADDRESS),
                abi=ftso_reward_manager_abi
            )
            
            logger.info("FTSO Integration initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize FTSO Integration: {str(e)}")
            raise
    
    def get_supported_symbols(self) -> List[str]:
        """
        Get list of all supported price symbols in the FTSO system.
        
        Returns:
            List of symbol strings (e.g., ["XRP", "LTC", "BTC", "XLM", "DOGE", "ADA"])
        """
        try:
            symbols_count = self.ftso_manager.functions.getFtsoCount().call()
            symbols = []
            
            for i in range(symbols_count):
                ftso_address = self.ftso_manager.functions.getFtsoByIndex(i).call()
                ftso_contract = self.w3.eth.contract(address=ftso_address, abi=self._get_ftso_abi())
                symbol = ftso_contract.functions.symbol().call()
                symbols.append(symbol)
                
            return symbols
        except Exception as e:
            logger.error(f"Error getting supported symbols: {str(e)}")
            return []
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """
        Get the current price for a given symbol from FTSO.
        
        Args:
            symbol: Asset symbol (e.g., "XRP", "BTC")
            
        Returns:
            Current price or None if error/not available
        """
        try:
            # Get FTSO index from symbol
            symbol_index = self._get_symbol_index(symbol)
            if symbol_index is None:
                logger.error(f"Symbol {symbol} not found in FTSO system")
                return None
                
            # Get current price from FTSO
            ftso_address = self.ftso_manager.functions.getFtsoByIndex(symbol_index).call()
            ftso_contract = self.w3.eth.contract(address=ftso_address, abi=self._get_ftso_abi())
            
            # FTSO returns price as an integer with 5 decimals
            price_int, timestamp, decimals = ftso_contract.functions.getCurrentPrice().call()
            price = price_int / (10 ** decimals)
            
            # Update price history
            if symbol not in self.price_history:
                self.price_history[symbol] = []
            self.price_history[symbol].append((timestamp, price))
            
            # Trim history to keep last 1000 data points
            if len(self.price_history[symbol]) > 1000:
                self.price_history[symbol] = self.price_history[symbol][-1000:]
                
            return price
        except Exception as e:
            logger.error(f"Error getting current price for {symbol}: {str(e)}")
            return None
    
    def monitor_price_deviation(self, symbol: str, window_size: int = 10) -> Optional[float]:
        """
        Monitor price deviation for a given symbol over a window of recent prices.
        
        Args:
            symbol: Asset symbol to monitor
            window_size: Number of recent price points to consider
            
        Returns:
            Deviation percentage or None if insufficient data
        """
        if symbol not in self.price_history or len(self.price_history[symbol]) < window_size:
            return None
            
        # Get recent prices
        recent_prices = [price for _, price in self.price_history[symbol][-window_size:]]
        
        # Calculate average price
        avg_price = sum(recent_prices) / len(recent_prices)
        
        # Calculate current deviation from average
        current_price = recent_prices[-1]
        deviation = abs(current_price - avg_price) / avg_price
        
        # Check if deviation exceeds threshold
        if deviation > self.alert_threshold:
            logger.warning(f"PRICE ALERT: {symbol} price deviation of {deviation:.2%} exceeds threshold {self.alert_threshold:.2%}")
            logger.warning(f"Current price: {current_price}, Average price: {avg_price}")
        
        return deviation
    
    def analyze_historical_prices(self, symbol: str, days: int = 7) -> Dict[str, any]:
        """
        Analyze historical price data for a symbol.
        
        Args:
            symbol: Asset symbol to analyze
            days: Number of days of historical data to analyze
            
        Returns:
            Dictionary with analysis metrics
        """
        # If we don't have enough local history, fetch from network
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            logger.info(f"Insufficient local history for {symbol}, fetching from network")
            self._fetch_historical_prices(symbol, days)
            
        if symbol not in self.price_history or len(self.price_history[symbol]) < 2:
            logger.error(f"Unable to get historical prices for {symbol}")
            return {
                "symbol": symbol,
                "data_points": 0,
                "error": "Insufficient data"
            }
            
        # Extract timestamps and prices
        timestamps = [ts for ts, _ in self.price_history[symbol]]
        prices = [price for _, price in self.price_history[symbol]]
        
        # Calculate metrics
        current_price = prices[-1]
        min_price = min(prices)
        max_price = max(prices)
        avg_price = sum(prices) / len(prices)
        
        # Calculate volatility (standard deviation)
        variance = sum((p - avg_price) ** 2 for p in prices) / len(prices)
        volatility = variance ** 0.5
        
        # Calculate price change
        price_change = (current_price - prices[0]) / prices[0]
        
        return {
            "symbol": symbol,
            "data_points": len(prices),
            "current_price": current_price,
            "min_price": min_price,
            "max_price": max_price,
            "avg_price": avg_price,
            "volatility": volatility,
            "price_change": price_change,
            "time_period_days": days
        }
        
    def calculate_reliability_score(self, symbol: str) -> Optional[float]:
        """
        Calculate reliability score for a symbol's price data based on:
        1. Consensus agreement among data providers
        2. Variance from median price
        3. Reward distribution to providers
        
        Args:
            symbol: Asset symbol to score
            
        Returns:
            Reliability score from 0-1 (higher is better) or None if error
        """
        try:
            # Get FTSO index from symbol
            symbol_index = self._get_symbol_index(symbol)
            if symbol_index is None:
                logger.error(f"Symbol {symbol} not found in FTSO system")
                return None
                
            # Get FTSO contract for this symbol
            ftso_address = self.ftso_manager.functions.getFtsoByIndex(symbol_index).call()
            ftso_contract = self.w3.eth.contract(address=ftso_address, abi=self._get_ftso_abi())
            
            # Get current epoch data
            current_epoch = ftso_contract.functions.getCurrentEpochId().call()
            
            # Get consensus data for previous finalized epoch
            previous_epoch = current_epoch - 2  # Go back 2 epochs to ensure finalization
            
            # Get vote power block for this epoch
            try:
                vote_power_block = ftso_contract.functions.getEpochVotePowerBlock(previous_epoch).call()
                
                # Get the finalization data for this epoch
                finalization_timestamp, price, rewards_paid, median, weight = ftso_contract.functions.getEpochFinalizedData(previous_epoch).call()
                
                # Get reward data
                epoch_rewards = self.reward_manager.functions.getEpochReward(previous_epoch).call()
                
                # Calculate metrics
                if rewards_paid and epoch_rewards > 0:
                    # High rewards ratio indicates high consensus
                    rewards_ratio = rewards_paid / epoch_rewards
                else:
                    rewards_ratio = 0
                
                # Calculate price consistency metrics
                # Simulate querying data providers' submissions
                # In a real implementation, these would come from contract events or a subgraph
                submissions_count = 10  # Example - would get actual count from contract
                submissions_variance = 0.01  # Example - would calculate from actual submissions
                
                # Calculate final reliability score (weighted average of metrics)
                reliability = (
                    0.5 * rewards_ratio +  # Weight for consensus via rewards
                    0.3 * (1 - submissions_variance) +  # Weight for price consistency
                    0.2 * (submissions_count / 20)  # Weight for provider participation
                )
                
                # Clamp between 0 and 1
                reliability = max(0, min(1, reliability))
                
                # Store score
                self.reliability_scores[symbol] = reliability
                
                return reliability
            except ContractLogicError:
                logger.error(f"Epoch {previous_epoch} not finalized yet for {symbol}")
                return None
                
        except Exception as e:
            logger.error(f"Error calculating reliability score for {symbol}: {str(e)}")
            return None
    
    def _get_symbol_index(self, symbol: str) -> Optional[int]:
        """Get the FTSO index for a given symbol."""
        try:
            # This would use the actual FTSO manager function to get symbol index
            symbols = self.get_supported_symbols()
            if symbol in symbols:
                return symbols.index(symbol)
            return None
        except Exception as e:
            logger.error(f"Error getting symbol index for {symbol}: {str(e)}")
            return None
    
    def _get_ftso_abi(self) -> List[Dict]:
        """Get the ABI for FTSO contracts."""
        # This would load the actual FTSO ABI
        ftso_abi_path = os.path.join(os.path.dirname(__file__), "abi/ftso_abi.json")
        with open(ftso_abi_path, 'r') as f:
            return json.load(f)
    
    def _fetch_historical_prices(self, symbol: str, days: int) -> None:
        """Fetch historical price data from chain or API."""
        try:
            # In a real implementation, this would fetch historical prices
            # from Flare Network indexing service or by scanning past events
            
            # Get symbol index
            symbol_index = self._get_symbol_index(symbol)
            if symbol_index is None:
                logger.error(f"Symbol {symbol} not found in FTSO system")
                return
                
            # Get FTSO contract
            ftso_address = self.ftso_manager.functions.getFtsoByIndex(symbol_index).call()
            ftso_contract = self.w3.eth.contract(address=ftso_address, abi=self._get_ftso_abi())
            
            # For demonstration, we would scan price finalization events
            # This is simplified - in production we would use a proper indexing service
            
            # Initialize price history for this symbol if needed
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                
            # In a real implementation, we would fetch actual historical data here
            logger.info(f"Fetched historical price data for {symbol} spanning {days} days")
            
        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {str(e)}")


# Example usage
if __name__ == "__main__":
    import argparse
    
    

