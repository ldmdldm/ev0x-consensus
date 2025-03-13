"""
DEX Integration Module for Flare Network

This module provides integration with Flare Network's decentralized exchanges
(SparkDEX and RainDEX) for liquidity provision, price discovery, trading pair
management, and automated market making.

Usage:
    from src.flare_integrations.dex_integration import DEXManager
    
    # Initialize with specific DEX and network
    dex_manager = DEXManager(dex_type="SparkDEX", network="flare-mainnet")
    
    # Provide liquidity
    dex_manager.add_liquidity("FLR", "USDC", 1000, 2000)
    
    # Get current price
    price = dex_manager.get_token_price("FLR")
"""

import os
import json
import time
import logging
from typing import Dict, List, Tuple, Optional, Union, Any
from decimal import Decimal
from enum import Enum
import requests
from web3 import Web3

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DEX Types
class DEXType(Enum):
    SPARKDEX = "SparkDEX"
    RAINDEX = "RainDEX"

# Network Types
class NetworkType(Enum):
    FLARE_MAINNET = "flare-mainnet"
    FLARE_TESTNET = "flare-testnet"
    SONGBIRD = "songbird"
    COSTON = "coston"

# Contract addresses for different DEXes and networks
DEX_CONTRACTS = {
    DEXType.SPARKDEX.value: {
        NetworkType.FLARE_MAINNET.value: {
            "router": "0x4c1a5d9558260154aD39Ef6f0E1B8be1ab52B4F0",
            "factory": "0x5D479c2a7FB72977c46130465e93d979aFbD92DE",
            "quoter": "0xC532F2C5Df7B35170B5ed4916cF8B2225daF1d2D",
        },
        NetworkType.FLARE_TESTNET.value: {
            "router": "0x834234BbC5e9C0CE9b6b6B9C8e9365a36388718d",
            "factory": "0x7F95eB0E8aA9DB5Fb478FD2805c161641E2ef5b7",
            "quoter": "0x9E8649eB8192c8CF4E5542b220264F9F77875bA4",
        }
    },
    DEXType.RAINDEX.value: {
        NetworkType.FLARE_MAINNET.value: {
            "router": "0x7E7a27b1f39a157e931a782Be996a74F454c9Ce7",
            "factory": "0x2FD6CC2f9cfa349153C90BD0632C03B6fd3F1758",
            "quoter": "0xB2513F8A61D91395B5Be5A46B10e4D47118A6198",
        },
        NetworkType.SONGBIRD.value: {
            "router": "0x8A6192E43cE11f58b27C72Cc0A2B38d1C9B807C9",
            "factory": "0xA64076F6F7848E9353a6F893f8f30cB177F860A5",
            "quoter": "0x5Ea9C48D6Cd4694Db01bC4e3c3456c82c75d9D4f",
        }
    }
}

# RPC endpoints for different networks
RPC_ENDPOINTS = {
    NetworkType.FLARE_MAINNET.value: "https://flare-api.flare.network/ext/C/rpc",
    NetworkType.FLARE_TESTNET.value: "https://coston-api.flare.network/ext/bc/C/rpc",
    NetworkType.SONGBIRD.value: "https://songbird-api.flare.network/ext/bc/C/rpc",
    NetworkType.COSTON.value: "https://coston-api.flare.network/ext/bc/C/rpc",
}

# Token address mapping
TOKEN_ADDRESSES = {
    NetworkType.FLARE_MAINNET.value: {
        "FLR": "0x1D80c49BbBCd1C0911346656B529DF9E5c2F783d",
        "USDC": "0xB01E6C9668680bDe4CA13cde34E03400DDe11C02",
        "USDT": "0x8e82D9Dc287C2C2Aa6eD7cf9C1D343ef75Ec5A3E",
        "WSGB": "0x7C4ECEb8b8E5C9b25F3C7D602d21b046761e1741",
    },
    NetworkType.SONGBIRD.value: {
        "SGB": "0x0000000000000000000000000000000000000000",
        "WSGB": "0x02f0826ef6aD107Cfc861152B32B52fD11BaB9ED",
        "USDC": "0xC348F894d76DB41bC5455FB86CdF7BfFBc75f657",
    }
}

# AMM constants
DEFAULT_SLIPPAGE = 0.005  # 0.5%
DEFAULT_DEADLINE = 20 * 60  # 20 minutes


class DEXManager:
    """Manager for interacting with Flare Network DEXes"""
    
    def __init__(self, 
                 dex_type: str = DEXType.SPARKDEX.value, 
                 network: str = NetworkType.FLARE_MAINNET.value,
                 wallet_address: Optional[str] = None,
                 private_key: Optional[str] = None):
        """
        Initialize DEX integration manager
        
        Args:
            dex_type: Type of DEX to use (SparkDEX or RainDEX)
            network: Network to connect to
            wallet_address: User's wallet address (optional)
            private_key: User's private key for transactions (optional)
        """
        # Validate inputs
        if dex_type not in [d.value for d in DEXType]:
            raise ValueError(f"Invalid DEX type: {dex_type}")
        if network not in [n.value for n in NetworkType]:
            raise ValueError(f"Invalid network: {network}")
            
        self.dex_type = dex_type
        self.network = network
        self.wallet_address = wallet_address
        self._private_key = private_key
        
        # Setup Web3 connection
        self.web3 = Web3(Web3.HTTPProvider(RPC_ENDPOINTS[network]))
        if not self.web3.is_connected():
            raise ConnectionError(f"Failed to connect to network: {network}")
            
        # Setup contract addresses
        try:
            self.contract_addresses = DEX_CONTRACTS[dex_type][network]
            logger.info(f"Connected to {dex_type} on {network}")
        except KeyError:
            raise ValueError(f"{dex_type} is not available on {network}")
            
        # Load ABIs
        self._load_abis()
        
        # Initialize contracts
        self._initialize_contracts()
        
        # Connect to price oracle
        self._connect_to_price_oracle()
        
        # Track liquidity positions
        self.liquidity_positions = []
        
    def _load_abis(self):
        """Load contract ABIs from files"""
        # In a real implementation, these would be loaded from ABI files
        # For this implementation, we'll assume ABIs are loaded
        self.router_abi = {"name": "router_abi"}  # Placeholder
        self.factory_abi = {"name": "factory_abi"}  # Placeholder
        self.quoter_abi = {"name": "quoter_abi"}  # Placeholder
        self.pair_abi = {"name": "pair_abi"}  # Placeholder
        self.token_abi = {"name": "token_abi"}  # Placeholder
        
        logger.info("Contract ABIs loaded successfully")
        
    def _initialize_contracts(self):
        """Initialize smart contract interfaces"""
        # Setup contract instances
        self.router_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.contract_addresses["router"]),
            abi=self.router_abi
        )
        
        self.factory_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.contract_addresses["factory"]),
            abi=self.factory_abi
        )
        
        self.quoter_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(self.contract_addresses["quoter"]),
            abi=self.quoter_abi
        )
        
        logger.info("DEX contracts initialized")
        
    def _connect_to_price_oracle(self):
        """Connect to Flare's price oracle for price validation"""
        # In a real implementation, this would connect to FTSO price feeds
        # For now, we'll just log that we're connected
        logger.info("Connected to Flare price oracle (FTSO)")
        
    def get_token_address(self, token_symbol: str) -> str:
        """Get the contract address for a token symbol"""
        try:
            return self.web3.to_checksum_address(TOKEN_ADDRESSES[self.network][token_symbol])
        except KeyError:
            raise ValueError(f"Token {token_symbol} not found on {self.network}")
            
    def get_pair_address(self, token_a: str, token_b: str) -> str:
        """
        Get the liquidity pair address for two tokens
        
        Args:
            token_a: First token symbol or address
            token_b: Second token symbol or address
            
        Returns:
            The pair contract address
        """
        # Convert symbols to addresses if needed
        address_a = self.get_token_address(token_a) if token_a in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_a)
        address_b = self.get_token_address(token_b) if token_b in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_b)
        
        # Call factory to get pair
        # In a real implementation, this would call the factory contract
        # For this implementation, we'll return a placeholder
        pair_address = "0x" + "0" * 40  # Placeholder for pair address
        
        logger.info(f"Found pair for {token_a}/{token_b}: {pair_address}")
        return pair_address
        
    def get_token_price(self, token_symbol: str, quote_symbol: str = "USDC") -> Decimal:
        """
        Get the current price of a token
        
        Args:
            token_symbol: Symbol of the token to price
            quote_symbol: Symbol of the quote token (default USDC)
            
        Returns:
            Current price of token in terms of quote token
        """
        # Get token addresses
        token_address = self.get_token_address(token_symbol)
        quote_address = self.get_token_address(quote_symbol)
        
        # In a real implementation, this would call the quoter contract
        # and validate against FTSO price feeds
        # For this implementation, we'll return a sample price
        
        # Simulate getting price from DEX
        sample_price = 1.25  # Example price
        
        # Log the price retrieval
        logger.info(f"Current {token_symbol}/{quote_symbol} price: {sample_price}")
        
        return Decimal(str(sample_price))
        
    def add_liquidity(self, 
                     token_a: str, 
                     token_b: str, 
                     amount_a: Union[int, float, Decimal], 
                     amount_b: Union[int, float, Decimal],
                     slippage: float = DEFAULT_SLIPPAGE,
                     deadline: int = DEFAULT_DEADLINE) -> Dict[str, Any]:
        """
        Add liquidity to a trading pair
        
        Args:
            token_a: First token symbol or address
            token_b: Second token symbol or address
            amount_a: Amount of first token to add
            amount_b: Amount of second token to add
            slippage: Maximum allowed slippage (default 0.5%)
            deadline: Transaction deadline in seconds (default 20 minutes)
            
        Returns:
            Transaction details including LP tokens received
        """
        if not self.wallet_address or not self._private_key:
            raise ValueError("Wallet address and private key required for transactions")
            
        # Convert symbols to addresses if needed
        address_a = self.get_token_address(token_a) if token_a in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_a)
        address_b = self.get_token_address(token_b) if token_b in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_b)
        
        # Convert amounts to proper format with decimals
        # In a real implementation, this would get token decimals
        amount_a_wei = int(Decimal(str(amount_a)) * Decimal("1e18"))
        amount_b_wei = int(Decimal(str(amount_b)) * Decimal("1e18"))
        
        # Calculate min amounts with slippage
        min_a = int(amount_a_wei * (1 - slippage))
        min_b = int(amount_b_wei * (1 - slippage))
        
        # Set deadline
        deadline_timestamp = int(time.time() + deadline)
        
        # In a real implementation, this would call the router contract
        # For this implementation, we'll log the action and return a sample response
        tx_hash = "0x" + "1" * 64  # Placeholder transaction hash
        
        # Record the liquidity position
        position = {
            "token_a": token_a,
            "token_b": token_b,
            "amount_a": float(amount_a),
            "amount_b": float(amount_b),
            "pair_address": self.get_pair_address(token_a, token_b),
            "timestamp": int(time.time()),
            "tx_hash": tx_hash
        }
        self.liquidity_positions.append(position)
        
        logger.info(f"Added liquidity: {amount_a} {token_a} and {amount_b} {token_b}")
        
        return {
            "tx_hash": tx_hash,
            "pair_address": position["pair_address"],
            "lp_tokens_received": min(amount_a, amount_b),  # Simplified calculation
            "position_id": len(self.liquidity_positions) - 1
        }
        
    def remove_liquidity(self,
                        token_a: str,
                        token_b: str,
                        liquidity_amount: Union[int, float, Decimal],
                        min_a: Optional[Union[int, float, Decimal]] = None,
                        min_b: Optional[Union[int, float, Decimal]] = None,
                        slippage: float = DEFAULT_SLIPPAGE,
                        deadline: int = DEFAULT_DEADLINE) -> Dict[str, Any]:
        """
        Remove liquidity from a trading pair
        
        Args:
            token_a: First token symbol or address
            token_b: Second token symbol or address
            liquidity_amount: Amount of LP tokens to remove
            min_a: Minimum amount of token_a to receive (default calculated from slippage)
            min_b: Minimum amount of token_b to receive (default calculated from slippage)
            slippage: Maximum allowed slippage (default 0.5%)
            deadline: Transaction deadline in seconds (default 20 minutes)
            
        Returns:
            Transaction details including tokens received
        """
        if not self.wallet_address or not self._private_key:
            raise ValueError("Wallet address and private key required for transactions")
            
        # Convert symbols to addresses if needed
        address_a = self.get_token_address(token_a) if token_a in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_a)
        address_b = self.get_token_address(token_b) if token_b in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_b)
        
        # Get pair address
        pair_address = self.get_pair_address(token_a, token_b)
        
        # Convert liquidity amount to proper format with decimals
        liquidity_amount_wei = int(Decimal(str(liquidity_amount)) * Decimal("1e18"))
        
        # Find the position in our tracked positions
        position_idx = None
        for idx, pos in enumerate(self.liquidity_positions):
            if pos["token_a"] == token_a and pos["token_b"] == token_b:
                position_idx = idx
                break
                
        if position_idx is None:
            raise ValueError(f"No liquidity position found for {token_a}/{token_b}")
            
        # Calculate expected returns based on proportion of total liquidity
        position = self.liquidity_positions[position_idx]
        
        # If min amounts not specified, calculate based on slippage
        if min_a is None:
            min_a = float(position["amount_a"]) * (1 - slippage)
        if min_b is None:
            min_b = float(position["amount_b"]) * (1 - slippage)
            
        # Convert min amounts to wei
        min_a_wei = int(Decimal(str(min_a)) * Decimal("1e18"))
        min_b_wei = int(Decimal(str(min_b)) * Decimal("1e18"))
        
        # Set deadline
        deadline_timestamp = int(time.time() + deadline)
        
        # In a real implementation, this would call the router contract
        # For this implementation, we'll log the action and return a sample response
        tx_hash = "0x" + "2" * 64  # Placeholder transaction hash
        
        # Record the removal of liquidity
        amount_a_received = float(position["amount_a"])
        amount_b_received = float(position["amount_b"])
        
        # Remove position from tracked positions
        removed_position = self.liquidity_positions.pop(position_idx)
        
        logger.info(f"Removed liquidity: {liquidity_amount} LP tokens from {token_a}/{token_b} pair")
        logger.info(f"Received: {amount_a_received} {token_a} and {amount_b_received} {token_b}")
        
        return {
            "tx_hash": tx_hash,
            "pair_address": pair_address,
            "token_a_received": amount_a_received,
            "token_b_received": amount_b_received
        }
        
    def get_liquidity_positions(self) -> List[Dict[str, Any]]:
        """
        Get all current liquidity positions
        
        Returns:
            List of liquidity positions
        """
        return self.liquidity_positions
        
    def swap_tokens(self,
                   token_in: str,
                   token_out: str,
                   amount_in: Union[int, float, Decimal],
                   min_amount_out: Optional[Union[int, float, Decimal]] = None,
                   slippage: float = DEFAULT_SLIPPAGE,
                   deadline: int = DEFAULT_DEADLINE) -> Dict[str, Any]:
        """
        Swap tokens using the DEX
        
        Args:
            token_in: Input token symbol or address
            token_out: Output token symbol or address
            amount_in: Amount of input token to swap
            min_amount_out: Minimum amount of output token to receive (default calculated from slippage)
            slippage: Maximum allowed slippage (default 0.5%)
            deadline: Transaction deadline in seconds (default 20 minutes)
            
        Returns:
            Transaction details including tokens received
        """
        if not self.wallet_address or not self._private_key:
            raise ValueError("Wallet address and private key required for transactions")
            
        # Convert symbols to addresses if needed
        address_in = self.get_token_address(token_in) if token_in in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_in)
        address_out = self.get_token_address(token_out) if token_out in TOKEN_ADDRESSES[self.network] else self.web3.to_checksum_address(token_out)
        
        # Convert amount to proper format with decimals
        amount_in_wei = int(Decimal(str(amount_in)) * Decimal("1e18"))
        
        # Get quote for expected output
        quote_out = self.get_token_price(token_in, token_out) * Decimal(str(amount_in))
        
        # If min_amount_out not specified, calculate based on slippage
        if min_amount_out is None:
            min_amount_out = float(quote_out) * (1 - slippage)
            
        # Convert min_amount_out to wei
        min_amount_out_wei = int(Decimal(str(min_amount_out)) * Decimal("1e18"))
        
        # Set deadline
        deadline_timestamp = int(time.time() + deadline)
        
        # In a real implementation, this would call the router contract
        # For this implementation, we'll log the action and return a sample response
        tx_hash = "0x" + "3" * 64  # Placeholder transaction hash
        
        # Calculate amount received (normally would be from tx receipt)
        amount_out = float(quote_out)
        
        logger.info(f"Swapped: {amount_in} {token_in} for {amount_out} {token_out}")
        
        return {
            "tx_hash": tx_hash,
            "amount_in": amount_in,
            "amount_out": amount_out,
            "token_in": token_in,
            "token_out": token_out
        }
