import json
import time
import requests
from web3 import Web3
from eth_account import Account
from typing import Dict, List, Any, Tuple

# Flare Network endpoints and contract addresses
FLARE_RPC = "https://flare-api.flare.network/ext/C/rpc"
SONGBIRD_RPC = "https://songbird-api.flare.network/ext/C/rpc"
COSTON_RPC = "https://coston-api.flare.network/ext/C/rpc"

# FTSO contract addresses
FTSO_MANAGER_ADDRESS = "0x1000000000000000000000000000000000000003"
FTSO_REGISTRY_ADDRESS = "0xc7DADA237BC8356CBF703C8e61D2d983EE18a64d"
PRICE_SUBMITTER_ADDRESS = "0x1000000000000000000000000000000000000003"

# StateConnector addresses
STATE_CONNECTOR_ADDRESS = "0x1000000000000000000000000000000000000001"

# ABI files for contract interaction
FTSO_MANAGER_ABI = [
    {"inputs":[],"name":"getCurrentRewardEpoch","outputs":[{"internalType":"uint256","name":"_currentEpochId","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"uint256","name":"_epoch","type":"uint256"}],"name":"getEpochData","outputs":[{"internalType":"uint256","name":"startTimestamp","type":"uint256"},{"internalType":"uint256","name":"endTimestamp","type":"uint256"}],"stateMutability":"view","type":"function"}
]

FTSO_REGISTRY_ABI = [
    {"inputs":[],"name":"getAllFtsos","outputs":[{"internalType":"address[]","name":"_ftsos","type":"address[]"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"string","name":"_symbol","type":"string"}],"name":"getFtsoBySymbol","outputs":[{"internalType":"address","name":"_activeFtsoAddress","type":"address"}],"stateMutability":"view","type":"function"}
]

FTSO_ABI = [
    {"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"getCurrentPrice","outputs":[{"internalType":"uint256","name":"_price","type":"uint256"},{"internalType":"uint256","name":"_timestamp","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[],"name":"decimals","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"}
]

STATE_CONNECTOR_ABI = [
    {"inputs":[],"name":"getNumberOfFinalizedRequests","outputs":[{"internalType":"uint256","name":"","type":"uint256"}],"stateMutability":"view","type":"function"},
    {"inputs":[{"internalType":"bytes32","name":"_requestId","type":"bytes32"}],"name":"getRequest","outputs":[{"components":[{"internalType":"bytes32","name":"requestId","type":"bytes32"},{"internalType":"uint16","name":"protocolId","type":"uint16"},{"internalType":"bytes","name":"data","type":"bytes"},{"internalType":"uint64","name":"timestamp","type":"uint64"},{"internalType":"uint64","name":"finalizationTimestamp","type":"uint64"},{"internalType":"bytes32","name":"merkleRoot","type":"bytes32"},{"components":[{"internalType":"bytes32","name":"key","type":"bytes32"},{"internalType":"bytes","name":"value","type":"bytes"}],"internalType":"struct StateConnector.MerkleProof[]","name":"status","type":"tuple[]"}],"internalType":"struct StateConnector.Request","name":"_request","type":"tuple"}],"stateMutability":"view","type":"function"}
]

class BlockchainConsensusTests:
    """
    Tests for consensus mechanisms on the Flare Network blockchain.
    Tests include FTSO data verification, cross-chain attestations, and DEX interactions.
    """
    
    def __init__(self):
        # Initialize Web3 providers for different networks
        self.flare_web3 = Web3(Web3.HTTPProvider(FLARE_RPC))
        self.songbird_web3 = Web3(Web3.HTTPProvider(SONGBIRD_RPC))
        self.coston_web3 = Web3(Web3.HTTPProvider(COSTON_RPC))
        
        # Initialize contract instances
        self.ftso_manager = self.flare_web3.eth.contract(
            address=self.flare_web3.to_checksum_address(FTSO_MANAGER_ADDRESS),
            abi=FTSO_MANAGER_ABI
        )
        
        self.ftso_registry = self.flare_web3.eth.contract(
            address=self.flare_web3.to_checksum_address(FTSO_REGISTRY_ADDRESS),
            abi=FTSO_REGISTRY_ABI
        )
        
        self.state_connector = self.flare_web3.eth.contract(
            address=self.flare_web3.to_checksum_address(STATE_CONNECTOR_ADDRESS),
            abi=STATE_CONNECTOR_ABI
        )
        
    def run_flare_network_tests(self) -> Dict[str, Any]:
        """
        Run all Flare Network related tests and return the results.
        
        Returns:
            Dict with test results including FTSO data verification, 
            cross-chain attestations, and consensus verification
        """
        results = {
            "timestamp": int(time.time()),
            "network_status": self._check_network_status(),
            "ftso_tests": self._test_ftso_data(),
            "cross_chain_tests": self._test_cross_chain_attestation(),
            "consensus_verification": self._verify_consensus_data()
        }
        
        return results
    
    def _check_network_status(self) -> Dict[str, Any]:
        """Check the status of the Flare Networks"""
        status = {}
        
        try:
            # Check Flare Network
            flare_block = self.flare_web3.eth.block_number
            status["flare"] = {
                "connected": True,
                "block_number": flare_block,
                "gas_price": self.flare_web3.eth.gas_price
            }
        except Exception as e:
            status["flare"] = {"connected": False, "error": str(e)}
            
        try:
            # Check Songbird Network
            songbird_block = self.songbird_web3.eth.block_number
            status["songbird"] = {
                "connected": True,
                "block_number": songbird_block,
                "gas_price": self.songbird_web3.eth.gas_price
            }
        except Exception as e:
            status["songbird"] = {"connected": False, "error": str(e)}
            
        try:
            # Check Coston Network
            coston_block = self.coston_web3.eth.block_number
            status["coston"] = {
                "connected": True,
                "block_number": coston_block,
                "gas_price": self.coston_web3.eth.gas_price
            }
        except Exception as e:
            status["coston"] = {"connected": False, "error": str(e)}
            
        return status
    
    def _test_ftso_data(self) -> Dict[str, Any]:
        """
        Test FTSO data by retrieving current prices for various assets
        and validating the data against external sources.
        
        Returns:
            Dict with FTSO test results
        """
        symbols = ["FLR", "XRP", "BTC", "ETH", "LTC", "XLM", "DOGE", "ADA"]
        results = {}
        
        # Get current reward epoch
        current_epoch = self.ftso_manager.functions.getCurrentRewardEpoch().call()
        epoch_data = self.ftso_manager.functions.getEpochData(current_epoch).call()
        
        results["current_epoch"] = {
            "epoch_id": current_epoch,
            "start_time": epoch_data[0],
            "end_time": epoch_data[1]
        }
        
        # Get prices from all supported FTSOs
        price_data = {}
        for symbol in symbols:
            try:
                # Get FTSO address for this symbol
                ftso_address = self.ftso_registry.functions.getFtsoBySymbol(symbol).call()
                
                # Create contract instance
                ftso = self.flare_web3.eth.contract(
                    address=self.flare_web3.to_checksum_address(ftso_address),
                    abi=FTSO_ABI
                )
                
                # Get current price and format it correctly
                price_data = ftso.functions.getCurrentPrice().call()
                decimals = ftso.functions.decimals().call()
                
                price_value = price_data[0] / (10 ** decimals)
                timestamp = price_data[1]
                
                # Store result
                price_data[symbol] = {
                    "price": price_value,
                    "timestamp": timestamp,
                    "decimals": decimals
                }
            except Exception as e:
                price_data[symbol] = {"error": str(e)}
        
        results["price_data"] = price_data
        
        # Compare with external sources for verification
        external_data = self._get_external_price_data(symbols)
        
        # Calculate deviation from external sources
        deviations = {}
        for symbol in symbols:
            if symbol in price_data and symbol in external_data:
                try:
                    ftso_price = price_data[symbol]["price"]
                    ext_price = external_data[symbol]["price"]
                    
                    deviation_pct = ((ftso_price - ext_price) / ext_price) * 100
                    
                    deviations[symbol] = {
                        "ftso_price": ftso_price,
                        "external_price": ext_price,
                        "deviation_pct": deviation_pct,
                        "consensus_quality": "High" if abs(deviation_pct) < 1.0 else 
                                           "Medium" if abs(deviation_pct) < 3.0 else "Low"
                    }
                except (KeyError, ZeroDivisionError, TypeError):
                    deviations[symbol] = {"error": "Unable to calculate deviation"}
        
        results["price_deviations"] = deviations
        return results
    
    def _get_external_price_data(self, symbols: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Retrieve price data from external APIs for comparison with FTSO prices.
        
        Args:
            symbols: List of asset symbols to fetch prices for
            
        Returns:
            Dict mapping symbols to their prices from external sources
        """
        # Use CoinGecko API for external price verification
        api_url = "https://api.coingecko.com/api/v3/simple/price"
        symbol_mapping = {
            "FLR": "flare-networks",
            "XRP": "ripple",
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "LTC": "litecoin",
            "XLM": "stellar",
            "DOGE": "dogecoin",
            "ADA": "cardano"
        }
        
        # Prepare the coin IDs for the API
        coin_ids = [symbol_mapping.get(symbol, "") for symbol in symbols if symbol in symbol_mapping]
        ids_param = ",".join(coin_ids)
        
        params = {
            "ids": ids_param,
            "vs_currencies": "usd"
        }
        
        try:
            response = requests.get(api_url, params=params)
            data = response.json()
            
            # Format the response to match our symbol list
            result = {}
            for symbol in symbols:
                if symbol in symbol_mapping:
                    coin_id = symbol_mapping[symbol]
                    if coin_id in data:
                        result[symbol] = {
                            "price": data[coin_id]["usd"],
                            "source": "CoinGecko"
                        }
            
            return result
        except Exception as e:
            print(f"Error fetching external price data: {e}")
            return {}
    
    def _test_cross_chain_attestation(self) -> Dict[str, Any]:
        """
        Test cross-chain attestation using the StateConnector protocol
        
        Returns:
            Dict with attestation test results
        """
        results = {
            "timestamp": int(time.time()),
            "attestations": []
        }
        
        try:
            # Get the number of finalized requests
            num_requests = self.state_connector.functions.getNumberOfFinalizedRequests().call()
            results["total_finalized_requests"] = num_requests
            
            # Get the latest 5 attestations for analysis (if available)
            max_to_fetch = min(5, num_requests)
            
            for i in range(max_to_fetch):
                # In a real implementation, we would have the request IDs
                # For demonstration, we'd need to get these through events or known request IDs
                # This is simplified and would need to be expanded with actual request IDs
                request_id = "0x" + "0" * 64  # Placeholder - would be actual request ID in production
                
                # For demonstration purposes only - in production code we would use actual request IDs
                results["attestations"].append({
                    "request_id": request_id,
                    "status": "This would contain actual attestation data from the StateConnector",
                    "verification": "This would show cross-chain verification results"
                })
        except Exception as e:
            results["error"] = str(e)
        
        return results
    
    def _verify_consensus_data(self) -> Dict[str, Any]:
        """
        Verify consensus data from multiple validators on the network
        
        Returns:
            Dict with consensus verification results
        """
        results = {
            "timestamp": int(time.time()),
            "network": "Flare",
            "consensus_metrics": {}
        }
        
        try:
            # Get basic network metrics
            latest_block = self.flare_web3.eth.get_block('latest')
            results["block_number"] = latest_block.number
            results["block_timestamp"] = latest_block.timestamp
            results["gas_used"] = latest_block.gasUsed
            results["gas_limit"] = latest_block.gasLimit
            results["transaction_count"] = len(latest_block.transactions)
            
            # For a real implementation, we would check multiple validators
            # and compare their blockchain state and consensus votes
            # This would involve either:
            # 1. Running multiple nodes and comparing their state
            # 2. Using validator APIs to get their voting history
            # 3. Checking on-chain governance/validation transactions
            
            # For this example, we'll get the last 5 blocks to analyze consensus
            block_times = []
            for i in range(1, 6):
                block_num = latest_block.number - i
                if block_num >= 0:
                    block = self.flare_web3.eth.get_block(block_num)
                    if i > 

import os
import json
import requests
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timezone

# Flare Network API endpoints and constants
FLARE_MAINNET_RPC = "https://flare-api.flare.network/ext/C/rpc"
COSTON_TESTNET_RPC = "https://coston-api.flare.network/ext/C/rpc"
SONGBIRD_RPC = "https://songbird-api.flare.network/ext/C/rpc"

# FTSO contract addresses on Flare Network
FTSO_MANAGER_ADDRESS = "0x1000000000000000000000000000000000000003"
FLARE_STATE_CONNECTOR = "0x1000000000000000000000000000000000000001"

# DEX contract addresses
SPARKDEX_ROUTER = "0x30D1A14D5e28F91C44AFD475129e4873c2CA0355"
RAINDEX_ROUTER = "0xA13E5C5E297B510546214cF9B9f4C3B455CA5b43"

class FlareNetworkTest:
    """Base class for Flare Network consensus tests using blockchain data."""
    
    def __init__(self, 
                 network_rpc: str = FLARE_MAINNET_RPC, 
                 block_number: Optional[int] = None):
        self.network_rpc = network_rpc
        self.block_number = block_number
        
    def _make_rpc_call(self, method: str, params: List[Any]) -> Dict[str, Any]:
        """Make an RPC call to the Flare Network."""
        payload = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": 1
        }
        
        response = requests.post(self.network_rpc, json=payload)
        return response.json()
    
    def get_block(self, block_number: Optional[int] = None) -> Dict[str, Any]:
        """Get a block from the Flare Network."""
        if block_number is None:
            block_number = self.block_number if self.block_number else "latest"
        
        return self._make_rpc_call("eth_getBlockByNumber", [hex(block_number) if isinstance(block_number, int) else block_number, True])
    
    def get_transaction(self, tx_hash: str) -> Dict[str, Any]:
        """Get a transaction from the Flare Network."""
        return self._make_rpc_call("eth_getTransactionByHash", [tx_hash])
    
    def call_contract(self, to: str, data: str, block_number: Optional[int] = None) -> Dict[str, Any]:
        """Call a contract on the Flare Network."""
        if block_number is None:
            block_number = self.block_number if self.block_number else "latest"
            
        return self._make_rpc_call("eth_call", [{"to": to, "data": data}, hex(block_number) if isinstance(block_number, int) else block_number])


class SmartContractInteractionTests(FlareNetworkTest):
    """Tests focusing on smart contract interactions using network data."""
    
    def test_ftso_price_epoch(self, epoch_id: int) -> Dict[str, Any]:
        """Test FTSO price epoch data using Flare Network data."""
        # Function selector for getPriceEpochData(uint256): 0x8a37185d
        data = f"0x8a37185d{epoch_id:064x}"
        result = self.call_contract(FTSO_MANAGER_ADDRESS, data)
        
        # Parse the returned data to extract price epoch information
        if "result" in result:
            # Process real blockchain data
            return {
                "epoch_id": epoch_id,
                "raw_data": result["result"],
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return {
            "epoch_id": epoch_id,
            "error": result.get("error", {"message": "Unknown error"}),
            "success": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def test_contract_state(self, contract_address: str, storage_slot: str) -> Dict[str, Any]:
        """Test contract state by reading specific storage slots from the blockchain."""
        # Get storage at specific slot
        result = self._make_rpc_call("eth_getStorageAt", [contract_address, storage_slot, "latest"])
        
        return {
            "contract_address": contract_address,
            "storage_slot": storage_slot,
            "value": result.get("result"),
            "success": "result" in result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def test_token_balance(self, token_address: str, wallet_address: str) -> Dict[str, Any]:
        """Test token balance retrieval using contract calls."""
        # Function selector for balanceOf(address): 0x70a08231
        data = f"0x70a08231000000000000000000000000{wallet_address[2:]}"
        result = self.call_contract(token_address, data)
        
        if "result" in result:
            # Convert hex value to integer
            balance = int(result["result"], 16)
            return {
                "token_address": token_address,
                "wallet_address": wallet_address,
                "balance": balance,
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return {
            "token_address": token_address,
            "wallet_address": wallet_address,
            "error": result.get("error", {"message": "Unknown error"}),
            "success": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class StateConnectorTests(FlareNetworkTest):
    """Tests focusing on StateConnector verifications with live chain data."""
    
    def test_attestation_request(self, source_chain_id: int, transaction_hash: str) -> Dict[str, Any]:
        """Test StateConnector attestation request using actual cross-chain data."""
        # Create attestation request data
        request_data = {
            "sourceChainId": source_chain_id,
            "transactionHash": transaction_hash
        }
        
        # Function selector for submitAttestationRequest: 0x23fcf05b
        encoded_request = json.dumps(request_data).encode("utf-8").hex()
        data = f"0x23fcf05b{len(encoded_request):064x}{encoded_request}"
        
        # This would submit a transaction in a real scenario, but we're simulating the call
        result = self.call_contract(FLARE_STATE_CONNECTOR, data)
        
        return {
            "source_chain_id": source_chain_id,
            "transaction_hash": transaction_hash,
            "request_status": "result" in result,
            "raw_result": result.get("result", "0x"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def test_proof_verification(self, attestation_id: str) -> Dict[str, Any]:
        """Test verification of attestation proofs using real StateConnector data."""
        # Function selector for getAttestationProof(bytes32): 0x3d15f701
        data = f"0x3d15f701{attestation_id[2:].rjust(64, '0')}"
        result = self.call_contract(FLARE_STATE_CONNECTOR, data)
        
        return {
            "attestation_id": attestation_id,
            "proof_available": "result" in result and result["result"] != "0x",
            "raw_proof": result.get("result", "0x"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class CrossChainAttestationTests(FlareNetworkTest):
    """Tests focusing on cross-chain attestations with network states."""
    
    def test_bitcoin_transaction_proof(self, btc_tx_hash: str, btc_block_number: int) -> Dict[str, Any]:
        """Test Bitcoin transaction proof attestation using real network data."""
        # Create attestation request for Bitcoin transaction
        request_data = {
            "bitcoinTxHash": btc_tx_hash,
            "bitcoinBlockNumber": btc_block_number
        }
        
        # Function selector for requestBitcoinTransactionProof: 0x7131b500
        encoded_request = json.dumps(request_data).encode("utf-8").hex()
        data = f"0x7131b500{len(encoded_request):064x}{encoded_request}"
        
        result = self.call_contract(FLARE_STATE_CONNECTOR, data)
        
        return {
            "btc_tx_hash": btc_tx_hash,
            "btc_block_number": btc_block_number,
            "request_status": "result" in result,
            "raw_result": result.get("result", "0x"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def test_ethereum_state_proof(self, eth_address: str, eth_block_number: int) -> Dict[str, Any]:
        """Test Ethereum state proof attestation using network data."""
        # Create attestation request for Ethereum state
        request_data = {
            "ethereumAddress": eth_address,
            "ethereumBlockNumber": eth_block_number
        }
        
        # Function selector for requestEthereumStateProof: 0x9b24f886
        encoded_request = json.dumps(request_data).encode("utf-8").hex()
        data = f"0x9b24f886{len(encoded_request):064x}{encoded_request}"
        
        result = self.call_contract(FLARE_STATE_CONNECTOR, data)
        
        return {
            "eth_address": eth_address,
            "eth_block_number": eth_block_number,
            "request_status": "result" in result,
            "raw_result": result.get("result", "0x"),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


class DEXIntegrationTests(FlareNetworkTest):
    """Tests focusing on SparkDEX/RainDEX integration using on-chain transactions."""
    
    def test_liquidity_pools(self, dex_router: str, token_a: str, token_b: str) -> Dict[str, Any]:
        """Test liquidity pool data using real DEX contract interactions."""
        # Function selector for getPool(address,address,uint24): 0x1698ee82
        data = f"0x1698ee82000000000000000000000000{token_a[2:]}000000000000000000000000{token_b[2:]}00000000000000000000000000000000000000000000000000000000000003e8"
        
        result = self.call_contract(dex_router, data)
        
        if "result" in result:
            pool_address = "0x" + result["result"][-40:]
            
            # Get pool reserves
            reserves_data = "0x0902f1ac"  # getReserves()
            reserves_result = self.call_contract(pool_address, reserves_data)
            
            return {
                "dex_router": dex_router,
                "token_a": token_a,
                "token_b": token_b,
                "pool_address": pool_address,
                "reserves_data": reserves_result.get("result", "0x"),
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return {
            "dex_router": dex_router,
            "token_a": token_a,
            "token_b": token_b,
            "error": result.get("error", {"message": "Pool not found"}),
            "success": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def test_price_oracle(self, dex_router: str, token_in: str, token_out: str, amount_in: int) -> Dict[str, Any]:
        """Test price oracle data using real DEX contract interactions."""
        # Function selector for getAmountsOut(uint256,address[]): 0xd06ca61f
        amount_in_hex = f"{amount_in:064x}"
        
        # Encode the array of 2 addresses
        addresses_offset = "0000000000000000000000000000000000000000000000000000000000000040"
        array_length = "0000000000000000000000000000000000000000000000000000000000000002"
        address_1 = f"000000000000000000000000{token_in[2:]}"
        address_2 = f"000000000000000000000000{token_out[2:]}"
        
        data = f"0xd06ca61f{amount_in_hex}{addresses_offset}{array_length}{address_1}{address_2}"
        
        result = self.call_contract(dex_router, data)
        
        if "result" in result:
            # Parse the returned amounts
            amounts_hex = result["result"]
            # This is a simplified parsing - real implementation would need to decode array properly
            amount_out = int(amounts_hex[-64:], 16)
            
            return {
                "dex_router": dex_router,
                "token_in": token_in,
                "token_out": token_out,
                "amount_in": amount_in,
                "amount_out": amount_out,
                "success": True,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        return {
            "dex_router": dex_router,
            "token_in": token_in,
            "token_out": token_out,
            "amount_in": amount_in,
            "error": result.get("error", {"message": "Unknown error"}),
            "success": False,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }


def run_flare_network_tests() -> Dict[str, List[Dict[str, Any]]]:
    """Run a comprehensive suite of Flare Network consensus tests."""
    results = {
        "smart_contract_tests": [],
        "state_connector_tests": [],
        "cross_chain_tests": [],
        "dex_integration_tests": []
    }
    
    # Real token addresses on Flare Network
    FLR_TOKEN = "0x1D80c49BbBCd1C0911346656B529DF9E5c2F783d"
    WFLR_TOKEN = "0x1D80c49BbBCd1C0911346656B529DF9E5c2F783d"
    SGB_TOKEN = "0x02f0826ef6aD107Cfc861152B32B52fD11BaB9ED"
    
    # Real wallet addresses with token balances
    TEST_WALLET = "0x977e0c21aA4AA68eA9d36d14a387cE13896F1e7F"
    
    # Instantiate test classes
    sc_tests =

