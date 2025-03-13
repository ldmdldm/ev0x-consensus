import os
import json
import time
import logging
import requests
from typing import Dict, List, Optional, Tuple, Union, Any

from web3 import Web3
from web3.contract import Contract

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StateConnector:
    """
    Integration with Flare Network's StateConnector protocol for cross-chain attestations
    and verifications. Provides methods for requesting attestations, verifying transactions,
    and monitoring attestation states.
    """
    
    # Standard StateConnector contract addresses
    CONTRACTS = {
        "flare": "0x0217BC70125AD115F8d27D0D483F21A3209d3101",  # Flare mainnet
        "songbird": "0x1000000000000000000000000000000000000001",  # Songbird
        "coston": "0x0c13AdA9D39A8F7Fec1F51A4591262d2b67E8a3E",  # Coston testnet
    }
    
    def __init__(self, network: str = "flare", provider_url: Optional[str] = None, private_key: Optional[str] = None):
        """
        Initialize the StateConnector client.
        
        Args:
            network: Network to connect to (flare, songbird, coston)
            provider_url: RPC provider URL (defaults to environment variable)
            private_key: Account private key for signing transactions
        """
        self.network = network.lower()
        
        # Set provider URL from args or environment
        if provider_url:
            self.provider_url = provider_url
        else:
            env_var = f"{self.network.upper()}_PROVIDER_URL"
            self.provider_url = os.environ.get(env_var)
            if not self.provider_url:
                raise ValueError(f"Provider URL not provided and {env_var} environment variable not set")
        
        # Set up Web3 connection
        self.w3 = Web3(Web3.HTTPProvider(self.provider_url))
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to {self.network} network at {self.provider_url}")
        
        # Load ABI for StateConnector contract
        try:
            with open(os.path.join(os.path.dirname(__file__), "abi/state_connector.json"), "r") as f:
                self.sc_abi = json.load(f)
        except FileNotFoundError:
            logger.warning("StateConnector ABI file not found, falling back to simplified ABI")
            # Simplified ABI with core functions
            self.sc_abi = [
                {"type": "function", "name": "submitAttestationRequest", "inputs": [{"type": "bytes", "name": "requestBytes"}], "outputs": [], "stateMutability": "nonpayable"},
                {"type": "function", "name": "verifyAttestationData", "inputs": [{"type": "bytes", "name": "attestationType"}, {"type": "bytes", "name": "sourceId"}, {"type": "bytes", "name": "messageHash"}], "outputs": [{"type": "bool", "name": ""}], "stateMutability": "view"},
                {"type": "function", "name": "getAttestationState", "inputs": [{"type": "bytes32", "name": "requestHash"}], "outputs": [{"type": "uint256", "name": "state"}], "stateMutability": "view"},
            ]
        
        # Initialize contract
        self.sc_contract_address = Web3.to_checksum_address(self.CONTRACTS.get(self.network))
        self.sc_contract = self.w3.eth.contract(address=self.sc_contract_address, abi=self.sc_abi)
        
        # Set up account for transaction signing if private key provided
        self.account = None
        if private_key:
            self.account = self.w3.eth.account.from_key(private_key)
            logger.info(f"Using account {self.account.address} for transactions")
    
    def request_attestation(self, attestation_type: str, source_id: str, 
                           data: Dict[str, Any]) -> str:
        """
        Submit an attestation request to the StateConnector protocol.
        
        Args:
            attestation_type: Type of attestation (payment, transaction, balance)
            source_id: Source chain identifier
            data: Attestation-specific data
            
        Returns:
            Transaction hash of the attestation request
        """
        if not self.account:
            raise ValueError("Private key required for submitting attestation requests")
            
        # Encode request data
        request_data = {
            "type": attestation_type,
            "sourceId": source_id,
            "data": data
        }
        
        # Convert to bytes
        request_bytes = self.w3.to_bytes(text=json.dumps(request_data))
        
        # Build transaction
        tx = self.sc_contract.functions.submitAttestationRequest(request_bytes).build_transaction({
            'from': self.account.address,
            'nonce': self.w3.eth.get_transaction_count(self.account.address),
            'gas': 3000000,
            'gasPrice': self.w3.eth.gas_price,
        })
        
        # Sign and send transaction
        signed_tx = self.account.sign_transaction(tx)
        tx_hash = self.w3.eth.send_raw_transaction(signed_tx.rawTransaction)
        
        logger.info(f"Attestation request submitted: {tx_hash.hex()}")
        return tx_hash.hex()
    
    def verify_transaction(self, source_chain: str, tx_hash: str, 
                          confirmations: int = 5) -> Dict[str, Any]:
        """
        Verify a transaction from another chain using the StateConnector protocol.
        
        Args:
            source_chain: Source blockchain identifier
            tx_hash: Transaction hash to verify
            confirmations: Required confirmations for verification
            
        Returns:
            Verification result with status and details
        """
        # Create transaction verification request
        attestation_data = {
            "transactionHash": tx_hash,
            "requiredConfirmations": confirmations
        }
        
        # Submit attestation request
        request_tx = self.request_attestation(
            attestation_type="TransactionVerification",
            source_id=source_chain,
            data=attestation_data
        )
        
        # Wait for attestation to be processed
        attestation_result = self._wait_for_attestation(request_tx)
        
        # Process and return verification details
        if attestation_result.get("status") == "confirmed":
            return {
                "verified": True,
                "source_chain": source_chain,
                "transaction_hash": tx_hash,
                "confirmation_block": attestation_result.get("confirmationBlock"),
                "timestamp": attestation_result.get("timestamp"),
                "attestation_id": attestation_result.get("attestationId")
            }
        else:
            return {
                "verified": False,
                "source_chain": source_chain,
                "transaction_hash": tx_hash,
                "error": attestation_result.get("error", "Verification failed")
            }
    
    def verify_payment(self, source_chain: str, tx_hash: str, 
                      recipient: str, amount: str, 
                      token_address: Optional[str] = None) -> Dict[str, Any]:
        """
        Verify a payment transaction from another chain.
        
        Args:
            source_chain: Source blockchain identifier
            tx_hash: Transaction hash to verify
            recipient: Expected recipient address
            amount: Expected payment amount
            token_address: Token contract address for token transfers
            
        Returns:
            Payment verification result with status and details
        """
        # Create payment verification request
        attestation_data = {
            "transactionHash": tx_hash,
            "recipient": recipient,
            "amount": amount
        }
        
        if token_address:
            attestation_data["tokenAddress"] = token_address
        
        # Submit attestation request
        request_tx = self.request_attestation(
            attestation_type="PaymentVerification",
            source_id=source_chain,
            data=attestation_data
        )
        
        # Wait for attestation to be processed
        attestation_result = self._wait_for_attestation(request_tx)
        
        # Process and return verification details
        if attestation_result.get("status") == "confirmed":
            return {
                "verified": True,
                "payment_confirmed": attestation_result.get("paymentConfirmed", False),
                "source_chain": source_chain,
                "transaction_hash": tx_hash,
                "recipient": recipient,
                "amount": amount,
                "token": token_address,
                "timestamp": attestation_result.get("timestamp"),
                "attestation_id": attestation_result.get("attestationId")
            }
        else:
            return {
                "verified": False,
                "source_chain": source_chain,
                "transaction_hash": tx_hash,
                "error": attestation_result.get("error", "Payment verification failed")
            }
    
    def get_attestation_state(self, request_hash: str) -> Dict[str, Any]:
        """
        Get the current state of an attestation request.
        
        Args:
            request_hash: Hash of the attestation request
            
        Returns:
            Current state of the attestation
        """
        # Convert hash to bytes32
        request_hash_bytes = self.w3.to_bytes(hexstr=request_hash)
        
        # Get state from contract
        state = self.sc_contract.functions.getAttestationState(request_hash_bytes).call()
        
        # Map state to human-readable format
        state_map = {
            0: "pending",
            1: "confirmed",
            2: "rejected",
            3: "expired"
        }
        
        return {
            "request_hash": request_hash,
            "state_code": state,
            "state": state_map.get(state, "unknown")
        }
    
    def _wait_for_attestation(self, request_tx: str, 
                             timeout: int = 300, 
                             poll_interval: int = 5) -> Dict[str, Any]:
        """
        Wait for an attestation to be processed.
        
        Args:
            request_tx: Transaction hash of the attestation request
            timeout: Maximum time to wait (seconds)
            poll_interval: Time between status checks (seconds)
            
        Returns:
            Attestation result
        """
        start_time = time.time()
        
        # Wait for transaction to be mined
        logger.info(f"Waiting for attestation request {request_tx} to be mined")
        receipt = None
        while receipt is None and (time.time() - start_time) < timeout:
            try:
                receipt = self.w3.eth.get_transaction_receipt(request_tx)
            except Exception:
                time.sleep(poll_interval)
        
        if not receipt:
            return {
                "status": "failed",
                "error": "Transaction not mined within timeout period"
            }
        
        # Extract request hash from logs
        request_hash = None
        try:
            # Parse event logs to find request hash
            # This is a simplified example - in production, use proper event parsing
            request_hash = receipt.logs[0].topics[1].hex()
        except (IndexError, AttributeError):
            return {
                "status": "failed",
                "error": "Could not extract request hash from transaction"
            }
        
        # Wait for attestation to be processed
        logger.info(f"Request mined, waiting for attestation to be processed. Request hash: {request_hash}")
        
        while (time.time() - start_time) < timeout:
            attestation_state = self.get_attestation_state(request_hash)
            state = attestation_state.get("state")
            
            if state == "confirmed":
                return {
                    "status": "confirmed",
                    "attestationId": request_hash,
                    "confirmationBlock": receipt.blockNumber,
                    "timestamp": self.w3.eth.get_block(receipt.blockNumber).timestamp
                }
            elif state == "rejected":
                return {
                    "status": "rejected", 
                    "error": "Attestation request rejected"
                }
            elif state == "expired":
                return {
                    "status": "expired",
                    "error": "Attestation request expired"
                }
            
            # Wait before next check
            time.sleep(poll_interval)
        
        return {
            "status": "timeout",
            "error": "Attestation not processed within timeout period"
        }
    
    def monitor_attestations(self, attestation_ids: List[str], 
                            callback=None, 
                            interval: int = 10) -> None:
        """
        Monitor multiple attestation requests and optionally execute a callback 
        when their state changes.
        
        Args:
            attestation_ids: List of attestation request hashes to monitor
            callback: Function to call when state changes
            interval: Time between checks (seconds)
        """
        states = {aid: None for aid in attestation_ids}
        running = True
        
        logger.info(f"Starting to monitor {len(attestation_ids)} attestations")
        
        try:
            while running and any(aid for aid, state in states.items() if state not in ["confirmed", "rejected", "expired"]):
                for attestation_id in attestation_ids:
                    # Skip already completed attestations
                    if states[attestation_id] in ["confirmed", "rejected", "expired"]:
                        continue
                    
                    current_state = self.get_attestation_state(attestation_id)
                    current_state_str = current_state.get("state")
                    
                    # Check if state has changed
                    if current_state_str != states[attestation_id]:
                        logger.info(f"Attestation {attestation_id} state changed: {states[attestation_id]} -> {current_state_str}")
                        states[attestation_id] = current_state_str
                        
                        # Call callback if provided
                        if callback:
                            callback(attestation_id, current_state)
                
                # Wait before next check
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        
        logger.info("Attestation monitoring completed")
        
        # Return final states
        return states


# Example usage in a cross-chain verification context
if __name__ == "__main__":
    # Test StateConnector functionality
    def test_state_connector():
        # Load private key from environment variable
        private_key = os.environ.get("FLARE_PRIVATE_KEY")
        if not private_key:
            logger.error("FLARE_PRIVATE_KEY environment variable not set")
            return
            
        # Initialize StateConnector client
        connector = StateConnector(
            network="flare",
            provider_url=os.environ.get("FLARE_PROVIDER_URL"),
            private_key=private_key
        )
        
        # Example: Verify an Ethereum transaction
        tx_result = connector.verify_transaction(
            source_chain="ethereum",
            tx_hash="0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef",
            confirmations=12
        )
        
        logger.info(

