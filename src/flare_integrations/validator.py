"""
Flare Network Validator Integration Module

This module provides integration between Flare Network's validator system and
ev0x's consensus mechanism. It handles validator governance for model weighting,
validator score tracking, and utility methods for consensus verification.
"""

import json
import logging
import time
from typing import Dict, List, Optional, Tuple, Any
from web3 import Web3

from src.consensus.synthesizer import ConsensusSynthesizer
from src.evaluation.metrics import PerformanceMetric, PerformanceTracker
from src.models.model_runner import ModelRunner

# Configure logging
logger = logging.getLogger(__name__)

class FlareValidator:
    """
    Integrates Flare Network validators with ev0x's consensus mechanism.
    
    This class provides methods for:
    1. Using validator governance to adjust model weights
    2. Tracking validator consensus accuracy scores
    3. Providing interface methods for ev0x's consensus system
    """
    
    def __init__(
        self, 
        rpc_endpoint: str, 
        validator_contract_address: str,
        governance_contract_address: str,
        performance_tracker: Optional[PerformanceTracker] = None
    ):
        """
        Initialize FlareValidator with network and contract details.
        
        Args:
            rpc_endpoint: Flare Network RPC endpoint URL
            validator_contract_address: Address of validator contract
            governance_contract_address: Address of governance contract
            performance_tracker: Optional tracker for measuring performance
        """
        self.web3 = Web3(Web3.HTTPProvider(rpc_endpoint))
        self.validator_address = validator_contract_address
        self.governance_address = governance_contract_address
        
        # Load contract ABIs
        self.validator_contract = self._load_contract(
            validator_contract_address, 
            self._get_validator_abi()
        )
        self.governance_contract = self._load_contract(
            governance_contract_address, 
            self._get_governance_abi()
        )
        
        # Initialize validator data structures
        self.validator_scores: Dict[str, float] = {}
        self.model_weights: Dict[str, float] = {}
        self.last_governance_update = 0
        
        # Link to performance tracker for ev0x integration
        self.performance_tracker = performance_tracker or PerformanceTracker()
        
    def _load_contract(self, address: str, abi: List[Dict[str, Any]]) -> Any:
        """Load a contract with given address and ABI."""
        return self.web3.eth.contract(address=address, abi=abi)
    
    def _get_validator_abi(self) -> List[Dict[str, Any]]:
        """Return the validator contract ABI."""
        # This would typically load from a file or API
        return [
            {
                "inputs": [],
                "name": "getActiveValidators",
                "outputs": [{"type": "address[]", "name": "validators"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"type": "address", "name": "validator"}],
                "name": "getValidatorScore",
                "outputs": [{"type": "uint256", "name": "score"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
    
    def _get_governance_abi(self) -> List[Dict[str, Any]]:
        """Return the governance contract ABI."""
        return [
            {
                "inputs": [{"type": "string", "name": "modelId"}],
                "name": "getModelWeight",
                "outputs": [{"type": "uint256", "name": "weight"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getActiveProposals",
                "outputs": [{"type": "uint256[]", "name": "proposalIds"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
    def update_model_weights_from_governance(self) -> Dict[str, float]:
        """
        Update model weights based on Flare Network governance decisions.
        
        Fetches the latest model weight proposals from the governance contract
        and updates the internal model weights accordingly.
        
        Returns:
            Dict mapping model IDs to their updated weights
        """
        # Check if we need to update (avoid too frequent updates)
        current_time = time.time()
        if current_time - self.last_governance_update < 300:  # 5 minutes
            logger.debug("Skipping governance update, last update was recent")
            return self.model_weights
            
        try:
            # Get active proposals that affect model weights
            proposal_ids = self.governance_contract.functions.getActiveProposals().call()
            
            # Process each approved proposal that impacts model weights
            updated_weights = {}
            for model_id in self._get_supported_models():
                try:
                    # Get weight from governance contract (normalize to 0-1 range)
                    raw_weight = self.governance_contract.functions.getModelWeight(model_id).call()
                    normalized_weight = float(raw_weight) / 10000.0  # Assuming weights are stored as basis points (0-10000)
                    
                    updated_weights[model_id] = normalized_weight
                    logger.info(f"Updated weight for model {model_id}: {normalized_weight}")
                except Exception as e:
                    logger.error(f"Error getting weight for model {model_id}: {e}")
                    # Keep existing weight if available, otherwise default to 1.0
                    updated_weights[model_id] = self.model_weights.get(model_id, 1.0)
            
            # Update internal state
            self.model_weights = updated_weights
            self.last_governance_update = current_time
            
            return self.model_weights
            
        except Exception as e:
            logger.error(f"Failed to update model weights from governance: {e}")
            return self.model_weights
    
    def track_validator_consensus_accuracy(
        self, 
        validator_address: str, 
        consensus_result: str, 
        ground_truth: str
    ) -> float:
        """
        Track validator consensus accuracy for a given validation event.
        
        Args:
            validator_address: The address of the validator
            consensus_result: The consensus result the validator validated
            ground_truth: The actual ground truth (for scoring)
            
        Returns:
            The updated validator score
        """
        # Calculate accuracy score using performance tracker
        accuracy = self.performance_tracker.calculate_accuracy(
            consensus_result, 
            ground_truth
        )
        
        # Update validator's score (exponential moving average)
        current_score = self.validator_scores.get(validator_address, 0.5)  # Default to 0.5
        alpha = 0.2  # Weight for new observations
        updated_score = (alpha * accuracy) + ((1 - alpha) * current_score)
        
        # Store updated score
        self.validator_scores[validator_address] = updated_score
        
        # Log the update
        logger.info(f"Updated validator {validator_address} score: {updated_score}")
        
        return updated_score
    
    def get_validator_weighted_consensus(
        self, 
        consensus_candidates: Dict[str, str]
    ) -> Tuple[str, float]:
        """
        Generate consensus result weighted by validator scores.
        
        Args:
            consensus_candidates: Dict mapping validator addresses to their
                                 proposed consensus results
                                 
        Returns:
            Tuple of (winning consensus result, confidence score)
        """
        if not consensus_candidates:
            return "", 0.0
            
        # Count votes weighted by validator scores
        weighted_votes: Dict[str, float] = {}
        
        for validator, result in consensus_candidates.items():
            validator_score = self.validator_scores.get(validator, 0.5)
            weighted_votes[result] = weighted_votes.get(result, 0.0) + validator_score
            
        # Find result with highest weighted vote
        best_result = ""
        best_score = 0.0
        
        for result, score in weighted_votes.items():
            if score > best_score:
                best_result = result
                best_score = score
                
        # Calculate confidence as the proportion of total weighted votes
        total_score = sum(weighted_votes.values())
        confidence = best_score / total_score if total_score > 0 else 0.0
        
        return best_result, confidence
    
    def apply_validator_weights_to_consensus(
        self, 
        consensus_synthesizer: ConsensusSynthesizer
    ) -> None:
        """
        Apply validator-derived weights to the consensus synthesizer.
        
        Args:
            consensus_synthesizer: ev0x consensus synthesizer instance
        """
        # First update weights from governance
        weights = self.update_model_weights_from_governance()
        
        # Apply weights to consensus synthesizer
        consensus_synthesizer.update_weights(weights)
        
        logger.info(f"Applied validator weights to consensus synthesizer: {weights}")
    
    def register_consensus_result(
        self, 
        query_id: str, 
        consensus_result: str, 
        contributing_models: Dict[str, float]
    ) -> bool:
        """
        Register a consensus result on-chain for future verification.
        
        Args:
            query_id: Unique identifier for the query
            consensus_result: The final consensus result
            contributing_models: Dict mapping model IDs to their contribution scores
            
        Returns:
            Success status of the registration
        """
        try:
            # This would typically involve a transaction to register the result
            # Here we just log it for demonstration
            logger.info(f"Registered consensus result for query {query_id}")
            logger.info(f"Models contributions: {contributing_models}")
            
            # In a full implementation, this would submit a transaction to the
            # validator contract to register the result
            
            return True
        except Exception as e:
            logger.error(f"Failed to register consensus result: {e}")
            return False
    
    def _get_supported_models(self) -> List[str]:
        """Get list of model IDs supported in the governance system."""
        # This would typically be fetched from the contract
        # For demonstration, we'll return a static list
        return [
            "gpt-4-turbo",
            "claude-3-opus",
            "claude-3-sonnet",
            "llama-3-70b",
            "mixtral-8x7b"
        ]

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import requests
from web3 import Web3

from src.consensus.synthesizer import ConsensusSynthesizer
from src.evaluation.metrics import ModelPerformanceTracker
from src.models.model_runner import ModelRunner


class FlareValidator:
    """
    Integrates Flare Network validators with ev0x's LLM consensus mechanism.
    
    This class enables using Flare validators as part of the model consensus
    verification process, with validator rewards tied to consensus accuracy,
    and validator governance determining model weights.
    """
    
    def __init__(
        self,
        flare_rpc_url: str = os.getenv("FLARE_RPC_URL", "https://flare-api.flare.network/ext/C/rpc"),
        validator_contract_address: str = os.getenv("VALIDATOR_CONTRACT", "0x1234567890123456789012345678901234567890"),
        consensus_synthesizer: Optional[ConsensusSynthesizer] = None,
        model_runner: Optional[ModelRunner] = None
    ):
        """
        Initialize FlareValidator with connection to Flare Network.
        
        Args:
            flare_rpc_url: Flare Network RPC endpoint
            validator_contract_address: Address of validator contract
            consensus_synthesizer: ConsensusSynthesizer instance for integration
            model_runner: ModelRunner instance for model interactions
        """
        self.web3 = Web3(Web3.HTTPProvider(flare_rpc_url))
        self.validator_contract_address = validator_contract_address
        
        # Load validator contract ABI (simplified for example)
        self.validator_abi = [
            {"type": "function", "name": "getActiveValidators", "inputs": [], "outputs": [{"type": "address[]"}]},
            {"type": "function", "name": "getValidatorWeight", "inputs": [{"type": "address"}], "outputs": [{"type": "uint256"}]},
            {"type": "function", "name": "distributeRewards", "inputs": [{"type": "address[]"}, {"type": "uint256[]"}], "outputs": []},
            {"type": "function", "name": "recordConsensusVote", "inputs": [{"type": "address"}, {"type": "bytes32"}, {"type": "uint8"}], "outputs": []}
        ]
        
        self.validator_contract = self.web3.eth.contract(
            address=self.web3.to_checksum_address(validator_contract_address),
            abi=self.validator_abi
        )
        
        self.consensus_synthesizer = consensus_synthesizer
        self.model_runner = model_runner
        
        # Track validator performance over time
        self.validator_performance = {}
        
        # Track model performance metrics for reward distribution
        self.performance_tracker = ModelPerformanceTracker()
    
    def get_active_validators(self) -> List[str]:
        """
        Fetch list of active validators from Flare Network.
        
        Returns:
            List of validator addresses
        """
        try:
            validators = self.validator_contract.functions.getActiveValidators().call()
            return validators
        except Exception as e:
            print(f"Error fetching validators: {e}")
            return []
    
    def get_validator_weights(self) -> Dict[str, float]:
        """
        Get weights for all active validators based on stake and reputation.
        
        Returns:
            Dictionary mapping validator addresses to normalized weights
        """
        validators = self.get_active_validators()
        weights = {}
        total_weight = 0
        
        for validator in validators:
            try:
                raw_weight = self.validator_contract.functions.getValidatorWeight(validator).call()
                
                # Apply credibility modifier based on historical performance
                credibility = self.get_validator_credibility(validator)
                adjusted_weight = raw_weight * credibility
                
                weights[validator] = adjusted_weight
                total_weight += adjusted_weight
            except Exception as e:
                print(f"Error getting weight for validator {validator}: {e}")
        
        # Normalize weights
        if total_weight > 0:
            return {v: w / total_weight for v, w in weights.items()}
        return {v: 1.0 / len(validators) for v in validators}
    
    def get_validator_credibility(self, validator_address: str) -> float:
        """
        Calculate validator credibility score based on historical consensus accuracy.
        
        Args:
            validator_address: Validator address to check
            
        Returns:
            Credibility score between 0.5 and 1.5
        """
        if validator_address not in self.validator_performance:
            return 1.0  # Default credibility
        
        # Calculate based on agreement with final consensus
        correct_votes = self.validator_performance[validator_address].get("correct_votes", 0)
        total_votes = self.validator_performance[validator_address].get("total_votes", 1)
        
        # Credibility starts at 1.0 and ranges from 0.5 to 1.5
        base_credibility = 1.0
        if total_votes > 0:
            accuracy = correct_votes / total_votes
            # Scale accuracy to [-0.5, 0.5] range around base credibility
            credibility = base_credibility + (accuracy - 0.5)
            return max(0.5, min(1.5, credibility))
        
        return base_credibility
    
    def consensus_verification(
        self, 
        query_id: str, 
        model_outputs: Dict[str, str],
        ground_truth: Optional[str] = None
    ) -> Tuple[str, Dict[str, float]]:
        """
        Use Flare validators to verify model outputs and determine consensus.
        
        This integrates with ev0x's consensus mechanism by assigning validator
        weights to different model outputs based on validator governance.
        
        Args:
            query_id: Unique identifier for the query
            model_outputs: Dictionary mapping model IDs to their outputs
            ground_truth: Optional ground truth for performance tracking
            
        Returns:
            Tuple of (consensus output, model weights used)
        """
        if not model_outputs:
            raise ValueError("No model outputs provided for consensus verification")
        
        # Get validator weights
        validator_weights = self.get_validator_weights()
        
        # Simulate validator votes on model outputs
        # In a full implementation, this would query the validators on-chain
        votes = self._simulate_validator_votes(query_id, model_outputs)
        
        # Calculate model weights based on validator votes and weights
        model_weights = {}
        
        for model_id in model_outputs:
            model_weights[model_id] = 0
            
            for validator, vote in votes.items():
                if vote == model_id:
                    model_weights[model_id] += validator_weights.get(validator, 0)
        
        # Normalize model weights
        total_weight = sum(model_weights.values())
        if total_weight > 0:
            model_weights = {m: w / total_weight for m, w in model_weights.items()}
        else:
            # Equal weights if no votes received
            model_weights = {m: 1.0 / len(model_outputs) for m in model_outputs}
            
        # Update consensus synthesizer with new model weights
        if self.consensus_synthesizer:
            self.consensus_synthesizer.update_weights(model_weights)
            
        # Generate consensus output using weighted model outputs
        consensus_output = self._generate_weighted_consensus(model_outputs, model_weights)
        
        # If ground truth is provided, update validator performance
        if ground_truth:
            self._update_validator_performance(votes, consensus_output, ground_truth)
            
        # Record votes on-chain
        self._record_consensus_votes(query_id, votes)
        
        return consensus_output, model_weights
    
    def _simulate_validator_votes(self, query_id: str, model_outputs: Dict[str, str]) -> Dict[str, str]:
        """
        Simulate validator votes on model outputs.
        
        In a production implementation, this would query validators on-chain.
        
        Args:
            query_id: Query identifier
            model_outputs: Dictionary of model outputs
            
        Returns:
            Dictionary mapping validator addresses to their voted model ID
        """
        validators = self.get_active_validators()
        votes = {}
        
        # For now, use a deterministic approach based on validator address and query
        for validator in validators:
            # In production, this would be an on-chain vote from the validator
            # For now, simulate based on validator address hash and query hash
            validator_hash = int(validator[-8:], 16)
            query_hash = hash(query_id) % 1000
            combined_hash = (validator_hash + query_hash) % len(model_outputs)
            
            # Select model based on hash
            voted_model = list(model_outputs.keys())[combined_hash]
            votes[validator] = voted_model
            
        return votes
    
    def _generate_weighted_consensus(
        self, 
        model_outputs: Dict[str, str], 
        model_weights: Dict[str, float]
    ) -> str:
        """
        Generate consensus output using weighted model outputs.
        
        Args:
            model_outputs: Dictionary mapping model IDs to their outputs
            model_weights: Dictionary mapping model IDs to their weights
            
        Returns:
            Consensus output text
        """
        if self.consensus_synthesizer:
            # Use the consensus synthesizer if available
            return self.consensus_synthesizer.generate_consensus(
                model_outputs, weights=model_weights
            )
        
        # Simple weighted selection as fallback
        max_weight = 0
        selected_output = ""
        
        for model_id, weight in model_weights.items():
            if weight > max_weight:
                max_weight = weight
                selected_output = model_outputs[model_id]
                
        return selected_output
    
    def _update_validator_performance(
        self, 
        votes: Dict[str, str], 
        consensus_output: str, 
        ground_truth: str
    ) -> None:
        """
        Update validator performance metrics based on voting results.
        
        Args:
            votes: Dictionary mapping validator addresses to voted model ID
            consensus_output: Final consensus output
            ground_truth: Ground truth for accuracy measurement
        """
        # Calculate accuracy of consensus output vs ground truth
        consensus_correct = self.performance_tracker.evaluate_factual_accuracy(
            consensus_output, ground_truth
        ) > 0.8  # Threshold for "correct"
        
        # Update performance for each validator
        for validator, voted_model in votes.items():
            if validator not in self.validator_performance:
                self.validator_performance[validator] = {
                    "correct_votes": 0,
                    "total_votes": 0
                }
                
            self.validator_performance[validator]["total_votes"] += 1
            
            # If consensus matches ground truth, validators who voted for models
            # that contributed to consensus get credit
            if consensus_correct:
                # In a more sophisticated implementation, we would analyze
                # how much each model's output influenced the final consensus
                voted_output = ""
                for model_id, output in self.model_outputs.items():
                    if model_id == voted_model:
                        voted_output = output
                        break
                
                # Simple similarity check
                if self._calculate_similarity(voted_output, consensus_output) > 0.7:
                    self.validator_performance[validator]["correct_votes"] += 1
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.
        
        Args:
            text1: First text string
            text2: Second text string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simple implementation - in production would use more sophisticated
        # semantic similarity metrics
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    def _record_consensus_votes(self, query_id: str, votes: Dict[str, str]) -> None:
        """
        Record consensus votes on-chain for transparency and auditability.
        
        Args:
            query_id: Unique query identifier
            votes: Dictionary mapping validator addresses to voted model IDs
        """
        try:
            for validator, voted_model in votes.items():
                # Create hash of voted model for on-chain storage
                model_hash = Web3.keccak(text=voted_model)
                
                # Record vote on-chain
                # This would send a transaction to the validator contract
                # For now, just print the intended transaction
                print(f"Would record vote: {validator} voted for {voted_model} on query {query_id}")
                
                # In production:
                # tx = self.validator_contract.functions.recordConsensusVote(
                #    validator, Web3.keccak(text=query_id), model_hash
                # ).build_transaction(...)
                # Followed by signing and sending the transaction
        except Exception as e:
            print(f"Error recording consensus votes: {e}")
    
    def distribute_rewards(self, query_id: str, rewards_per_validator: Dict[str, float]) -> None:
        """
        Distribute rewards to validators based on consensus performance.
        
        Args:
            query_id: Unique query identifier
            rewards_per_validator: Dictionary mapping validator addresses to reward amounts
        """
        try:
            # In production, this would actually distribute tokens
            # Here we just print the intended rewards
            validator_list = []
            reward_list = []
            
            for validator, reward in rewards_per_validator.items():
                validator_list.append(Web3.to_checksum_address(validator))
                # Convert to wei units (assuming reward is in FLR)
                reward_wei = int(reward * 10**18)
                reward_list.append(reward_wei)
                
            print(f"Would distribute rewards for query {query_id}:")
            for v, r in zip(validator_list, reward_list):
                print(f"  {v}: {r / 10**18} FLR")
                
            # In production:
            # tx = self.validator_contract.functions.distributeRewards(
            #     validator_list, reward_list
            # ).build_transaction(...)
            # Followed by signing and sending the transaction
        except Exception as e:
            print(f"Error distributing rewards: {e}")
    
    def calculate_rewards(self, query_id: str, consensus_accuracy: float) -> Dict[str, float]:
        """
        Calculate rewards for validators based on consensus accuracy and contribution.
        
        Args:
            query_id: Unique query identifier
            consensus_accuracy: Measured accuracy of the consensus output
            
        Returns:
            Dictionary mapping validator addresses to calculated rewards
        """
        validators = self.get_active_validators()
        rewards = {}
        
        # Base reward per validator
        base_reward = 0.1  # FLR tokens
        
        for validator in validators:
            # Get validator credibility
            credibility = self.get_validator_credibility(validator)
            
            # Calculate reward based on credibility and consensus accuracy
            reward = base_reward * credibility * consensus_accuracy
            
            rewards[validator] = reward
            
        return rewards
    
    def update_model_weights_from_governance(self

