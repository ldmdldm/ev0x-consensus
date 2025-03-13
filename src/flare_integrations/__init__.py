"""
Flare Network Integration Package for ev0x

This package provides integration components for connecting ev0x's Evolutionary Model Consensus (EMC)
with Flare Network's decentralized infrastructure, enhancing model verification, consensus
accuracy, and cross-chain capabilities.

Components:
- FTSO Integration: Price feed data for model evaluation and economic incentives
- StateConnector: Cross-chain verification of model performance and data sources
- DEX Integration: Liquidity and economic mechanisms for consensus rewards
- Validator Integration: Decentralized governance for model weighting and consensus 

Each component is designed to enhance ev0x's consensus mechanism through decentralized
verification, economic incentives, and cross-chain capabilities.
"""

from .ftso_integration import FTSOPriceFeed, PriceReliabilityTracker
from .state_connector import StateConnector, CrossChainVerifier
from .dex_integration import DEXIntegration, LiquidityProvider
from .validator import FlareValidator, ModelWeightGovernance

__all__ = [
    'FTSOPriceFeed', 'PriceReliabilityTracker',
    'StateConnector', 'CrossChainVerifier',
    'DEXIntegration', 'LiquidityProvider',
    'FlareValidator', 'ModelWeightGovernance',
    'FlareIntegrator'
]


class FlareIntegrator:
    """
    Main integration class that combines all Flare Network components to enhance
    ev0x's Evolutionary Model Consensus (EMC) mechanism.
    
    This class provides a unified interface for interacting with Flare Network's
    decentralized infrastructure, including FTSO price feeds, StateConnector
    cross-chain verification, DEX integration, and validator governance.
    
    Each component enhances ev0x's consensus mechanism in specific ways:
    - FTSO: Provides economic signals for model performance evaluation
    - StateConnector: Enables cross-chain verification of model outputs
    - DEX: Creates economic incentives for model accuracy through token mechanisms
    - Validators: Enables decentralized governance of model weights and parameters
    """
    
    def __init__(self, network_config=None):
        """
        Initialize the FlareIntegrator with configuration for Flare Network.
        
        Args:
            network_config (dict, optional): Configuration for Flare Network
                connections, including RPC endpoints, contract addresses, and
                API keys.
        """
        self.config = network_config or {}
        
        # Initialize individual components
        self.ftso = FTSOPriceFeed(self.config.get('ftso', {}))
        self.state_connector = StateConnector(self.config.get('state_connector', {}))
        self.dex = DEXIntegration(self.config.get('dex', {}))
        self.validator = FlareValidator(self.config.get('validator', {}))
        
    def enhance_consensus(self, consensus_system):
        """
        Integrate Flare Network components with ev0x's consensus system.
        
        Args:
            consensus_system: The ev0x consensus system to enhance
            
        Returns:
            Enhanced consensus system with Flare Network integration
        """
        # Register price feeds for economic signaling
        consensus_system.register_economic_signals(self.ftso)
        
        # Add cross-chain verification for model outputs
        consensus_system.add_verification_layer(self.state_connector)
        
        # Integrate token economics for reward distribution
        consensus_system.register_reward_mechanism(self.dex)
        
        # Add decentralized governance for model weights
        consensus_system.register_weight_governance(self.validator)
        
        return consensus_system
    
    def get_model_weights_from_validators(self):
        """
        Fetch model weights determined by validator governance.
        
        Returns:
            dict: Mapping of model IDs to their weights based on validator consensus
        """
        return self.validator.get_model_weights()
    
    def verify_model_output_crosschain(self, model_id, output_data, source_chain):
        """
        Verify a model's output using cross-chain attestation.
        
        Args:
            model_id (str): ID of the model
            output_data (dict): Output data to verify
            source_chain (str): Source blockchain for verification
            
        Returns:
            dict: Verification results including confidence score
        """
        return self.state_connector.verify_data(model_id, output_data, source_chain)
    
    def calculate_economic_incentives(self, model_performance_data):
        """
        Calculate economic incentives for models based on performance.
        
        Args:
            model_performance_data (dict): Performance metrics for each model
            
        Returns:
            dict: Economic incentives/rewards for each model
        """
        price_data = self.ftso.get_current_prices(['FLR', 'SGB'])
        return self.dex.calculate_rewards(model_performance_data, price_data)
    
    def register_consensus_result_onchain(self, consensus_result, confidence_score):
        """
        Register a consensus result on the Flare Network for verification.
        
        Args:
            consensus_result (dict): The result of the consensus process
            confidence_score (float): Confidence score of the consensus
            
        Returns:
            str: Transaction hash of the registration
        """
        # First verify through state connector
        verification = self.state_connector.verify_consensus_data(consensus_result)
        
        # Then register with validator governance
        tx_hash = self.validator.register_consensus_result(
            consensus_result, 
            confidence_score,
            verification['verification_id']
        )
        
        return tx_hash
    
    def get_integration_status(self):
        """
        Get the status of all Flare Network integrations.
        
        Returns:
            dict: Status of all integration components
        """
        return {
            'ftso': self.ftso.get_status(),
            'state_connector': self.state_connector.get_status(),
            'dex': self.dex.get_status(),
            'validator': self.validator.get_status()
        }

