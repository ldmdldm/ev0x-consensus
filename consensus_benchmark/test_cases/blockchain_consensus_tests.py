"""
Blockchain Consensus Tests

This module contains test cases focused on blockchain consensus mechanisms,
with special emphasis on Flare Time Series Oracle (FTSO), validator consensus,
cross-chain verification, and smart contract interactions.
"""

from enum import Enum, auto
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, field

from .base_types import TestCase, DifficultyLevel, Category, EvaluationMetric

class BlockchainSubCategory(Enum):
    """Subcategories for blockchain consensus test cases."""
    FTSO = auto()
    VALIDATOR_CONSENSUS = auto()
    CROSS_CHAIN = auto() 
    SMART_CONTRACT = auto()

# FTSO (Flare Time Series Oracle) Test Cases
ftso_test_cases = [
    TestCase(
        id="ftso-price-data-consensus",
        name="FTSO Price Data Consensus",
        category=Category.BLOCKCHAIN,
        subcategory=BlockchainSubCategory.FTSO,
        difficulty=DifficultyLevel.MEDIUM,
        description="""
        Multiple data providers submit price data for XRP/USD. You need to:
        1. Identify and remove outliers
        2. Weight submissions based on provider stake
        3. Calculate the consensus price
        4. Determine rewards for accurate providers
        """,
        input_data={
            "timestamp": "2023-07-15T14:30:00Z",
            "price_submissions": [
                {"provider": "Provider1", "price": 0.4721, "stake": 1500000},
                {"provider": "Provider2", "price": 0.4735, "stake": 2300000},
                {"provider": "Provider3", "price": 0.4712, "stake": 1800000},
                {"provider": "Provider4", "price": 0.5210, "stake": 900000},  # outlier
                {"provider": "Provider5", "price": 0.4728, "stake": 3100000},
                {"provider": "Provider6", "price": 0.4719, "stake": 1200000},
                {"provider": "Provider7", "price": 0.4100, "stake": 750000},  # outlier
                {"provider": "Provider8", "price": 0.4730, "stake": 2800000},
            ],
            "reference_exchanges": [
                {"exchange": "Binance", "price": 0.4725},
                {"exchange": "Coinbase", "price": 0.4732},
                {"exchange": "Kraken", "price": 0.4718}
            ]
        },
        expected_output={
            "consensus_price": 0.4724,  # weighted average after removing outliers
            "valid_providers": ["Provider1", "Provider2", "Provider3", "Provider5", "Provider6", "Provider8"],
            "outliers": ["Provider4", "Provider7"],
            "reward_distribution": {
                "Provider1": 0.13,
                "Provider2": 0.20,
                "Provider3": 0.16,
                "Provider5": 0.27,
                "Provider6": 0.10,
                "Provider8": 0.24
            }
        },
        evaluation_metrics={
            EvaluationMetric.ACCURACY: 0.9,
            EvaluationMetric.CONSISTENCY: 0.8,
            EvaluationMetric.COMPLETENESS: 0.9
        },
        metadata={
            "blockchain": "Flare",
            "oracle_type": "FTSO",
            "asset_pair": "XRP/USD"
        }
    ),
    
    TestCase(
        id="ftso-reward-epoch-calculation",
        name="FTSO Reward Epoch Calculation",
        category=Category.BLOCKCHAIN,
        subcategory=BlockchainSubCategory.FTSO,
        difficulty=DifficultyLevel.HARD,
        description="""
        Calculate the rewards for FTSO data providers across multiple price epochs.
        You need to:
        1. Determine which providers were within the valid price range for each epoch
        2. Calculate the weighted median price for each epoch
        3. Distribute rewards based on stake and accuracy
        4. Handle multiple assets (XRP, LTC, DOGE)
        """,
        input_data={
            "reward_epoch_id": 127,
            "reward_amount": 1000000,  # in FLR
            "price_epochs": [
                {
                    "epoch_id": 1270,
                    "asset": "XRP/USD",
                    "submissions": [
                        {"provider": "Provider1", "price": 0.4721, "stake": 1500000},
                        {"provider": "Provider2", "price": 0.4735, "stake": 2300000},
                        {"provider": "Provider3", "price": 0.4712, "stake": 1800000}
                    ],
                    "weighted_median": 0.4727
                },
                {
                    "epoch_id": 1271,
                    "asset": "LTC/USD",
                    "submissions": [
                        {"provider": "Provider1", "price": 92.15, "stake": 1500000},
                        {"provider": "Provider2", "price": 92.45, "stake": 2300000},
                        {"provider": "Provider3", "price": 92.30, "stake": 1800000}
                    ],
                    "weighted_median": 92.34
                },
                {
                    "epoch_id": 1272,
                    "asset": "DOGE/USD",
                    "submissions": [
                        {"provider": "Provider1", "price": 0.0731, "stake": 1500000},
                        {"provider": "Provider2", "price": 0.0728, "stake": 2300000},
                        {"provider": "Provider3", "price": 0.0730, "stake": 1800000}
                    ],
                    "weighted_median": 0.0729
                }
            ],
            "threshold_percentage": 3.0  # % deviation allowed from weighted median
        },
        expected_output={
            "reward_distributions": {
                "Provider1": {
                    "total_reward": 320000,
                    "per_asset": {
                        "XRP/USD": 110000,
                        "LTC/USD": 105000,
                        "DOGE/USD": 105000
                    }
                },
                "Provider2": {
                    "total_reward": 380000,
                    "per_asset": {
                        "XRP/USD": 125000,
                        "LTC/USD": 130000,
                        "DOGE/USD": 125000
                    }
                },
                "Provider3": {
                    "total_reward": 300000,
                    "per_asset": {
                        "XRP/USD": 95000,
                        "LTC/USD": 105000,
                        "DOGE/USD": 100000
                    }
                }
            },
            "statistics": {
                "valid_submissions_percentage": 100.0,
                "reward_distribution_ratio": {
                    "Provider1": 0.32,
                    "Provider2": 0.38,
                    "Provider3": 0.30
                }
            }
        },
        evaluation_metrics={
            EvaluationMetric.ACCURACY: 0.9,
            EvaluationMetric.COMPLEXITY: 0.8,
            EvaluationMetric.COMPLETENESS: 0.9
        },
        metadata={
            "blockchain": "Flare",
            "oracle_type": "FTSO",
            "reward_epoch": 127
        }
    )
]

# Decentralized Validator Consensus Test Cases
validator_consensus_test_cases = [
    TestCase(
        id="validator-byzantine-fault-tolerance",
        name="Validator Byzantine Fault Tolerance",
        category=Category.BLOCKCHAIN,
        subcategory=BlockchainSubCategory.VALIDATOR_CONSENSUS,
        difficulty=DifficultyLevel.HARD,
        description="""
        In a network of 21 validators, some are reporting incorrect data or acting maliciously.
        You need to:
        1. Identify the validators that are Byzantine (malicious)
        2. Determine if consensus can still be reached
        3. Calculate the honest validator votes needed for valid consensus
        4. Recommend a minimum threshold for Flare's consensus mechanism
        """,
        input_data={
            "total_validators": 21,
            "validators": [
                {"id": "V1", "vote": "0x7a8b...", "stake": 4500000},
                {"id": "V2", "vote": "0x7a8b...", "stake": 3800000},
                {"id": "V3", "vote": "0xdef1...", "stake": 2900000},  # malicious
                {"id": "V4", "vote": "0x7a8b...", "stake": 5200000},
                {"id": "V5", "vote": "0x7a8b...", "stake": 3100000},
                {"id": "V6", "vote": "0xdef1...", "stake": 1800000},  # malicious
                {"id": "V7", "vote": "0x7a8b...", "stake": 4200000},
                {"id": "V8", "vote": "0x7a8b...", "stake": 2800000},
                {"id": "V9", "vote": "0x7a8b...", "stake": 3600000},
                {"id": "V10", "vote": "0xabcd...", "stake": 1500000},  # malicious (unique vote)
                {"id": "V11", "vote": "0x7a8b...", "stake": 4100000},
                {"id": "V12", "vote": "0x7a8b...", "stake": 3900000},
                {"id": "V13", "vote": "0x7a8b...", "stake": 3300000},
                {"id": "V14", "vote": "0xdef1...", "stake": 2200000},  # malicious
                {"id": "V15", "vote": "0x7a8b...", "stake": 4700000},
                {"id": "V16", "vote": "0x7a8b...", "stake": 3500000},
                {"id": "V17", "vote": "0x7a8b...", "stake": 2700000},
                {"id": "V18", "vote": "0x7a8b...", "stake": 4900000},
                {"id": "V19", "vote": "0xdef1...", "stake": 1900000},  # malicious
                {"id": "V20", "vote": "0x7a8b...", "stake": 3700000},
                {"id": "V21", "vote": "0x7a8b...", "stake": 4200000},
            ],
            "consensus_threshold": 0.7,  # percentage of stake needed for consensus
            "expected_result": "0x7a8b..."  # the correct block hash
        },
        expected_output={
            "byzantine_validators": ["V3", "V6", "V10", "V14", "V19"],
            "consensus_possible": True,
            "consensus_vote": "0x7a8b...",
            "honest_validators_percentage": 76.19,  # percentage of validators
            "honest_stake_percentage": 82.14,  # percentage of stake
            "minimum_threshold_recommendation": {
                "validators": 15,  # minimum honest validators needed
                "stake_percentage": 67.0  # minimum stake percentage
            },
            "fault_tolerance_analysis": {
                "maximum_byzantine_validators_tolerable": 7,
                "maximum_byzantine_stake_tolerable": 33.0,
                "system_security_rating": "High"
            }
        },
        evaluation_metrics={
            EvaluationMetric.ACCURACY: 0.9,
            EvaluationMetric.SECURITY: 0.95,
            EvaluationMetric.COMPLETENESS: 0.85
        },
        metadata={
            "blockchain": "Flare",
            "consensus_type": "Federated Byzantine Agreement",
            "validators": 21
        }
    ),
    
    TestCase(
        id="weighted-vote-delegation",
        name="Weighted Vote Delegation Chain",
        category=Category.BLOCKCHAIN,
        subcategory=BlockchainSubCategory.VALIDATOR_CONSENSUS,
        difficulty=DifficultyLevel.MEDIUM,
        description="""
        In Flare's delegation system, token holders can delegate voting power to validators.
        Some validators further delegate to others, creating chains of delegation.
        You need to:
        1. Calculate the final voting power of each validator
        2. Resolve circular delegations
        3. Determine if the network maintains sufficient decentralization
        4. Identify the validators with the most influence
        """,
        input_data={
            "validators": [
                {"id": "V1", "own_stake": 1000000},
                {"id": "V2", "own_stake": 1500000},
                {"id": "V3", "own_stake": 2000000},
                {"id": "V4", "own_stake": 1800000},
                {"id": "V5", "own_stake": 2500000}
            ],
            "delegations": [
                {"from": "User1", "to": "V1", "amount": 500000},
                {"from": "User2", "to": "V2", "amount": 700000},
                {"from": "User3", "to": "V3", "amount": 600000},
                {"from": "User4", "to": "V4", "amount": 400000},
                {"from": "User5", "to": "V5", "amount": 900000},
                {"from": "V1", "to": "V3", "amount": 300000},  # Validator delegating to another
                {"from": "V2", "to": "V5", "amount": 400000},
                {"from": "V3", "to": "V4", "amount": 500000},
                {"from": "V4", "to": "V2", "amount": 200000},  # Creates a cycle V4->V2->V5
                {"from": "V5", "to": "V1", "amount": 600000}   # Creates a cycle V5->V1->V3->V4->V2->V5
            ],
            "proposals": [
                {
                    "id": "P1",
                    "votes": {
                        "V1": "Yes",
                        "V2": "No", 
                        "V3": "Yes",
                        "V4": "Yes",
                        "V5": "No"
                    }
                }
            ]
        },
        expected_output={
            "final_voting_power": {
                "V1": 1200000,  # Own stake + User1 - delegation to V3 + delegation from V5
                "V2": 2000000,  # Own stake + User2 - delegation to V5 + delegation from V4
                "V3": 2700000,  # Own stake + User3 + delegation from V1
                "V4": 2000000,  # Own stake + User4 + delegation from V3 - delegation to V2
                "V5": 3400000   # Own stake + User5 + delegation from V2 - delegation to V1
            },
            "circular_delegations_detected": True,
            "circular_delegation_paths":

