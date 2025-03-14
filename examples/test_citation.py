#!/usr/bin/env python3
"""
Test script for citation verification using arXiv and PubMed integrations.
This script demonstrates the ability to verify both scientific and medical claims
with proper citations from academic sources.
"""

import sys
import time
from typing import List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("citation_test")

# Add the src directory to the path
sys.path.append(".")

# Import the citation verification module
from src.factual.citation import Citation, CitationVerifier, SourceType, VerificationStatus

def run_test_case(verifier: CitationVerifier, claim: str, expected_source_type: SourceType = None) -> Dict[str, Any]:
    """Run a test case for citation verification and return the results."""
    start_time = time.time()
    logger.info(f"Testing claim: {claim}")
    
    # Verify the claim
    is_verified, citations = verifier.verify_claim(claim, expected_source_type)
    
    # Calculate verification time
    verification_time = time.time() - start_time
    
    # Get the best citation if available
    citation = citations[0] if citations else None
    
    # Prepare result data
    result = {
        "claim": claim,
        "verified": is_verified,
        "source_type": citation.source_type.name if citation and citation.source_type else "None",
        "confidence": citation.relevance_score if citation else 0.0,
        "verification_time": verification_time,
        "citation": citation
    }
    
    # Display results
    display_result(result, verifier)
    
    return result

def display_result(result: Dict[str, Any], verifier: CitationVerifier) -> None:
    """Display the result of a verification test case."""
    # Validate input parameters
    if not result:
        print("Error: No result data provided")
        return
        
    if not verifier:
        print("Error: Verifier is required for citation formatting")
        return
        
    citation = result.get("citation")
    
    # Print the result header
    print("\n" + "="*80)
    print(f"CLAIM: {result.get('claim', 'No claim provided')}")
    print("="*80)
    
    # Print verification status
    status_str = "✅ VERIFIED" if result.get("verified", False) else "❌ UNVERIFIED"
    print(f"Status: {status_str}")
    print(f"Source Type: {result.get('source_type', 'Unknown')}")
    print(f"Confidence: {result.get('confidence', 0.0):.2f}")
    print(f"Verification Time: {result.get('verification_time', 0.0):.2f} seconds")
    
    # Print citation details if verified and citation exists
    if result.get("verified", False) and citation:
        try:
            print("\nCITATION DETAILS:")
            print(f"  Title: {getattr(citation, 'title', 'Unknown')}")
            
            # Handle authors safely
            if hasattr(citation, 'authors') and citation.authors:
                try:
                    authors_str = ', '.join(citation.authors)
                    print(f"  Authors: {authors_str}")
                except Exception as e:
                    print(f"  Authors: Error displaying authors - {str(e)}")
            else:
                print("  Authors: None specified")
            
            # Handle publication date safely
            if hasattr(citation, 'publication_date') and citation.publication_date:
                print(f"  Publication Date: {citation.publication_date}")
            else:
                print("  Publication Date: Unknown")
                
            # Handle DOI and URL safely
            if hasattr(citation, 'doi') and citation.doi:
                print(f"  DOI: {citation.doi}")
            if hasattr(citation, 'url') and citation.url:
                print(f"  URL: {citation.url}")
            
            # Print formatted citation
            print("\nAPA Citation:")
            try:
                apa_format = verifier.get_citation_format(citation, 'apa')
                print(f"  {apa_format}")
            except Exception as e:
                print(f"  Error formatting citation: {str(e)}")
                logger.error(f"Citation formatting error: {e}", exc_info=True)
            
            # Print relevance information
            print("\nRELEVANCE DETAILS:")
            if hasattr(citation, 'relevance_score'):
                print(f"  Relevance Score: {citation.relevance_score:.2f}")
            else:
                print("  Relevance Score: Not available")
                
            if hasattr(citation, 'abstract') and citation.abstract:
                try:
                    abstract_preview = citation.abstract[:200] + "..." if len(citation.abstract) > 200 else citation.abstract
                    print(f"  Abstract: {abstract_preview}")
                except Exception as e:
                    print(f"  Abstract: Error displaying abstract - {str(e)}")
        except Exception as e:
            print(f"\nError displaying citation details: {str(e)}")
            logger.error(f"Error displaying citation details: {e}", exc_info=True)
    elif result.get("verified", False) and not citation:
        print("\nWARNING: Claim marked as verified but no citation information available")
        logger.warning("Claim marked as verified but no citation provided")
    
    print("-"*80 + "\n")

def main():
    """Main test function to verify claims using both arXiv and PubMed."""
    print("\n" + "="*40 + " CITATION VERIFICATION TEST " + "="*40)
    
    # Initialize the citation verifier
    verifier = CitationVerifier()
    
    # Define scientific claims for arXiv testing
    scientific_claims = [
        "Transformer models have revolutionized natural language processing by using self-attention mechanisms.",
        "Quantum computing uses qubits which can exist in superposition states, allowing for potential computational advantages.",
        "Deep learning models can now generate realistic images using generative adversarial networks.",
        "Graph neural networks have been effective for social network analysis.",
        "The BERT model introduced bidirectional training for language representation."
    ]
    
    # Define medical claims for PubMed testing
    medical_claims = [
        "Statins are effective in reducing LDL cholesterol levels and cardiovascular risk.",
        "CRISPR-Cas9 has been used for gene editing in various genetic disorders.",
        "Metformin is commonly prescribed as a first-line treatment for type 2 diabetes.",
        "Vaccines work by stimulating the immune system to recognize and fight specific pathogens.",
        "Antibiotics are not effective against viral infections such as the common cold or flu."
    ]
    
    # Test scientific claims with arXiv
    print("\n" + "="*30 + " TESTING SCIENTIFIC CLAIMS (arXiv) " + "="*30)
    scientific_results = []
    for claim in scientific_claims:
        result = run_test_case(verifier, claim, SourceType.ARXIV)
        scientific_results.append(result)
    
    # Test medical claims with PubMed
    print("\n" + "="*30 + " TESTING MEDICAL CLAIMS (PubMed) " + "="*30)
    medical_results = []
    for claim in medical_claims:
        result = run_test_case(verifier, claim, SourceType.PUBMED)
        medical_results.append(result)
    
    # Test a mixed paragraph with both types of claims
    print("\n" + "="*30 + " TESTING MIXED PARAGRAPH " + "="*30)
    mixed_paragraph = """
    Recent advances in AI research have shown that Transformer models with self-attention mechanisms 
    are highly effective for natural language tasks. In the medical field, studies have demonstrated 
    that statins significantly reduce cardiovascular risk by lowering LDL cholesterol. Meanwhile, 
    CRISPR-Cas9 technology has revolutionized genetic engineering, while GANs have enabled the 
    generation of photorealistic images in computer vision research.
    """
    
    print(f"Paragraph: {mixed_paragraph}")
    print("\nExtracting and verifying claims...")
    
    # Extract claims from paragraph
    extracted_claims = verifier.extract_claims(mixed_paragraph)
    mixed_results = []
    
    for claim in extracted_claims:
        # Let the verifier determine the appropriate source type
        result = run_test_case(verifier, claim)
        mixed_results.append(result)
    
    # Display summary statistics
    print("\n" + "="*40 + " TEST SUMMARY " + "="*40)
    
    # Calculate statistics for scientific claims
    scientific_verified = sum(1 for r in scientific_results if r["verified"])
    scientific_percentage = (scientific_verified / len(scientific_claims)) * 100
    scientific_avg_time = sum(r["verification_time"] for r in scientific_results) / len(scientific_results)
    scientific_avg_confidence = sum(r["confidence"] for r in scientific_results if r["verified"]) / max(1, scientific_verified)
    
    # Calculate statistics for medical claims
    medical_verified = sum(1 for r in medical_results if r["verified"])
    medical_percentage = (medical_verified / len(medical_claims)) * 100
    medical_avg_time = sum(r["verification_time"] for r in medical_results) / len(medical_results)
    medical_avg_confidence = sum(r["confidence"] for r in medical_results if r["verified"]) / max(1, medical_verified)
    
    # Display statistics
    print(f"Scientific Claims (arXiv):")
    print(f"  Total: {len(scientific_claims)}")
    print(f"  Verified: {scientific_verified} ({scientific_percentage:.1f}%)")
    print(f"  Average Verification Time: {scientific_avg_time:.2f} seconds")
    print(f"  Average Confidence: {scientific_avg_confidence:.2f}")
    
    print(f"\nMedical Claims (PubMed):")
    print(f"  Total: {len(medical_claims)}")
    print(f"  Verified: {medical_verified} ({medical_percentage:.1f}%)")
    print(f"  Average Verification Time: {medical_avg_time:.2f} seconds")
    print(f"  Average Confidence: {medical_avg_confidence:.2f}")
    
    # Mixed paragraph statistics
    mixed_verified = sum(1 for r in mixed_results if r["verified"])
    mixed_percentage = (mixed_verified / len(mixed_results)) * 100 if mixed_results else 0
    
    print(f"\nMixed Paragraph:")
    print(f"  Claims Extracted: {len(mixed_results)}")
    print(f"  Claims Verified: {mixed_verified} ({mixed_percentage:.1f}%)")
    
    print("\n" + "="*80)
    print("TEST COMPLETED")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()

