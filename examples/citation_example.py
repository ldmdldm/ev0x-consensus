#!/usr/bin/env python3
"""
Example script demonstrating real citation verification with scientific and medical claims.

This script shows how to verify scientific claims using real sources from arXiv
and medical claims using PubMed, providing detailed citation information for both.
"""

import asyncio
import sys
import os
import json
from datetime import datetime

# Add the project root to Python path to make imports work correctly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.factual.citation import CitationVerifier, SourceType, Citation


async def display_citation_details(citation):
    """
    Display detailed information about a citation.
    
    Args:
        citation: Citation object to display
    """
    print(f"\nCitation Details for: '{citation.claim}'")
    print("-" * 70)
    print(f"Verified: {citation.verified}")
    print(f"Confidence: {citation.confidence:.2f}")
    print(f"Source: {citation.source_name}")
    
    if citation.source_url:
        print(f"URL: {citation.source_url}")
    
    if citation.authors:
        print(f"Authors: {', '.join(citation.authors[:3])}", end="")
        if len(citation.authors) > 3:
            print(" et al.")
        else:
            print()
    
    if citation.publication_date:
        print(f"Published: {citation.publication_date}")
    
    if citation.doi:
        print(f"DOI: {citation.doi}")
    
    if citation.pmid:
        print(f"PMID: {citation.pmid}")
        
    print(f"Source Type: {citation.source_type.name if citation.source_type else 'Unknown'}")
    print(f"Verification Method: {citation.verification_method}")


async def test_scientific_claim(claim, verifier=None):
    """
    Test real citation verification with a scientific claim using arXiv.
    
    Args:
        claim: The scientific claim to verify
        verifier: Optional CitationVerifier instance
    """
    if not verifier:
        verifier = CitationVerifier()
    
    print(f"\nVerifying scientific claim: '{claim}'")
    print("-" * 70)
    start_time = datetime.now()
    
    # Verify the claim
    citation = await verifier.verify_claim(claim, domain="scientific", source_type=SourceType.ARXIV)
    
    # Calculate verification time
    verification_time = (datetime.now() - start_time).total_seconds()
    
    # Display verification results
    print(f"Verification completed in {verification_time:.2f} seconds")
    await display_citation_details(citation)
    
    return citation


async def test_medical_claim(claim, verifier=None):
    """
    Test real citation verification with a medical claim using PubMed.
    
    Args:
        claim: The medical claim to verify
        verifier: Optional CitationVerifier instance
    """
    if not verifier:
        verifier = CitationVerifier()
    
    print(f"\nVerifying medical claim: '{claim}'")
    print("-" * 70)
    start_time = datetime.now()
    
    # Verify the claim
    citation = await verifier.verify_claim(claim, domain="medical", source_type=SourceType.PUBMED)
    
    # Calculate verification time
    verification_time = (datetime.now() - start_time).total_seconds()
    
    # Display verification results
    print(f"Verification completed in {verification_time:.2f} seconds")
    await display_citation_details(citation)
    
    return citation


async def main():
    """Run the real citation verification example."""
    print("EV0X Real Citation Verification Example")
    print("=" * 70)
    
    # Initialize the citation verifier
    verifier = CitationVerifier()
    
    # ===== TESTING ARXIV INTEGRATION =====
    print("\nTesting verification against scientific sources (arXiv)")
    print("=" * 70)
    
    # Test specific scientific claims with known references in arXiv
    scientific_claims = [
        "Transformer models have revolutionized natural language processing since their introduction in 2017.",
        "Deep learning models can be used for protein structure prediction.",
        "Quantum computing can potentially break current encryption methods.",
        "BERT is a transformer-based machine learning technique for natural language processing.",
        "Climate change is causing rising global temperatures and extreme weather events."
    ]
    
    arxiv_citations = []
    
    # Test each scientific claim
    for claim in scientific_claims:
        citation = await test_scientific_claim(claim, verifier)
        arxiv_citations.append(citation)
    
    # Summary of arXiv results
    print("\nSummary of arXiv Citation Verification")
    print("=" * 70)
    
    verified_count = sum(1 for c in arxiv_citations if c.verified)
    print(f"Total scientific claims tested: {len(arxiv_citations)}")
    print(f"Claims verified: {verified_count} ({verified_count/len(arxiv_citations)*100:.1f}% if any)")
    
    if verified_count > 0:
        avg_confidence = sum(c.confidence for c in arxiv_citations if c.verified) / verified_count
        print(f"Average confidence for verified claims: {avg_confidence:.2f}")
    
    # Example of direct verification of a text with multiple scientific claims
    print("\nVerifying a multi-claim scientific paragraph:")
    print("-" * 70)
    
    scientific_paragraph = """
    Machine learning has become a cornerstone of artificial intelligence research.
    Convolutional neural networks have achieved remarkable success in image recognition tasks.
    The attention mechanism in transformer models has enabled significant advances in natural language processing.
    Reinforcement learning has been used to train agents that can play games at superhuman levels.
    """
    
    print(scientific_paragraph.strip())
    
    # Verify the entire scientific paragraph
    verified_output = await verifier.verify_output(scientific_paragraph, domain="scientific", source_type=SourceType.ARXIV)
    
    print("\nVerified Scientific Output with Citations:")
    print("-" * 70)
    print(verified_output.verified_output)
    
    print(f"\nOverall Factual Confidence: {verified_output.overall_confidence:.2f}")
    print(f"Number of factual claims identified: {len(verified_output.citations)}")
    print(f"Number of claims verified with sources: {sum(1 for c in verified_output.citations if c.verified)}")
    
    # ===== TESTING PUBMED INTEGRATION =====
    print("\n\nTesting verification against medical sources (PubMed)")
    print("=" * 70)
    
    # Test specific medical claims with known references in PubMed
    medical_claims = [
        "Aspirin can reduce the risk of heart attacks in certain populations.",
        "Type 2 diabetes is associated with insulin resistance.",
        "Statins are effective in lowering LDL cholesterol levels.",
        "Vaccination has been proven effective in preventing infectious diseases.",
        "Hypertension is a risk factor for cardiovascular disease."
    ]
    
    pubmed_citations = []
    
    # Test each medical claim
    for claim in medical_claims:
        citation = await test_medical_claim(claim, verifier)
        pubmed_citations.append(citation)
    
    # Summary of PubMed results
    print("\nSummary of PubMed Citation Verification")
    print("=" * 70)
    
    verified_count = sum(1 for c in pubmed_citations if c.verified)
    print(f"Total medical claims tested: {len(pubmed_citations)}")
    print(f"Claims verified: {verified_count} ({verified_count/len(pubmed_citations)*100:.1f}% if any)")
    
    if verified_count > 0:
        avg_confidence = sum(c.confidence for c in pubmed_citations if c.verified) / verified_count
        print(f"Average confidence for verified claims: {avg_confidence:.2f}")
    
    # Example of direct verification of a text with multiple medical claims
    print("\nVerifying a multi-claim medical paragraph:")
    print("-" * 70)
    
    medical_paragraph = """
    Regular physical activity can help prevent chronic diseases such as heart disease and type 2 diabetes.
    Smoking is a major risk factor for lung cancer and cardiovascular disease.
    Proper hand hygiene is essential for preventing the spread of infectious diseases.
    Cognitive behavioral therapy is effective for treating anxiety disorders.
    """
    
    print(medical_paragraph.strip())
    
    # Verify the entire medical paragraph
    verified_output = await verifier.verify_output(medical_paragraph, domain="medical", source_type=SourceType.PUBMED)
    
    print("\nVerified Medical Output with Citations:")
    print("-" * 70)
    print(verified_output.verified_output)
    
    print(f"\nOverall Factual Confidence: {verified_output.overall_confidence:.2f}")
    print(f"Number of factual claims identified: {len(verified_output.citations)}")
    print(f"Number of claims verified with sources: {sum(1 for c in verified_output.citations if c.verified)}")
    
    # ===== COMBINED STATISTICS =====
    all_citations = arxiv_citations + pubmed_citations
    print("\n\nOverall Citation Verification Statistics")
    print("=" * 70)
    print(f"Total claims tested: {len(all_citations)}")
    
    verified_by_type = {}
    for citation in all_citations:
        source_type = citation.source_type.name if citation.source_type else "Unknown"
        if source_type not in verified_by_type:
            verified_by_type[source_type] = {"total": 0, "verified": 0}
        verified_by_type[source_type]["total"] += 1
        if citation.verified:
            verified_by_type[source_type]["verified"] += 1
    
    for source_type, stats in verified_by_type.items():
        percent = (stats["verified"] / stats["total"]) * 100 if stats["total"] > 0 else 0
        print(f"{source_type}: {stats['verified']}/{stats['total']} verified ({percent:.1f}%)")

if __name__ == "__main__":
    asyncio.run(main())
