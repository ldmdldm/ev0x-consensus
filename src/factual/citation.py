"""Citation verification module for ensuring factual correctness in AI outputs.

This module provides classes for creating, validating, and verifying citations 
in AI-generated content. It includes integration with academic sources like 
arXiv and PubMed to provide real factual verification.
"""

import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any, Union

import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SourceType(Enum):
    """Enumeration of supported citation source types."""
    ARXIV = "arxiv"
    PUBMED = "pubmed"
    UNKNOWN = "unknown"


class VerificationStatus(Enum):
    """Enumeration of verification statuses for citations."""
    VERIFIED = "verified"
    UNVERIFIED = "unverified"
    PARTIALLY_VERIFIED = "partially_verified"
    ERROR = "error"


class Citation:
    """Class representing a citation with metadata.
    
    A Citation includes the source information, relevance scores,
    and verification status.
    """
    
    def __init__(
        self,
        source_id: str,
        source_type: SourceType,
        title: str,
        authors: List[str],
        publication_date: Optional[datetime] = None,
        url: Optional[str] = None,
        doi: Optional[str] = None,
        abstract: Optional[str] = None,
        relevance_score: float = 0.0,
        verification_status: VerificationStatus = VerificationStatus.UNVERIFIED,
    ):
        """Initialize a Citation object.
        
        Args:
            source_id: Unique identifier for the citation source (e.g., arXiv ID)
            source_type: Type of source (e.g., arXiv, PubMed)
            title: Title of the cited work
            authors: List of authors
            publication_date: Publication date of the cited work
            url: URL to access the cited work
            doi: Digital Object Identifier
            abstract: Abstract or summary of the cited work
            relevance_score: Score indicating relevance to the claim (0.0-1.0)
            verification_status: Status of the verification
        """
        self.source_id = source_id
        self.source_type = source_type
        self.title = title
        self.authors = authors
        self.publication_date = publication_date
        self.url = url
        self.doi = doi
        self.abstract = abstract
        self.relevance_score = relevance_score
        self.verification_status = verification_status
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert Citation object to a dictionary.
        
        Returns:
            Dictionary representation of the Citation
        """
        return {
            "source_id": self.source_id,
            "source_type": self.source_type.value,
            "title": self.title,
            "authors": self.authors,
            "publication_date": self.publication_date.isoformat() if self.publication_date else None,
            "url": self.url,
            "doi": self.doi,
            "abstract": self.abstract,
            "relevance_score": self.relevance_score,
            "verification_status": self.verification_status.value,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Citation':
        """Create a Citation object from a dictionary.
        
        Args:
            data: Dictionary containing citation data
            
        Returns:
            A new Citation object
        """
        publication_date = None
        if data.get("publication_date"):
            try:
                publication_date = datetime.fromisoformat(data["publication_date"])
            except ValueError:
                logger.warning(f"Failed to parse publication date: {data['publication_date']}")
        
        return cls(
            source_id=data["source_id"],
            source_type=SourceType(data["source_type"]),
            title=data["title"],
            authors=data["authors"],
            publication_date=publication_date,
            url=data.get("url"),
            doi=data.get("doi"),
            abstract=data.get("abstract"),
            relevance_score=data.get("relevance_score", 0.0),
            verification_status=VerificationStatus(data["verification_status"]),
        )
    
    def __str__(self) -> str:
        """Return a string representation of the Citation.
        
        Returns:
            Formatted citation string
        """
        author_text = ", ".join(self.authors[:3])
        if len(self.authors) > 3:
            author_text += " et al."
        
        date_text = ""
        if self.publication_date:
            date_text = f" ({self.publication_date.year})"
        
        return f'"{self.title}" by {author_text}{date_text}. {self.source_type.value.upper()} ID: {self.source_id}'


class CitationVerifier:
    """Class for verifying claims using academic sources."""
    
    def __init__(self, cache_enabled: bool = True):
        """Initialize a CitationVerifier object.
        
        Args:
            cache_enabled: Whether to cache verification results
        """
        self.cache_enabled = cache_enabled
        self.cache = {}  # Simple in-memory cache
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def verify_claim(self, claim: str, source_types: Optional[Union[SourceType, List[SourceType]]] = None) -> Tuple[bool, List[Citation]]:
        """Verify a claim using available academic sources.
        
        Args:
            claim: The claim to verify
            source_types: Source type(s) to check - can be a single SourceType or a list of SourceTypes (default: all)
            
        Returns:
            Tuple of (is_verified, list_of_citations)
        """
        if not claim or not claim.strip():
            logger.warning("Empty claim provided for verification")
            return False, []
        
        # Check cache if enabled
        if self.cache_enabled and claim in self.cache:
            logger.info(f"Using cached verification for claim: {claim[:50]}...")
            return self.cache[claim]
        
        # Default to all source types if none specified
        if not source_types:
            source_types = [SourceType.ARXIV, SourceType.PUBMED]
        # Convert single SourceType to a list if needed
        elif isinstance(source_types, SourceType):
            source_types = [source_types]
        
        all_citations = []
        start_time = time.time()
        
        # Try each source type
        for source_type in source_types:
            try:
                if source_type == SourceType.ARXIV:
                    citations = self._search_arxiv(claim)
                elif source_type == SourceType.PUBMED:
                    citations = self._search_pubmed(claim)
                else:
                    logger.warning(f"Unsupported source type: {source_type}")
                    continue
                
                all_citations.extend(citations)
            except Exception as e:
                logger.error(f"Error verifying claim with {source_type}: {str(e)}")
        
        # Calculate relevance scores
        all_citations = self._calculate_relevance_scores(claim, all_citations)
        
        # Sort by relevance score (descending)
        all_citations.sort(key=lambda x: x.relevance_score, reverse=True)
        
        # Determine verification status
        # Determine verification status with adjusted thresholds for technical content
        if all_citations:
            # Check if the claim has technical terms that would justify lower thresholds
            has_technical_terms = any(term in claim.lower() for term in [
                'algorithm', 'model', 'neural', 'bert', 'gpt', 'transformer', 'nlp', 'learning',
                'convolutional', 'lstm', 'attention', 'embedding', 'fine-tuning', 'pre-training'
            ])
            
            # Apply different thresholds based on content type
            if has_technical_terms:
                # Use lower thresholds for technical content
                if all_citations[0].relevance_score > 0.45:  # Further reduced for technical content
                    is_verified = True
                    all_citations[0].verification_status = VerificationStatus.VERIFIED
                elif len(all_citations) >= 2 and all_citations[0].relevance_score > 0.37 and all_citations[1].relevance_score > 0.32:
                    # Even lower threshold for multiple technical citations
                    is_verified = True
                    all_citations[0].verification_status = VerificationStatus.PARTIALLY_VERIFIED
                    all_citations[1].verification_status = VerificationStatus.PARTIALLY_VERIFIED
            else:
                # Standard thresholds for non-technical content
                if all_citations[0].relevance_score > 0.5:
                    is_verified = True
                    all_citations[0].verification_status = VerificationStatus.VERIFIED
                elif len(all_citations) >= 2 and all_citations[0].relevance_score > 0.4 and all_citations[1].relevance_score > 0.35:
                    # Consider multiple supporting citations with moderate scores as verified
                    is_verified = True
                    all_citations[0].verification_status = VerificationStatus.PARTIALLY_VERIFIED
                    all_citations[1].verification_status = VerificationStatus.PARTIALLY_VERIFIED
        elapsed_time = time.time() - start_time
        logger.info(f"Verification completed in {elapsed_time:.2f}s, found {len(all_citations)} citations")
        
        # Cache the result if enabled
        if self.cache_enabled:
            self.cache[claim] = (is_verified, all_citations)
        
        return is_verified, all_citations
    
    def _search_arxiv(self, query: str, max_results: int = 5) -> List[Citation]:
        """Search arXiv for relevant papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of Citation objects from arXiv
        """
        logger.info(f"Searching arXiv for: {query[:50]}...")
        
        # Clean and format the query for arXiv API
        # Extract key terms for more focused arXiv search
        # Identify and extract important scientific/academic terms
        cleaned_query = query.strip()
        # Extract key academic terms using a more focused approach
        academic_terms = []
        
        # Look for key scientific terms with specific patterns that indicate important concepts
        term_patterns = [
            # Technical/scientific terms
            r'\b(?:algorithm|model|framework|method|approach|technique|system|architecture|mechanism)\b',
            # Scientific measurements and concepts
            r'\b(?:accuracy|precision|recall|efficiency|performance|throughput|latency|scalability)\b',
            # Technical fields - expanded to catch more variants
            r'\b(?:machine learning|deep learning|neural network|artificial intelligence|computer vision|NLP|natural language processing|data mining|information retrieval)\b',
            # Mathematical concepts - expanded
            r'\b(?:function|equation|theorem|probability|statistical|distribution|optimization|bayesian|linear algebra|vector|matrix|tensor)\b',
            # Research elements
            r'\b(?:experiment|result|evaluation|analysis|assessment|evidence|validation|measurement|benchmark|baseline|ablation study)\b',
        ]
        
        # Technical/scientific compound terms with high relevance - expanded with more specific technical terms
        specific_technical_terms = [
            # AI/ML specific models and techniques - expanded with more model names
            r'\b(?:deep reinforcement learning|convolutional neural network|transformer model|attention mechanism|generative adversarial network|large language model|LLM|GPT|ResNet|BERT|NLP|T5|RoBERTa|DeBERTa|DistilBERT|XLNet|ALBERT|ELECTRA|CLIP|DALL-E|Stable Diffusion|diffusion model)\b',
            # Graph and knowledge terms
            r'\b(?:knowledge graph|semantic network|graph neural network|transfer learning|semi-supervised learning|self-supervised learning|contrastive learning|multimodal learning|few-shot learning|zero-shot learning|meta-learning|active learning)\b',
            # Architectures and components
            r'\b(?:LSTM|GRU|RNN|CNN|VAE|autoencoder|transformer|encoder-decoder|embedding|fine-tuning|pre-training|tokenization|self-attention|cross-attention|multi-head attention)\b'
        ]
        # Extract significant terms
        for pattern in term_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            academic_terms.extend(matches)
            
        # Extract specific technical compound terms with higher priority
        for pattern in specific_technical_terms:
            matches = re.findall(pattern, query, re.IGNORECASE)
            # Add these to the front of the list as they're more important
            academic_terms = matches + academic_terms
            
        # Extract quoted phrases - these are likely important specific concepts
        exact_phrases = re.findall(r'"([^"]+)"', cleaned_query)
        academic_terms.extend(exact_phrases)

        # Handle year mentions (publications are often cited by year)
        years = re.findall(r'\b(19|20)\d{2}\b', cleaned_query)
        academic_terms.extend(years)

        # If we found key terms, build a query focused on them
        if academic_terms:
            # Remove duplicates while preserving order
            # Keep more technical terms to improve search results
            # Increase limit to preserve more technical terms (from 8 to 12)
            if len(academic_terms) > 12:
                # Prioritize specific technical terms over general ones
                # Sort terms by length (longer terms tend to be more specific technical terms)
                academic_terms.sort(key=len, reverse=True)
                academic_terms = academic_terms[:12]
                
            # Create a more focused query with the key terms
            simplified_query = " ".join(academic_terms)
            logger.info(f"Simplified arXiv query from '{query[:50]}...' to '{simplified_query}'")
            logger.info(f"Simplified arXiv query from '{query[:50]}...' to '{simplified_query}'")
            cleaned_query = simplified_query
        # Handle exact phrases in quotes
        exact_phrases = re.findall(r'"([^"]+)"', cleaned_query)
        for phrase in exact_phrases:
            # Replace space with + inside quotes for exact phrase matching
            formatted_phrase = phrase.replace(' ', '+')
            cleaned_query = cleaned_query.replace(f'"{phrase}"', f'"{formatted_phrase}"')
        
        # Clean other parts of the query
        parts = re.split(r'("[^"]*")', cleaned_query)
        for i in range(len(parts)):
            if not parts[i].startswith('"'):
                # Only clean non-quoted parts
                parts[i] = re.sub(r'[^\w\s]', ' ', parts[i])
                parts[i] = re.sub(r'\s+', ' ', parts[i]).strip()
        
        cleaned_query = ''.join(parts)
        
        # Prepare the final query string
        # Replace spaces with + but preserve quoted sections
        parts = re.split(r'("[^"]*")', cleaned_query)
        for i in range(len(parts)):
            if not parts[i].startswith('"'):
                parts[i] = parts[i].replace(' ', '+')
        
        final_query = ''.join(parts)
        # Build advanced query with proper syntax
        # Use ti: for title, au: for author if detected
        if re.search(r'\b(?:by|author)\b', query, re.IGNORECASE):
            # Extract potential author names after "by" or "author"
            author_match = re.search(r'\b(?:by|author)[s]?\s+([A-Za-z\s,\.]+)', query, re.IGNORECASE)
            if author_match:
                author_part = author_match.group(1).strip()
                # Remove the author part from the main query to avoid duplication
                query_without_author = re.sub(r'\b(?:by|author)[s]?\s+[A-Za-z\s,\.]+', '', query, flags=re.IGNORECASE)
                search_query = f'au:"{author_part}" AND all:({query_without_author})'
            else:
                search_query = f"all:{final_query}"
        else:
            search_query = f"all:{final_query}"
        
        base_url = "https://export.arxiv.org/api/query"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        
        # Add rate limiting compliance - pause briefly to avoid overwhelming the API
        time.sleep(0.5)  # Sleep for 500ms before making the request
        
        try:
            headers = {
                'User-Agent': 'ev0x-citation-verifier/1.0 (https://github.com/flare-research/flare-ai-consensus)'
            }
            
            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            
            # Check for various HTTP status codes
            if response.status_code == 429:
                logger.warning("arXiv API rate limit exceeded. Waiting before retrying.")
                time.sleep(5)  # Wait 5 seconds before potentially retrying
                return []
            
            response.raise_for_status()
            
            # Parse XML response with proper namespace handling
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError as e:
                logger.error(f"XML parsing error: {str(e)}")
                logger.debug(f"Response content: {response.text[:500]}...")
                return []
            
            # Define namespaces - ensure all required namespaces are included
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom',
                'opensearch': 'http://a9.com/-/spec/opensearch/1.1/'
            }
            
            # Check if we got any results
            total_results_elem = root.find('.//opensearch:totalResults', ns)
            if total_results_elem is not None:
                total_results = int(total_results_elem.text)
                if total_results == 0:
                    logger.info("No arXiv results found for query.")
                    return []
                logger.info(f"Found {total_results} total results on arXiv, retrieving top {max_results}")
            
            citations = []
            
            # Process each entry with better error handling
            for entry in root.findall('.//atom:entry', ns):
                try:
                    # Skip the first entry if it's the OpenSearch Description
                    if entry.find('./atom:title', ns) is not None and entry.find('./atom:title', ns).text == 'ArXiv Query: search_query=all:':
                        continue
                    
                    title_elem = entry.find('./atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None and title_elem.text else "Unknown title"
                    
                    # Extract arXiv ID
                    id_elem = entry.find('./atom:id', ns)
                    arxiv_id = "unknown"
                    if id_elem is not None and id_elem.text:
                        id_text = id_elem.text
                        # Improve regex to handle more arXiv ID formats
                        arxiv_id_match = re.search(r'arxiv\.org/abs/([^/\s]+)', id_text)
                        if arxiv_id_match:
                            arxiv_id = arxiv_id_match.group(1)
                        else:
                            # Try alternate format
                            alt_match = re.search(r'([\d\.]+(?:v\d+)?)', id_text)
                            if alt_match:
                                arxiv_id = alt_match.group(1)
                    
                    # Extract authors with better handling
                    authors = []
                    for author_elem in entry.findall('./atom:author/atom:name', ns):
                        if author_elem is not None and author_elem.text:
                            # Clean up author name
                            author_name = author_elem.text.strip()
                            # Handle cases where names are in "LastName, FirstName" format
                            if ',' in author_name:
                                parts = author_name.split(',', 1)
                                if len(parts) == 2:
                                    author_name = f"{parts[1].strip()} {parts[0].strip()}"
                            authors.append(author_name)
                    
                    # Extract primary category (field of study)
                    primary_category = None
                    category_elem = entry.find('./arxiv:primary_category', ns)
                    if category_elem is not None and 'term' in category_elem.attrib:
                        primary_category = category_elem.attrib['term']
                    
                    # Extract publication date with better error handling
                    published_elem = entry.find('./atom:published', ns)
                    publication_date = None
                    if published_elem is not None and published_elem.text:
                        try:
                            # Handle different date formats
                            date_text = published_elem.text.strip()
                            if date_text.endswith('Z'):
                                date_text = date_text.replace('Z', '+00:00')
                            # Try multiple date formats
                            try:
                                publication_date = datetime.fromisoformat(date_text)
                            except ValueError:
                                try:
                                    publication_date = datetime.strptime(date_text, '%Y-%m-%dT%H:%M:%SZ')
                                except ValueError:
                                    publication_date = datetime.strptime(date_text, '%Y-%m-%d')
                        except Exception as e:
                            logger.warning(f"Could not parse publication date: {published_elem.text} - {str(e)}")
                    
                    # Extract abstract with better text cleaning
                    summary_elem = entry.find('./atom:summary', ns)
                    abstract = None
                    if summary_elem is not None and summary_elem.text:
                        # Clean up abstract text - normalize whitespace, remove any XML entities
                        abstract = re.sub(r'\s+', ' ', summary_elem.text).strip()
                    
                    # Extract DOI with improved parsing
                    doi = None
                    journal_ref = None
                    for link_elem in entry.findall('./atom:link', ns):
                        if link_elem.get('title') == 'doi' or link_elem.get('rel') == 'related' and 'doi.org' in link_elem.get('href', ''):
                            doi_url = link_elem.get('href')
                            doi_match = re.search(r'doi\.org/(.+)$', doi_url)
                            if doi_match:
                                doi = doi_match.group(1)
                    
                    # Extract URL to the paper
                    url = None
                    for link_elem in entry.findall('./atom:link', ns):
                        # Primary URL is usually the one with rel='alternate'
                        if link_elem.get('rel') == 'alternate' or 'arxiv.org/abs/' in link_elem.get('href', ''):
                            url = link_elem.get('href')
                            break
                    
                    # If we couldn't find the alternate link, use the id URL
                    if not url and id_elem is not None and id_elem.text:
                        url = id_elem.text
                    
                    # Create citation object with proper error handling for missing fields
                    citation = Citation(
                        source_id=arxiv_id,
                        source_type=SourceType.ARXIV,
                        title=title,
                        authors=authors if authors else ["Unknown author"],
                        publication_date=publication_date,
                        url=url,
                        doi=doi,
                        abstract=abstract,
                        relevance_score=0.0,  # Will be calculated later
                        verification_status=VerificationStatus.UNVERIFIED,
                    )
                    
                    citations.append(citation)
                    
                except Exception as e:
                    logger.error(f"Error processing arXiv entry: {str(e)}")
            
            logger.info(f"Found {len(citations)} arXiv citations")
            return citations
            
        except requests.RequestException as e:
            logger.error(f"arXiv API request failed: {str(e)}")
            return []
        except ET.ParseError as e:
            logger.error(f"Failed to parse arXiv XML response: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in arXiv search: {str(e)}")
            return []
    
    def _search_pubmed(self, query: str, max_results: int = 5) -> List[Citation]:
        """Search PubMed for relevant papers matching the query.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            
        Returns:
            List of Citation objects from PubMed
        """
        logger.info(f"Searching PubMed for: {query[:50]}...")
        
        # Process the query to optimize for medical/scientific literature
        # Extract potential medical terms using regex patterns
        medical_terms = []
        
        # Look for medical terminology patterns
        term_patterns = [
            r'\b(?:disease|syndrome|disorder|condition|infection|virus|bacteria|pathogen|treatment|therapy|medication|drug|diagnosis|prognosis|symptom|medicine|vaccine|immunity|antibody|gene|protein|receptor|enzyme|cell|tissue|organ)\b',
            r'\b(?:clinical|medical|therapeutic|pharmaceutical|biological|genetic|molecular|cellular|pathological|immunological|neurological|cardiovascular|endocrine|gastrointestinal|respiratory|renal|hepatic|musculoskeletal|dermatological|hematological|oncological)\b',
        ]
        
        for pattern in term_patterns:
            terms = re.findall(pattern, query, re.IGNORECASE)
            medical_terms.extend(terms)
        
        # Build an enhanced query with MeSH terms if available
        enhanced_query = query
        if medical_terms:
            # Boost important medical terms with field filters
            medical_terms_str = " OR ".join([f'"{term}"[Title/Abstract]' for term in medical_terms])
            if medical_terms_str:
                enhanced_query = f"({query}) AND ({medical_terms_str})"
                logger.info(f"Enhanced PubMed query with medical terms: {enhanced_query[:100]}...")
        
        # Clean the query for PubMed API format
        clean_query = enhanced_query.strip()
        
        # Step 1: Search for PMIDs using E-utilities
        esearch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        esearch_params = {
            "db": "pubmed",
            "term": clean_query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "usehistory": "y",  # Use history server for more efficient subsequent requests
            "tool": "ev0x-citation-verifier",
            "email": "info@flare-research.org",  # Best practice to include contact info
        }
        
        try:
            # Add rate limiting compliance
            time.sleep(0.3)  # Sleep to comply with NCBI's rate limit recommendations
            
            search_response = requests.get(esearch_url, params=esearch_params, timeout=10)
            
            # Check specific PubMed error codes
            if search_response.status_code == 429:
                logger.warning("PubMed API rate limit exceeded. Waiting before retrying.")
                time.sleep(3)  # Wait longer before potential retry
                return []
            
            search_response.raise_for_status()
            
            # Parse response with better error handling
            try:
                search_data = search_response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse PubMed search response as JSON: {str(e)}")
                logger.debug(f"Response content: {search_response.text[:500]}...")
                return []
            
            # Extract PMIDs and query information for efficient fetching
            esearchresult = search_data.get('esearchresult', {})
            pmids = esearchresult.get('idlist', [])
            
            # Get search history information for more efficient batch retrieval
            webenv = esearchresult.get('webenv', '')
            query_key = esearchresult.get('querykey', '')
            
            if not pmids:
                logger.info("No PubMed results found")
                return []
            
            # Log the number of results found
            count = esearchresult.get('count', '0')
            logger.info(f"Found {count} total results on PubMed, retrieving top {len(pmids)}")
            
            # Step 2: Fetch detailed information for PMIDs using history parameters if available
            efetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            efetch_params = {
                "db": "pubmed",
                "retmode": "xml",
                "tool": "ev0x-citation-verifier",
                "email": "info@flare-research.org",
            }
            
            # Use WebEnv and query_key if available, otherwise fall back to ID list
            if webenv and query_key:
                efetch_params.update({
                    "webenv": webenv,
                    "query_key": query_key,
                    "retmax": max_results,
                })
            else:
                efetch_params["id"] = ",".join(pmids)
            
            # Add a slight delay to respect rate limits
            time.sleep(0.3)
            
            fetch_response = requests.get(efetch_url, params=efetch_params, timeout=15)
            fetch_response.raise_for_status()
            
            # Parse XML response with better error handling
            try:
                root = ET.fromstring(fetch_response.content)
            except ET.ParseError as e:
                logger.error(f"XML parsing error: {str(e)}")
                logger.debug(f"Response content: {fetch_response.text[:500]}...")
                return []
            
            citations = []
            
            # Process each PubMed article with proper XML namespace handling
            # Define XML namespaces used in PubMed responses
            ns = {
                'p': 'https://www.ncbi.nlm.nih.gov/pubmed/',
                'n': 'https://www.ncbi.nlm.nih.gov/ncbi_dtd',
            }
            
            # Find articles with or without namespaces (namespace handling can be tricky in PubMed responses)
            articles = root.findall(".//PubmedArticle") or root.findall(".//p:PubmedArticle", ns)
            for article in articles:
                try:
                    # Extract PMID with fallback for different XML structures
                    pmid_elem = (
                        article.find(".//PMID") or 
                        article.find(".//p:PMID", ns) or
                        article.find(".//ArticleId[@IdType='pubmed']") or
                        article.find(".//p:ArticleId[@IdType='pubmed']", ns)
                    )
                    pmid = pmid_elem.text if pmid_elem is not None else "unknown"
                    
                    # Extract title with better text handling for XML entities and formatting
                    title_elem = article.find(".//ArticleTitle") or article.find(".//p:ArticleTitle", ns)
                    
                    if title_elem is not None:
                        # Handle cases where title contains nested XML elements
                        if title_elem.text is not None:
                            title = title_elem.text
                        else:
                            # Concatenate all text content from child elements
                            title = "".join(title_elem.itertext())
                    else:
                        title = "Unknown title"
                    
                    # Clean up title text - normalize whitespace, remove any XML artifacts
                    title = re.sub(r'\s+', ' ', title).strip()
                    
                    # Extract authors with comprehensive name handling
                    authors = []
                    author_list = (
                        article.find(".//AuthorList") or 
                        article.find(".//p:AuthorList", ns) or
                        article.find(".//Authors")
                    )
                    
                    if author_list is not None:
                        for author_elem in author_list.findall(".//Author") or author_list.findall(".//p:Author", ns):
                            # Check for different name element formats
                            last_name = (
                                author_elem.find(".//LastName") or 
                                author_elem.find(".//p:LastName", ns) or
                                author_elem.find(".//Surname")
                            )
                            
                            # Check multiple possible element names for first name
                            fore_name = (
                                author_elem.find(".//ForeName") or 
                                author_elem.find(".//p:ForeName", ns) or
                                author_elem.find(".//FirstName") or
                                author_elem.find(".//GivenName")
                            )
                            
                            # Check for collective/group author names
                            collective_name = (
                                author_elem.find(".//CollectiveName") or
                                author_elem.find(".//p:CollectiveName", ns)
                            )
                            
                            if collective_name is not None and collective_name.text:
                                authors.append(collective_name.text.strip())
                                continue
                                
                            author_name_parts = []
                            if last_name is not None and last_name.text:
                                author_name_parts.append(last_name.text.strip())
                            
                            if fore_name is not None and fore_name.text:
                                author_name_parts.insert(0, fore_name.text.strip())
                            
                            if author_name_parts:
                                author_name = " ".join(author_name_parts)
                                authors.append(author_name)
                    
                    # Extract publication date
                    pub_date = article.find(".//PubDate")
                    publication_date = None
                    if pub_date is not None:
                        year_elem = pub_date.find(".//Year")
                        month_elem = pub_date.find(".//Month")
                        day_elem = pub_date.find(".//Day")
                        
                        year = year_elem.text if year_elem is not None else None
                        month = month_elem.text if month_elem is not None else "1"
                        day = day_elem.text if day_elem is not None else "1"
                        
                        # Convert month name to number if needed
                        try:
                            # Check if month is a name rather than a number
                            if month and month.isalpha():
                                # Convert month name to number
                                month_names = {
                                    'jan': '1', 'feb': '2', 'mar': '3', 'apr': '4', 
                                    'may': '5', 'jun': '6', 'jul': '7', 'aug': '8', 
                                    'sep': '9', 'oct': '10', 'nov': '11', 'dec': '12'
                                }
                                month_abbr = month.lower()[:3]
                                if month_abbr in month_names:
                                    month = month_names[month_abbr]
                            
                            # Create datetime object if year is available
                            if year:
                                publication_date = datetime(int(year), int(month), int(day))
                        except ValueError:
                            logger.warning(f"Could not parse PubMed date: {year}-{month}-{day}")
                    
                    # Extract abstract
                    abstract_elem = article.find(".//AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else None
                    
                    # Extract DOI
                    doi = None
                    article_id_list = article.find(".//ArticleIdList")
                    if article_id_list is not None:
                        for id_elem in article_id_list.findall(".//ArticleId"):
                            if id_elem.get("IdType") == "doi":
                                doi = id_elem.text
                    # Build URL from PMID
                    url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid != "unknown" else None
                    
                    citation = Citation(
                        source_id=pmid,
                        source_type=SourceType.PUBMED,
                        title=title,
                        authors=authors,
                        publication_date=publication_date,
                        url=url,
                        doi=doi,
                        abstract=abstract,
                        relevance_score=0.0,  # Will be calculated later
                        verification_status=VerificationStatus.UNVERIFIED,
                    )
                    
                    citations.append(citation)
                    
                except Exception as e:
                    logger.error(f"Error processing PubMed article: {str(e)}")
            
            logger.info(f"Found {len(citations)} PubMed citations")
            return citations
            
        except requests.RequestException as e:
            logger.error(f"PubMed API request failed: {str(e)}")
            return []
        except ET.ParseError as e:
            logger.error(f"Failed to parse PubMed XML response: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error in PubMed search: {str(e)}")
            return []
    
    def _extract_key_phrases(self, text: str) -> List[str]:
        """Extract key phrases and important keywords from text.
        
        This method identifies and extracts important terms, phrases, and concepts
        from the provided text to improve relevance matching.
        
        Args:
            text: The text to extract key phrases from
            
        Returns:
            List of extracted key phrases and important keywords
        """
        if not text or not isinstance(text, str):
            return []
            
        try:
            # Initialize the list to store key phrases
            key_phrases = []
            
            # Extract quoted content - these are considered important phrases
            try:
                quoted_phrases = re.findall(r'"([^"]+)"', text)
                key_phrases.extend([phrase.strip() for phrase in quoted_phrases if phrase.strip()])
            except Exception as e:
                logger.warning(f"Error extracting quoted phrases: {str(e)}")
            
            # Extract numerical data with units - important for scientific/medical claims
            try:
                numerical_data = re.findall(r'\b\d+(?:\.\d+)?(?:\s*%|\s*percent|\s*mg|\s*kg|\s*ml|\s*g)\b', text, re.IGNORECASE)
                key_phrases.extend([data.strip() for data in numerical_data if data.strip()])
            except Exception as e:
                logger.warning(f"Error extracting numerical data: {str(e)}")
            
            # Extract years - often important in citations
            try:
                years = re.findall(r'\b(19|20)\d{2}\b', text)
                key_phrases.extend(years)
            except Exception as e:
                logger.warning(f"Error extracting years: {str(e)}")
            
            # Extract scientific/technical terms based on domain-specific patterns
            scientific_patterns = [
                # Academic concepts
                r'\b(?:algorithm|theorem|model|framework|method|approach|theory|paradigm)\b',
                # Academic concepts - expanded with more specific technical terms
                r'\b(?:architecture|pipeline|system|module)\b',
                # Data and metrics - expanded with more ML metrics
                r'\b(?:accuracy|precision|recall|f1|specificity|sensitivity|error rate|auc|roc|map|ndcg|perplexity|bleu|rouge|meteor)\b',
                # Technical fields - expanded with more technical areas
                r'\b(?:machine learning|deep learning|neural network|artificial intelligence|computer vision|NLP|reinforcement learning|generative AI|large language model|foundation model|multimodal learning)\b',
                # ML/AI specific terms with high relevance for technical content
                r'\b(?:BERT|GPT|transformer|embedding|fine-tuning|pre-training|attention|self-attention|cross-attention|head|layer|token|tokenization|parameter|weight|bias|gradient|backpropagation)\b'
            ]
            
            medical_patterns = [
                # Medical concepts
                r'\b(?:disease|syndrome|disorder|condition|diagnosis|prognosis|treatment|therapy)\b',
                # Medications and interventions
                r'\b(?:medication|drug|vaccine|antibody|protein|gene|receptor|enzyme)\b',
                # Medical measurements
                r'\b(?:mortality|morbidity|prevalence|incidence|risk factor|biomarker)\b'
            ]
            
            # Combine all patterns for extraction
            all_patterns = scientific_patterns + medical_patterns
            
            for pattern in all_patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    key_phrases.extend([match.strip() for match in matches if match.strip()])
                except Exception as e:
                    logger.warning(f"Error with pattern {pattern}: {str(e)}")
            
            # Extract multi-word technical terms (noun phrases)
            # Extract multi-word technical terms (noun phrases)
            # Extract multi-word technical terms (noun phrases)
            # This is a simple approach - in production would use NLP libraries
            multi_word_patterns = [
                # Multi-word scientific terms - expanded with more ML terms
                r'\b(?:convolutional neural network|recurrent neural network|generative adversarial network|support vector machine|decision tree|random forest|gradient boosting|deep reinforcement learning|variational autoencoder|graph neural network|mixture of experts|encoder-decoder architecture|sequence-to-sequence model)\b',
                # Multi-word medical terms
                r'\b(?:randomized controlled trial|systematic review|meta analysis|clinical trial|case report|cohort study|double blind|placebo controlled)\b',
                # Add specific technical AI/ML terms that are commonly used in queries - expanded with versions and variants
                r'\b(?:BERT|GPT-3|GPT-4|GPT-3\.5|transformer|attention mechanism|ResNet|language model|neural architecture|self-supervised learning|masked language modeling|next token prediction|contrastive learning|prompt engineering|in-context learning|chain-of-thought|few-shot learning|zero-shot learning)\b',
                # Add specific architectures and versions
                r'\b(?:BERT-base|BERT-large|RoBERTa|T5|T5-base|T5-large|DeBERTa|XLNet|ALBERT|ELECTRA|DALL-E|CLIP|Llama|Llama-2|Mistral|Stable Diffusion|MidJourney|PaLM|Bard|Claude|Gemini)\b'
            ]
            
            for pattern in multi_word_patterns:
                try:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    key_phrases.extend([match.strip() for match in matches if match.strip()])
                except Exception as e:
                    logger.warning(f"Error with multi-word pattern {pattern}: {str(e)}")
            
            # Remove duplicates while preserving order
            # In case no patterns matched, extract some nouns and noun phrases as fallback
            # Simple noun extraction - look for capitalized words not at beginning of sentences
            # Using a simpler pattern to avoid look-behind issues
            capitalized_nouns = []
            try:
                # Find sentence boundaries first
                sentence_starts = [0] + [m.end() for m in re.finditer(r'[.!?]\s+', text)]
                
                # Then find capitalized words that aren't at the start of sentences
                for match in re.finditer(r'\b[A-Z][a-z]+\b', text):
                    is_sentence_start = False
                    for start in sentence_starts:
                        if match.start() == start:
                            is_sentence_start = True
                            break
                    if not is_sentence_start:
                        capitalized_nouns.append(match.group())
            except Exception as e:
                logger.warning(f"Error extracting capitalized nouns: {str(e)}")
            
            # Add the capitalized nouns to key phrases
            key_phrases.extend(capitalized_nouns)
            
            # Add some important words as backup
            important_words = re.findall(r'\b(?:significant|important|novel|effective|improved|advanced|innovative)\b', text, re.IGNORECASE)
            key_phrases.extend(important_words)
            return key_phrases
            
        except Exception as e:
            logger.error(f"Error extracting key phrases: {str(e)}")
            # Return empty list on error - relevance scoring will fall back to regular text
            return []
    
    def _calculate_relevance_scores(self, claim: str, citations: List[Citation]) -> List[Citation]:
        """Calculate relevance scores for citations based on text similarity to claim.
        
        Args:
            claim: The claim being verified
            citations: List of Citation objects to score
            
        Returns:
            List of Citation objects with updated relevance scores
        """
        if not citations:
            return []
            
        logger.info(f"Calculating relevance scores for {len(citations)} citations")
        
        # Improved text matching through claim preprocessing
        # Extract key phrases and concepts from the claim
        key_phrases = self._extract_key_phrases(claim)
        # Create two versions of the claim for matching:
        # 1. The original claim (for full context matching)
        # 2. A focused version with extracted key phrases (for concept matching)
        focused_claim = " ".join(key_phrases) if key_phrases else claim
        
        # Prepare texts for comparison with dual matching approach
        original_texts = [claim]  # For original matching
        focused_texts = [focused_claim]  # For focused concept matching
        
        for citation in citations:
            # Combine title and abstract for better matching
            citation_text = citation.title
            if citation.abstract:
                citation_text += " " + citation.abstract
            
            original_texts.append(citation_text)
            
            # For focused matching, extract key phrases from the citation as well
            try:
                citation_key_phrases = self._extract_key_phrases(citation_text)
                focused_citation = " ".join(citation_key_phrases) if citation_key_phrases else citation_text
            except Exception as e:
                logger.warning(f"Error extracting key phrases from citation: {str(e)}")
                focused_citation = citation_text  # Fallback to full text if extraction fails
                
            focused_texts.append(focused_citation)
        
        try:
            # Generate TF-IDF matrices for both approaches
            original_tfidf_matrix = self.vectorizer.fit_transform(original_texts)
            
            # Initialize variables for fallback if focused matching fails
            focused_similarities = None
            
            # Try the focused approach, but have a fallback if it fails
            try:
                # Create a new vectorizer for the focused approach to avoid interference
                focused_vectorizer = TfidfVectorizer(stop_words='english')
                focused_tfidf_matrix = focused_vectorizer.fit_transform(focused_texts)
                
                # Calculate cosine similarity for the focused approach
                focused_claim_vector = focused_tfidf_matrix[0:1]
                focused_citation_vectors = focused_tfidf_matrix[1:]
                focused_similarities = cosine_similarity(focused_claim_vector, focused_citation_vectors).flatten()
            except Exception as e:
                logger.warning(f"Focused matching failed, falling back to original method: {str(e)}")
                # Focused matching failed (e.g., if we have no common terms), fall back to None
            
            # Calculate cosine similarity for the original approach
            # Calculate cosine similarity for the original approach
            original_citation_vectors = original_tfidf_matrix[1:]
            original_similarities = cosine_similarity(original_claim_vector, original_citation_vectors).flatten()
            # Check for exact technical term matches to apply boosting
            # Extract technical terms from the claim
            technical_terms = []
            specific_technical_terms = [
                r'\b(?:BERT|GPT|transformer|embedding|fine-tuning|pre-training|attention)\b',
                r'\b(?:neural network|machine learning|deep learning|reinforcement learning)\b',
                r'\b(?:convolutional|recurrent|generative|adversarial|autoencoder)\b'
            ]
            for pattern in specific_technical_terms:
                matches = re.findall(pattern, claim, re.IGNORECASE)
                technical_terms.extend(matches)
            
            # Initialize boosting factors array
            term_match_boosts = np.ones_like(original_similarities)
            
            # Apply boosting for exact technical term matches
            for i, citation in enumerate(citations):
                citation_text = citation.title
                if citation.abstract:
                    citation_text += " " + citation.abstract
                
                # Count how many technical terms match
                match_count = 0
                for term in technical_terms:
                    if re.search(r'\b' + re.escape(term) + r'\b', citation_text, re.IGNORECASE):
                        match_count += 1
                
                # Apply incremental boost based on number of exact matches
                if match_count > 0:
                    # Apply higher boost for more matches (up to 25% boost)
                    boost_factor = min(1.25, 1 + (match_count * 0.05))
                    term_match_boosts[i] = boost_factor
            
            # Determine final similarities based on whether focused matching worked
            if focused_similarities is not None:
                # Combine the similarity scores using a weighted approach
                # Give slightly more weight to the focused matching for relevant concept detection
                # Apply technical term match boosting as a multiplier
                similarities = ((0.4 * original_similarities) + (0.6 * focused_similarities)) * term_match_boosts
            else:
                # Fall back to original similarities if focused matching failed
                # Still apply technical term match boosting
                similarities = original_similarities * term_match_boosts
            
            
            # Update relevance scores with improved scoring logic
            for i, score in enumerate(similarities):
                base_score = float(score)
                
                # Apply additional scoring factors based on metadata quality
                citation = citations[i]
                
                # Boost score if the paper has a DOI (indicates peer review)
                if citation.doi:
                    base_score = min(1.0, base_score * 1.1)
                
                # Boost newer papers slightly (more recent research)
                if citation.publication_date:
                    current_year = datetime.now().year
                    pub_year = citation.publication_date.year
                    # Papers from last 5 years get a small boost
                    if current_year - pub_year <= 5:
                        recency_boost = 1.0 + (0.02 * (5 - (current_year - pub_year)))
                        base_score = min(1.0, base_score * recency_boost)
                
                # Adjust score based on amount of content available for matching
                # More content = more reliable matching
                content_length = len(citation.title)
                if citation.abstract:
                    content_length += len(citation.abstract)
                
                if content_length > 500:  # Good amount of content for matching
                    content_boost = 1.05
                elif content_length < 100:  # Limited content for matching
                    content_boost = 0.95
                else:
                    content_boost = 1.0
                
                base_score = min(1.0, base_score * content_boost)
                
                # Set final score
                citations[i].relevance_score = base_score
                
                # Update verification status based on score
            logger.info(f"Relevance scores calculated. Max score: {max(similarities) if similarities.size > 0 else 0}")
            
        except Exception as e:
            logger.error(f"Error calculating relevance scores: {str(e)}")
            
        return citations
    
    def extract_claims(self, text: str, max_claims: int = 10) -> List[str]:
        """Extract potential factual claims from a text.
        
        Args:
            text: Text to extract claims from
            max_claims: Maximum number of claims to extract
            
        Returns:
            List of extracted claim statements
        """
        if not text or not text.strip():
            return []
            
        logger.info(f"Extracting claims from text ({len(text)} chars)")
        
        # Simple claim extraction heuristics
        # In a production environment, this would use a more sophisticated NLP approach
        
        # Split by sentence endings
        # Split text into sentences using a simple regex pattern
        # This avoids the problematic look-behind pattern and simplifies the logic
        try:
            # Simple sentence splitter that handles common abbreviations and edge cases
            sentences = []
            # First split on obvious sentence boundaries
            raw_splits = re.split(r'([.!?])\s+(?=[A-Z])', text)
            
            i = 0
            current_sentence = ""
            while i < len(raw_splits):
                current_sentence += raw_splits[i]
                if i + 1 < len(raw_splits) and raw_splits[i+1] in ".!?":
                    current_sentence += raw_splits[i+1]
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                    i += 2
                else:
                    i += 1
                    
            # Add any remaining text as a sentence
            if current_sentence:
                sentences.append(current_sentence.strip())
                
        except Exception as e:
            logger.warning(f"Error splitting text into sentences: {str(e)}. Falling back to simpler method.")
            # Fallback to simpler method if regex fails
            sentences = [s.strip() for s in re.split(r'[.!?]+\s+', text) if s.strip()]
        # Filter potential factual statements
        claims = []
        for sentence in sentences:
            # Skip short sentences and questions
            if len(sentence) < 20 or sentence.endswith('?'):
                continue
                
            # Look for factual indicators
            factual_indicators = [
                r'\b(?:in|according to|research|study|evidence|data|results|analysis)\b',
                r'\b(?:show|prove|demonstrate|confirm|verify|suggest|indicate|reveal)\b',
                r'\b(?:percent|percentage|proportion|ratio|fraction|rate)\b',
                r'\b\d+(?:\.\d+)?%\b',
                r'\bin \d{4}\b',  # years
                r'\b(?:increase|decrease|reduce|expand|grow)\b',
            ]
            
            is_factual = any(re.search(pattern, sentence, re.IGNORECASE) for pattern in factual_indicators)
            
            if is_factual:
                claims.append(sentence)
                
            # Stop if we've reached the maximum number of claims
            if len(claims) >= max_claims:
                break
                
        logger.info(f"Extracted {len(claims)} claims from text")
        return claims
    
    def verify_text(self, text: str, source_types: Optional[List[SourceType]] = None) -> Dict[str, Any]:
        """Verify all claims in a text and provide verification results.
        
        Args:
            text: Text containing claims to verify
            source_types: List of source types to check (default: all)
            
        Returns:
            Dictionary with verification results
        """
        claims = self.extract_claims(text)
        
        verified_claims = 0
        total_citations = 0
        claim_results = []
        
        start_time = time.time()
        
        for claim in claims:
            is_verified, citations = self.verify_claim(claim, source_types)
            
            if is_verified:
                verified_claims += 1
                
            total_citations += len(citations)
            
            claim_results.append({
                "claim": claim,
                "is_verified": is_verified,
                "citations": [citation.to_dict() for citation in citations[:3]]  # Top 3 citations
            })
            
        elapsed_time = time.time() - start_time
        
        return {
            "total_claims": len(claims),
            "verified_claims": verified_claims,
            "verification_rate": verified_claims / len(claims) if claims else 0,
            "total_citations": total_citations,
            "processing_time_seconds": elapsed_time,
            "claim_results": claim_results,
        }
    
    def get_citation_format(self, citation: Citation, format_type: str = "apa") -> str:
        """Format a citation according to standard citation styles.
        
        Args:
            citation: Citation object to format
            format_type: Citation format style (apa, mla, etc.)
            
        Returns:
            Formatted citation string
        """
        if format_type.lower() == "apa":
            # APA format: Author(s) (Year). Title. Source. DOI
            author_text = ""
            if citation.authors:
                if len(citation.authors) == 1:
                    author_text = citation.authors[0]
                elif len(citation.authors) == 2:
                    author_text = f"{citation.authors[0]} & {citation.authors[1]}"
                else:
                    author_text = f"{citation.authors[0]} et al."
                    
            year_text = ""
            if citation.publication_date:
                year_text = f" ({citation.publication_date.year})"
                
            doi_text = ""
            if citation.doi:
                doi_text = f" https://doi.org/{citation.doi}"
                
            return f"{author_text}{year_text}. {citation.title}. {citation.source_type.value.capitalize()}{doi_text}"
            
        elif format_type.lower() == "mla":
            # MLA format: Author(s). "Title." Source, Publication Date.
            author_text = ""
            if citation.authors:
                if len(citation.authors) == 1:
                    author_text = citation.authors[0]
                elif len(citation.authors) == 2:
                    author_text = f"{citation.authors[0]} and {citation.authors[1]}"
                else:
                    author_text = f"{citation.authors[0]} et al."
                    
            date_text = ""
            if citation.publication_date:
                date_text = citation.publication_date.strftime("%d %b. %Y")
                
            return f"{author_text}. \"{citation.title}.\" {citation.source_type.value.capitalize()}, {date_text}."
            
        else:
            # Default simple format
            return str(citation)
