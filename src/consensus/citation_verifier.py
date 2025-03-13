import re
import logging
import aiohttp
from typing import List, Dict, Optional, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class CitationVerifier:
    """Verifies citations in text by checking URLs and extracting content."""

    @staticmethod
    async def extract_citations(text: str) -> List[Dict[str, str]]:
        """Extract citations from text."""
        citations = []
        # Match citation patterns like [1] https://...
        pattern = r'\[(\d+)\]\s*(.*?)(?=\[\d+\]|\s*$)'
        matches = re.finditer(pattern, text, re.DOTALL)

        for match in matches:
            citation_num, citation_text = match.groups()
            # Extract URL from citation text
            url_match = re.search(r'https?://\S+', citation_text)
            if url_match:
                citations.append({
                    "number": citation_num,
                    "text": citation_text.strip(),
                    "url": url_match.group(0)
                })

        return citations

    @staticmethod
    async def verify_url(url: str) -> Tuple[bool, Optional[str]]:
        """Verify URL accessibility and get content."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        content = await response.text()
                        # Basic content extraction
                        soup = BeautifulSoup(content, 'html.parser')
                        # Remove script and style elements
                        for script in soup(["script", "style"]):
                            script.decompose()
                        text = soup.get_text()
                        # Normalize whitespace
                        text = " ".join(text.split())
                        return True, text
                    return False, None
        except Exception as e:
            logger.error(f"Error verifying URL {url}: {e}")
            return False, None

    @staticmethod
    async def verify_citations(text: str) -> Dict[str, any]:
        """Verify all citations in the text."""
        try:
            citations = await CitationVerifier.extract_citations(text)

            if not citations:
                return {
                    "is_verified": False,
                    "message": "No citations found in text"
                }

            verified_count = 0
            verification_results = []

            for citation in citations:
                url = citation.get("url")
                if url:
                    is_valid, content = await CitationVerifier.verify_url(url)
                    if is_valid:
                        verified_count += 1
                    verification_results.append({
                        "citation_number": citation["number"],
                        "is_valid": is_valid,
                        "url": url
                    })

            return {
                "is_verified": verified_count > 0,
                "total_citations": len(citations),
                "verified_citations": verified_count,
                "results": verification_results
            }

        except Exception as e:
            logger.error(f"Error in citation verification: {e}")
            return {
                "is_verified": False,
                "message": f"Error during verification: {str(e)}"
            }
