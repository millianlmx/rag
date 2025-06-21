"""
Internet Search Tool - A tool for searching the web and retrieving information
This tool provides web search capabilities and can extract content from web pages.
"""

import asyncio
import httpx
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urlencode, urlparse
from bs4 import BeautifulSoup
import re
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.llama_cpp_call import ModelCaller


class InternetSearchTool:
    """
    A tool for searching the internet and extracting information from web pages.
    Supports multiple search engines and content extraction.
    """
    
    def __init__(self, 
                 llm_url: str = "http://127.0.0.1:8080/v1",
                 embedding_model: str = "Lajavaness/sentence-camembert-large",
                 default_search_engine: str = "duckduckgo"):
        """
        Initialize the Internet Search tool.
        
        Args:
            llm_url: URL for the LLM API endpoint
            embedding_model: Name of the sentence transformer model for embeddings
            default_search_engine: Default search engine to use
        """
        self.model_caller = ModelCaller(llm_url=llm_url, embedding_model=embedding_model)
        self.default_search_engine = default_search_engine
        self.session = None
        
    async def _get_session(self) -> httpx.AsyncClient:
        """Get or create an HTTP session"""
        if self.session is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            self.session = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self.session
    
    async def search_duckduckgo(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Search using DuckDuckGo search engine.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries containing title, url, and snippet
        """
        try:
            session = await self._get_session()
            
            # First, get the search token
            search_url = "https://html.duckduckgo.com/html/"
            params = {'q': query}
            
            response = await session.get(search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            result_divs = soup.find_all('div', class_='result')
            
            for i, div in enumerate(result_divs[:num_results]):
                title_elem = div.find('a', class_='result__a')
                snippet_elem = div.find('a', class_='result__snippet')
                
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    url = title_elem.get('href', '')
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    
                    results.append({
                        'title': title,
                        'url': url,
                        'snippet': snippet,
                        'search_engine': 'duckduckgo'
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching DuckDuckGo: {str(e)}")
            return []
    
    async def search_bing(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Search using Bing search engine (requires Bing Search API key).
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of dictionaries containing title, url, and snippet
        """
        # Note: This requires a Bing Search API key
        # For now, we'll implement a basic web scraping version
        try:
            session = await self._get_session()
            
            search_url = "https://www.bing.com/search"
            params = {'q': query, 'count': num_results}
            
            response = await session.get(search_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results from Bing
            result_divs = soup.find_all('li', class_='b_algo')
            
            for div in result_divs[:num_results]:
                title_elem = div.find('h2')
                if title_elem:
                    link_elem = title_elem.find('a')
                    if link_elem:
                        title = link_elem.get_text(strip=True)
                        url = link_elem.get('href', '')
                        
                        # Find snippet
                        snippet_elem = div.find('p') or div.find('div', class_='b_caption')
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        results.append({
                            'title': title,
                            'url': url,
                            'snippet': snippet,
                            'search_engine': 'bing'
                        })
            
            return results
            
        except Exception as e:
            print(f"Error searching Bing: {str(e)}")
            return []
    
    async def search(self, 
                    query: str, 
                    num_results: int = 10, 
                    search_engine: str = None) -> List[Dict[str, str]]:
        """
        Search the internet using the specified search engine.
        
        Args:
            query: Search query
            num_results: Number of results to return
            search_engine: Search engine to use ('duckduckgo', 'bing')
            
        Returns:
            List of search results
        """
        engine = search_engine or self.default_search_engine
        
        if engine.lower() == 'duckduckgo':
            return await self.search_duckduckgo(query, num_results)
        elif engine.lower() == 'bing':
            return await self.search_bing(query, num_results)
        else:
            print(f"Unsupported search engine: {engine}")
            return await self.search_duckduckgo(query, num_results)
    
    async def extract_content(self, url: str, max_chars: int = 5000) -> Dict[str, str]:
        """
        Extract content from a web page.
        
        Args:
            url: URL to extract content from
            max_chars: Maximum number of characters to extract
            
        Returns:
            Dictionary containing title, content, and url
        """
        try:
            session = await self._get_session()
            
            response = await session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Extract title
            title_elem = soup.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "No title"
            
            # Extract main content
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post-content', '.article-content', 'body'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no specific content area found, get all text
            if not content:
                content = soup.get_text(separator=' ', strip=True)
            
            # Clean up content
            content = re.sub(r'\s+', ' ', content)
            content = content[:max_chars]
            
            return {
                'title': title,
                'content': content,
                'url': url,
                'length': len(content)
            }
            
        except Exception as e:
            print(f"Error extracting content from {url}: {str(e)}")
            return {
                'title': "Error",
                'content': f"Could not extract content: {str(e)}",
                'url': url,
                'length': 0
            }
    
    async def search_and_extract(self, 
                                query: str, 
                                num_results: int = 5,
                                num_extract: int = 3,
                                search_engine: str = None) -> Dict[str, Any]:
        """
        Search the internet and extract content from top results.
        
        Args:
            query: Search query
            num_results: Number of search results to get
            num_extract: Number of pages to extract content from
            search_engine: Search engine to use
            
        Returns:
            Dictionary containing search results and extracted content
        """
        # Perform search
        search_results = await self.search(query, num_results, search_engine)
        
        if not search_results:
            return {
                'query': query,
                'search_results': [],
                'extracted_content': [],
                'error': 'No search results found'
            }
        
        # Extract content from top results
        extracted_content = []
        for i, result in enumerate(search_results[:num_extract]):
            content = await self.extract_content(result['url'])
            content['search_rank'] = i + 1
            content['search_snippet'] = result['snippet']
            extracted_content.append(content)
        
        return {
            'query': query,
            'search_results': search_results,
            'extracted_content': extracted_content,
            'total_results': len(search_results),
            'total_extracted': len(extracted_content)
        }
    
    async def search_and_summarize(self, 
                                  query: str, 
                                  num_results: int = 5,
                                  num_extract: int = 3,
                                  search_engine: str = None,
                                  custom_prompt: str = None) -> str:
        """
        Search the internet and provide a summarized answer using LLM.
        
        Args:
            query: Search query
            num_results: Number of search results to get
            num_extract: Number of pages to extract content from
            search_engine: Search engine to use
            custom_prompt: Custom system prompt for the LLM
            
        Returns:
            Summarized answer from the LLM
        """
        # Get search results and extracted content
        data = await self.search_and_extract(query, num_results, num_extract, search_engine)
        
        if 'error' in data:
            return f"Error: {data['error']}"
        
        # Prepare context from extracted content
        context_parts = []
        for i, content in enumerate(data['extracted_content']):
            context_parts.append(f"""
Source {i+1}: {content['title']}
URL: {content['url']}
Content: {content['content'][:2000]}...
""")
        
        context = "\n".join(context_parts)
        
        # Use custom or default system prompt
        if custom_prompt is None:
            custom_prompt = (
                "Tu es un assistant utile qui répond aux questions en utilisant "
                "les informations trouvées sur Internet. Fournis une réponse complète "
                "et précise basée sur les sources fournies. Cite les sources quand possible."
            )
        
        # Generate summary using LLM
        response = await self.model_caller.chat(
            messages=[
                {"role": "system", "content": custom_prompt},
                {"role": "user", "content": f"Sources Internet:\n{context}\n\nQuestion: {query}"}
            ]
        )
        
        return response
    
    async def close(self):
        """Close the HTTP session"""
        if self.session:
            await self.session.aclose()
            self.session = None


# Example usage and testing functions
async def test_internet_search_tool():
    """Test function for the Internet Search tool"""
    search_tool = InternetSearchTool()
    
    try:
        # Test basic search
        query = "artificial intelligence latest news"
        print(f"Searching for: {query}")
        
        results = await search_tool.search(query, num_results=5)
        print(f"Found {len(results)} results")
        
        for i, result in enumerate(results):
            print(f"{i+1}. {result['title']}")
            print(f"   URL: {result['url']}")
            print(f"   Snippet: {result['snippet'][:100]}...")
            print()
        
        # Test content extraction
        if results:
            print("Extracting content from first result...")
            content = await search_tool.extract_content(results[0]['url'])
            print(f"Title: {content['title']}")
            print(f"Content length: {content['length']} characters")
            print(f"Content preview: {content['content'][:200]}...")
            print()
        
        # Test search and summarize
        print("Getting summarized answer...")
        summary = await search_tool.search_and_summarize(
            "What are the latest developments in artificial intelligence?",
            num_results=3,
            num_extract=2
        )
        print(f"Summary: {summary}")
        
    finally:
        await search_tool.close()


if __name__ == "__main__":
    # Run test
    asyncio.run(test_internet_search_tool())
