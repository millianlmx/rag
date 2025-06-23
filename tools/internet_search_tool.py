import httpx
from typing import List, Dict, Any
from urllib.parse import urlencode, urlparse
from bs4 import BeautifulSoup
import re
import sys
import os

# Add parent directory to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from utils.model_caller import ModelCaller


class InternetSearchTool:
    """
    A tool for searching the internet and extracting information from web pages.
    Supports multiple search engines and content extraction.
    """
    
    def __init__(self):
        """
        Initialize the Internet Search tool.
        """
        self.model_caller = ModelCaller()
        self.session = None

    async def _get_http_session(self) -> httpx.AsyncClient:
        """Get or create an HTTP session"""
        if self.session is None:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            self.session = httpx.AsyncClient(timeout=30.0, headers=headers)
        return self.session
    
    async def fetch_search_results(self, query: str, num_results: int = 10) -> List[Dict[str, str]]:
        """
        Perform a web search using DuckDuckGo and return an array of {num_results} length.
        """
        try:
            session = await self._get_http_session()
            
            # First, get the search token
            search_url = "https://html.duckduckgo.com/html/"
            params = {'q': query}
            
            response = await session.get(search_url, params=params)
            response.raise_for_status()
            
            parser = BeautifulSoup(response.text, 'html.parser')
            results = []
            
            # Extract search results
            result_divs = parser.find_all('div', class_='result')
            
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
                    })
            
            return results
            
        except Exception as e:
            print(f"Error searching DuckDuckGo: {str(e)}")
            return []
    
    async def extract_webpage_text(self, url: str, max_chars: int = 5000) -> Dict[str, str]:
        try:
            session = await self._get_http_session()
            
            response = await session.get(url)
            response.raise_for_status()
            
            parser = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in parser(["script", "style"]):
                script.decompose()
            
            # Extract title
            title_elem = parser.find('title')
            title = title_elem.get_text(strip=True) if title_elem else "No title"
            
            # Extract main content
            # Try to find main content areas
            content_selectors = [
                'main', 'article', '.content', '#content', 
                '.post-content', '.article-content', 'body'
            ]
            
            content = ""
            for selector in content_selectors:
                content_elem = parser.select_one(selector)
                if content_elem:
                    content = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no specific content area found, get all text
            if not content:
                content = parser.get_text(separator=' ', strip=True)

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
    
    async def retrieve_relevant_web_content(self, 
                                query: str, 
                                num_results: int = 5,
                                num_extract: int = 3) -> Dict[str, Any]:
        # Perform search
        search_results = await self.fetch_search_results(query, num_results)
        
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
            content = await self.extract_webpage_text(result['url'])
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
    
    async def generate_answer_from_web(self, 
                                  query: str, 
                                  num_results: int = 5,
                                  num_extract: int = 3,
                                  custom_prompt: str = None) -> str:
        """
        Search the internet and provide a summarized answer using LLM.
        
        Args:
            query: Search query
            num_results: Number of search results to get
            num_extract: Number of pages to extract content from
            custom_prompt: Custom system prompt for the LLM
            
        Returns:
            Summarized answer from the LLM
        """
        # Get search results and extracted content
        data = await self.retrieve_relevant_web_content(query, num_results, num_extract)

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
                "Tu es un assistant expert qui analyse et synthétise l'information provenant de sources Internet. "
                "Ta tâche est de créer un condensé pertinent et structuré des informations les plus importantes "
                "trouvées dans les sources fournies. Concentre-toi sur les points clés, les faits essentiels "
                "et les informations les plus récentes ou significatives. "
                "Termine toujours ta réponse en citant clairement toutes les sources utilisées avec leurs URLs."
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
