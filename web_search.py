import os, requests, logging
from urllib.parse import urlencode
from config.config import SERPAPI_API_KEY

logger = logging.getLogger(__name__)

def serpapi_search(query, num=3):
    # requires SERPAPI_API_KEY
    if not SERPAPI_API_KEY:
        raise ValueError('SERPAPI_API_KEY not set in environment/config.')
    params = {
        'engine': 'google',
        'q': query,
        'api_key': SERPAPI_API_KEY,
        'num': num
    }
    url = 'https://serpapi.com/search?' + urlencode(params)
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get('organic_results', [])[:num]:
            results.append({'title': item.get('title'), 'link': item.get('link'), 'snippet': item.get('snippet')})
        return results
    except Exception as e:
        logger.warning(f'SerpAPI search failed: {e}')
        return []

def duckduckgo_search(query, num=3):
    # lightweight fallback using duckduckgo html scraping via 'html.duckduckgo.com'
    try:
        params = {'q': query}
        r = requests.get('https://html.duckduckgo.com/html/', data=params, timeout=10)
        r.raise_for_status()
        text = r.text
        # naive parsing: extract <a class="result__a" href="..."> title
        results = []
        import re
        anchors = re.findall(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>', text, flags=re.S)
        for href, title_html in anchors[:num]:
            title = re.sub(r'<.*?>', '', title_html).strip()
            results.append({'title': title, 'link': href, 'snippet': ''})
        return results
    except Exception as e:
        logger.warning(f'DuckDuckGo search failed: {e}')
        return []
