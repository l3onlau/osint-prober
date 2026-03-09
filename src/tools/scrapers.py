"""
Web scraping and search tool factories.

Returns LangChain-compatible tool instances. Optional tools are only registered
when the corresponding API key is set in config.

Sources:
  - DuckDuckGo    — always active, no key needed
  - Wikipedia     — always active, no key needed
  - Wayback CDX   — always active, no key needed (Internet Archive public CDX API)
  - NewsAPI       — active only if NEWSAPI_KEY is set in .env
  - HIBP          — active only if HIBP_API_KEY is set in .env
  - Firecrawl     — active only if FIRECRAWL_API_KEY is set in .env
"""

import requests
from src.logger import log_step

from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.tools.wikipedia.tool import WikipediaQueryRun
from langchain_community.utilities.wikipedia import WikipediaAPIWrapper
from langchain_core.tools import tool

from src.config import config


def get_duckduckgo_tool(max_results: int = 5):
    """Returns a DuckDuckGo search tool (snippet + title + link)."""
    return DuckDuckGoSearchResults(num_results=max_results)


def get_wikipedia_tool():
    """Returns a Wikipedia search tool (top-1 result, max 1500 chars)."""
    api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=1500)
    return WikipediaQueryRun(api_wrapper=api_wrapper)


# ---------------------------------------------------------------------------
# Wayback Machine CDX API  (always active — no API key required)
# ---------------------------------------------------------------------------


def get_wayback_tool():
    """Returns a Wayback Machine snapshot lookup tool (Internet Archive CDX API).

    Free and key-less. Returns a summary of archived snapshots for a URL,
    useful for retrieving deleted or edited web pages about a target.
    """

    @tool
    def lookup_wayback(url_or_query: str) -> str:
        """Looks up archived snapshots of a URL in the Wayback Machine (Internet Archive).
        Use this when a URL is unavailable, deleted, or you want to check historical page content.
        Input can be a full URL or a domain name.
        """
        try:
            # Use CDX API to find available snapshots
            cdx_url = (
                f"http://web.archive.org/cdx/search/cdx"
                f"?url={url_or_query}&output=json&limit=3&fl=timestamp,original,statuscode&filter=statuscode:200"
            )
            resp = requests.get(cdx_url, timeout=10)
            resp.raise_for_status()
            rows = resp.json()

            if not rows or len(rows) <= 1:  # First row is header
                log_step(
                    "Gatherer",
                    f"Wayback search: '{url_or_query}' → 0 results",
                    level="search",
                )
                return f"No Wayback Machine snapshots found for: {url_or_query}"

            results = []
            for row in rows[1:4]:  # Skip header row
                ts, orig_url, status = row
                # Format timestamp: 20201231120000 → 2020-12-31
                date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ts
                wayback_link = f"https://web.archive.org/web/{ts}/{orig_url}"
                results.append(f"[{date_str}] {wayback_link}")

            output = "Wayback Machine snapshots:\n" + "\n".join(results)
            log_step(
                "Gatherer",
                f"Wayback search: '{url_or_query}' → {len(rows) - 1} snapshots found",
                level="search",
            )
            return output
        except Exception as e:
            log_step("Gatherer", f"Wayback search failed: {e}", level="warning")
            return f"Wayback lookup error: {e}"

    return lookup_wayback


# ---------------------------------------------------------------------------
# NewsAPI  (optional — requires NEWSAPI_KEY in .env)
# ---------------------------------------------------------------------------


def get_newsapi_tool(max_results: int = 5):
    """Returns a NewsAPI search tool. Returns None if NEWSAPI_KEY is not set."""
    if not config.newsapi_key:
        return None

    @tool
    def search_news(query: str) -> str:
        """Searches NewsAPI for recent news articles about a person or organization.
        Use this to find recent events, criminal charges, and news coverage.
        Returns article titles, sources, and a snippet for each result.
        """
        try:
            resp = requests.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "apiKey": config.newsapi_key,
                    "pageSize": max_results,
                    "sortBy": "relevancy",
                    "language": "en",
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            articles = data.get("articles", [])
            if not articles:
                log_step(
                    "Gatherer",
                    f"NewsAPI search: '{query}' → 0 articles",
                    level="search",
                )
                return f"No news articles found for: {query}"

            results = []
            for art in articles[:max_results]:
                title = art.get("title", "No title")
                source = art.get("source", {}).get("name", "Unknown")
                desc = art.get("description") or art.get("content") or ""
                url = art.get("url", "")
                results.append(f"**{title}** [{source}]\n{desc[:300]}\n{url}")

            output = "\n\n".join(results)
            log_step(
                "Gatherer",
                f"NewsAPI search: '{query}' → {len(articles[:max_results])} articles found",
                level="search",
            )
            return output
        except Exception as e:
            log_step("Gatherer", f"NewsAPI search failed: {e}", level="warning")
            return f"NewsAPI error: {e}"

    return search_news


# ---------------------------------------------------------------------------
# Have I Been Pwned  (optional — requires HIBP_API_KEY in .env)
# ---------------------------------------------------------------------------


def get_hibp_tool():
    """Returns a Have I Been Pwned breach lookup tool. Returns None if HIBP_API_KEY not set."""
    if not config.hibp_api_key:
        return None

    @tool
    def check_breach_records(email_or_username: str) -> str:
        """Checks Have I Been Pwned (HIBP) for data breaches associated with an email or username.
        Use this to find if a target's digital accounts appear in known data breaches.
        Input should be an email address or a username.
        """
        try:
            # HIBP v3 requires a paid API key for /breachedaccount (per-account lookup)
            resp = requests.get(
                f"https://haveibeenpwned.com/api/v3/breachedaccount/{email_or_username}",
                headers={
                    "hibp-api-key": config.hibp_api_key,
                    "User-Agent": "OSINT_Prober/1.0",
                },
                params={"truncateResponse": "false"},
                timeout=10,
            )
            if resp.status_code == 404:
                log_step(
                    "Gatherer",
                    f"HIBP lookup for '{email_or_username}': No breaches found (404).",
                    level="search",
                )
                return f"No breaches found for: {email_or_username}"
            resp.raise_for_status()
            breaches = resp.json()
            if not breaches:
                log_step(
                    "Gatherer",
                    f"HIBP lookup for '{email_or_username}': 0 breaches.",
                    level="search",
                )
                return f"No data breaches found for: {email_or_username}"

            summary = []
            for b in breaches[:5]:
                name = b.get("Name", "Unknown")
                date = b.get("BreachDate", "Unknown date")
                count = b.get("PwnCount", 0)
                data_classes = ", ".join(b.get("DataClasses", [])[:5])
                summary.append(
                    f"- **{name}** ({date}): {count:,} accounts. Data: {data_classes}"
                )
            output = f"Data breaches for {email_or_username}:\n" + "\n".join(summary)
            log_step(
                "Gatherer",
                f"HIBP lookup for '{email_or_username}': {len(breaches)} breaches found.",
                level="search",
            )
            return output
        except Exception as e:
            log_step("Gatherer", f"HIBP lookup failed: {e}", level="warning")
            return f"HIBP lookup error: {e}"

    return check_breach_records


# ---------------------------------------------------------------------------
# Firecrawl Scraper (optional — requires FIRECRAWL_API_KEY in .env)
# ---------------------------------------------------------------------------


def get_firecrawl_tool(scrape_content_chars_max: int = 3000):
    """Returns a Deep web scraper using Firecrawl. Returns None if FIRECRAWL_API_KEY not set."""
    if not config.firecrawl_api_key:
        return None

    @tool
    def firecrawl_scrape(url: str) -> str:
        """Scrapes the full text content of a web page URL using Firecrawl API.
        This scraper handles Javascript-heavy domains, Single Page Applications, bypassing anti-bot measures, and extracting clean markdown.
        Use this to get detailed information from specific URLs found in search results.
        """
        try:
            headers = {
                "Authorization": f"Bearer {config.firecrawl_api_key}",
                "Content-Type": "application/json",
            }
            json_data = {"url": url, "formats": ["markdown"]}

            resp = requests.post(
                "https://api.firecrawl.dev/v1/scrape",
                headers=headers,
                json=json_data,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()

            if data.get("success"):
                markdown = data.get("data", {}).get("markdown", "")
                truncated = markdown[:scrape_content_chars_max] if markdown else ""
                log_step(
                    "Gatherer",
                    f"Firecrawl Scraped `{url[:60]}…` → {len(truncated)} chars",
                    level="extract",
                )
                return truncated or "Could not scrape page content."
            else:
                return f"Firecrawl scraping error: {data.get('error', 'Unknown error')}"
        except Exception as e:
            return f"Firecrawl request error: {e}"

    return firecrawl_scrape
