"""websearch MCP 서버 — mock: 패키지 내 고정 코퍼스 키워드 매칭 (스펙 §4), real: Tavily API.

config.yaml의 mock.websearch(또는 테스트용 COSCIENTIST_WEBSEARCH_MOCK 오버라이드)로 분기한다.
research 소비 배선은 실 LLM 도입 시(MockLLM은 검색 활용 불가).
"""
import os
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from ai_co_scientist.core.config import load_config, load_dotenv

mcp = FastMCP("websearch")
_CORPUS_DIR = Path(__file__).parent / "corpus"
_TAVILY_URL = "https://api.tavily.com/search"


def _is_mock() -> bool:
    override = os.environ.get("COSCIENTIST_WEBSEARCH_MOCK")
    if override is not None:
        return override == "1"
    return bool(load_config()["mock"]["websearch"])


def _mock_search(query: str, top_k: int) -> list[dict]:
    terms = set(query.lower().split())
    scored = []
    for path in sorted(_CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        score = sum(1 for t in terms if t in text.lower())
        if score > 0:
            scored.append((score, path.stem, text[:200]))
    scored.sort(reverse=True)
    return [{"title": title, "snippet": snippet} for _, title, snippet in scored[:top_k]]


def _real_search(query: str, top_k: int) -> list[dict]:
    load_dotenv()
    api_key = os.environ.get("WEBSEARCH_API_KEY")
    if not api_key:
        raise RuntimeError("WEBSEARCH_API_KEY 필요 (.env 참고)")
    resp = httpx.post(
        _TAVILY_URL,
        json={"api_key": api_key, "query": query, "max_results": top_k},
        timeout=30.0,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    return [{"title": r.get("title", ""), "snippet": r.get("content", "")}
            for r in results[:top_k]]


@mcp.tool()
def search(query: str, top_k: int = 3) -> list[dict]:
    """검색. mock: 코퍼스에서 단어 겹침 수로 랭킹해 상위 top_k 반환. real: Tavily API 호출."""
    if _is_mock():
        return _mock_search(query, top_k)
    return _real_search(query, top_k)


if __name__ == "__main__":
    mcp.run()
