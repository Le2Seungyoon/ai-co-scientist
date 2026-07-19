"""websearch mock MCP 서버 — 패키지 내 고정 코퍼스 키워드 매칭 (스펙 §4).

real 백엔드는 골격만. research 소비 배선은 실 LLM 도입 시(MockLLM은 검색 활용 불가).
"""
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("websearch")
_CORPUS_DIR = Path(__file__).parent / "corpus"


def _real_search(query: str, top_k: int) -> list[dict]:
    # TODO(M5+): 실 검색 API 연동 (예: Tavily/SerpAPI)
    raise NotImplementedError("real websearch 백엔드는 골격만")


@mcp.tool()
def search(query: str, top_k: int = 3) -> list[dict]:
    """코퍼스에서 단어 겹침 수로 랭킹해 상위 top_k 반환."""
    terms = set(query.lower().split())
    scored = []
    for path in sorted(_CORPUS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        score = sum(1 for t in terms if t in text.lower())
        if score > 0:
            scored.append((score, path.stem, text[:200]))
    scored.sort(reverse=True)
    return [{"title": title, "snippet": snippet} for _, title, snippet in scored[:top_k]]


if __name__ == "__main__":
    mcp.run()
