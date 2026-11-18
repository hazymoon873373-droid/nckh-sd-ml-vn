#!/usr/bin/env python3
"""
scan_200_q1_papers.py — Scan 200+ strictly Q1 journal articles via Crossref REST API
across 6 thematic sub-areas for:
  "Stock Preference & Capital Flow Prediction in Vietnam via Stochastic Dominance and ML"

Outputs:
  - q1_200plus_corpus.csv  (deduplicated unique Q1 journal articles, target > 200)
  - q1_200plus_corpus.json (per-sub-area search structure for cross_search_aggregator.py)
"""

from __future__ import annotations

import csv
import html
import json
import pathlib
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from typing import Any

OUT_DIR = pathlib.Path(r"C:\Users\hazym\.gemini\antigravity\scratch\nckh-sd-ml-vn")
MAILTO = "hazymoon873373-droid@users.noreply.github.com"

# Strictly SJR Q1 journal ISSN whitelist (24 verified Q1 journals)
Q1_ISSNS: dict[str, str] = {
    "0025-1909": "Management Science",
    "0030-364X": "Operations Research",
    "0377-2217": "European Journal of Operational Research",
    "0022-1082": "The Journal of Finance",
    "0304-405X": "Journal of Financial Economics",
    "0893-9454": "The Review of Financial Studies",
    "0012-9682": "Econometrica",
    "0034-6527": "The Review of Economic Studies",
    "0304-4076": "Journal of Econometrics",
    "0735-0015": "Journal of Business & Economic Statistics",
    "0022-1090": "Journal of Financial and Quantitative Analysis",
    "0378-4266": "Journal of Banking & Finance",
    "1057-5219": "International Review of Financial Analysis",
    "0927-538X": "Pacific-Basin Finance Journal",
    "1544-6123": "Finance Research Letters",
    "0927-5398": "Journal of Empirical Finance",
    "1386-4181": "Journal of Financial Markets",
    "1469-7688": "Quantitative Finance",
    "0169-2070": "International Journal of Forecasting",
    "0957-4174": "Expert Systems with Applications",
    "0952-1976": "Engineering Applications of Artificial Intelligence",
    "1566-0141": "Emerging Markets Review",
    "1932-6203": "PLOS ONE",
    "2055-1045": "Humanities and Social Sciences Communications",
}

SUB_AREA_QUERIES = [
    {
        "sub_area": "SD_Theory_Tests",
        "query": "stochastic dominance portfolio asset pricing utility",
        "rows": 70,
        "required_terms": ["stochastic dominance", "dominance", "utility", "portfolio", "asset"],
    },
    {
        "sub_area": "Almost_SD",
        "query": "almost stochastic dominance second-order stochastic dominance portfolio",
        "rows": 70,
        "required_terms": ["stochastic dominance", "almost stochastic", "second-order", "portfolio", "dominance"],
    },
    {
        "sub_area": "ML_Distributional_CDF",
        "query": "quantile regression quantile neural return distribution stock return forecasting",
        "rows": 70,
        "required_terms": ["quantile", "distribution", "stock", "return", "forecast"],
    },
    {
        "sub_area": "ML_Stock_Portfolio",
        "query": "machine learning stock return prediction portfolio optimization",
        "rows": 70,
        "required_terms": ["stock", "return", "portfolio", "machine learning", "neural", "forecast", "asset"],
    },
    {
        "sub_area": "Investor_Preference_Flows",
        "query": "investor sentiment order flow foreign investors fund flow stock return",
        "rows": 70,
        "required_terms": ["flow", "sentiment", "investor", "stock", "return", "herding", "trading"],
    },
    {
        "sub_area": "Vietnam_Emerging_Markets",
        "query": "Vietnam emerging market stock return herding foreign ownership machine learning",
        "rows": 70,
        "required_terms": ["vietnam", "emerging", "stock", "market", "foreign", "herding", "return", "hose"],
    },
]

# Extra fallback queries (only used if unique count <= 210)
BACKUP_QUERIES = [
    {
        "sub_area": "SD_Theory_Tests",
        "query": "stochastic dominance efficiency stock market indexation",
        "rows": 50,
        "required_terms": ["stochastic dominance", "dominance", "portfolio", "stock"],
    },
    {
        "sub_area": "ML_Distributional_CDF",
        "query": "conformal prediction density forecasting stock returns quantile",
        "rows": 50,
        "required_terms": ["stock", "return", "forecast", "quantile", "density", "distribution"],
    },
    {
        "sub_area": "Investor_Preference_Flows",
        "query": "mutual fund flow capital flow institutional investors stock preference",
        "rows": 50,
        "required_terms": ["flow", "fund", "stock", "investor", "return"],
    },
]

FINANCE_ML_GUARD = [
    "stock", "equity", "equities", "portfolio", "asset", "return", "finance",
    "financial", "market", "investor", "trading", "fund", "stochastic dominance",
    "utility", "quantile", "risk", "capital", "vietnam", "forecast", "index",
]


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    txt = re.sub(r"<[^>]+>", " ", raw)
    txt = html.unescape(txt)
    return re.sub(r"\s+", " ", txt).strip()


def extract_year(item: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        part = item.get(key, {})
        dp = part.get("date-parts")
        if dp and dp[0] and dp[0][0]:
            return int(dp[0][0])
    return None


def match_whitelist_issn(item: dict[str, Any]) -> tuple[str | None, str | None]:
    for issn in item.get("ISSN", []):
        norm = issn.upper().strip()
        if norm in Q1_ISSNS:
            return norm, Q1_ISSNS[norm]
    return None, None


def is_relevant(title: str, abstract: str, required_terms: list[str]) -> bool:
    blob = f"{title} {abstract}".lower()
    if not any(g in blob for g in FINANCE_ML_GUARD):
        return False
    return any(t in blob for t in required_terms)


def query_crossref(query: str, rows: int) -> list[dict[str, Any]]:
    issn_filters = [f"issn:{issn}" for issn in Q1_ISSNS]
    filter_str = ",".join(issn_filters) + ",type:journal-article"
    params = {
        "filter": filter_str,
        "query": query,
        "rows": str(rows),
        "sort": "relevance",
        "select": "DOI,title,author,published-print,published-online,issued,container-title,ISSN,is-referenced-by-count,abstract,URL",
        "mailto": MAILTO,
    }
    url = "https://api.crossref.org/works?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": f"NCKH-Q1-Scanner/1.0 (mailto:{MAILTO})"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("message", {}).get("items", [])


def run_scan() -> None:
    searches_for_agg: list[dict[str, Any]] = []
    unique_by_doi: dict[str, dict[str, Any]] = {}
    search_logs: list[dict[str, Any]] = []

    def process_batch(spec: dict[str, Any]) -> int:
        sub_area = spec["sub_area"]
        q = spec["query"]
        items = query_crossref(q, spec["rows"])
        papers_in_search: list[dict[str, Any]] = []
        added_unique = 0

        for it in items:
            doi = (it.get("DOI") or "").strip().lower()
            if not doi:
                continue
            issn, canonical_journal = match_whitelist_issn(it)
            if not issn:
                continue
            title = clean_text((it.get("title") or [""])[0])
            if not title:
                continue
            abstract = clean_text(it.get("abstract") or "")
            if not is_relevant(title, abstract, spec["required_terms"]):
                continue

            authors = []
            for a in it.get("author", []):
                fam = (a.get("family") or "").strip()
                giv = (a.get("given") or "").strip()
                if fam and giv:
                    authors.append(f"{fam}, {giv}")
                elif fam:
                    authors.append(fam)
            year = extract_year(it)
            citations = int(it.get("is-referenced-by-count") or 0)
            url = f"https://doi.org/{doi}"
            container = clean_text((it.get("container-title") or [canonical_journal])[0]) or canonical_journal

            papers_in_search.append(
                {
                    "url": url,
                    "title": title,
                    "authors": authors,
                    "year": year,
                    "citations": citations,
                }
            )

            if doi not in unique_by_doi:
                unique_by_doi[doi] = {
                    "doi": doi,
                    "year": year,
                    "authors": "; ".join(authors),
                    "title": title,
                    "journal": canonical_journal or container,
                    "issn": issn,
                    "sjr_quartile": "Q1",
                    "sub_area": sub_area,
                    "citations": citations,
                    "url": url,
                    "abstract_clean": abstract[:700],
                }
                added_unique += 1

        searches_for_agg.append(
            {
                "query": q,
                "sub_area": sub_area,
                "papers": papers_in_search,
            }
        )
        search_logs.append(
            {
                "sub_area": sub_area,
                "query": q,
                "raw_returned": len(items),
                "relevant_q1_kept": len(papers_in_search),
                "new_unique_added": added_unique,
            }
        )
        print(
            f"[{sub_area}] raw={len(items)} | relevant_Q1={len(papers_in_search)} "
            f"| new_unique={added_unique} | cumulative_unique={len(unique_by_doi)}"
        )
        return len(unique_by_doi)

    for spec in SUB_AREA_QUERIES:
        process_batch(spec)
        time.sleep(1.0)

    if len(unique_by_doi) <= 210:
        print("Running backup Q1 queries to ensure >200 comfortably...")
        for spec in BACKUP_QUERIES:
            process_batch(spec)
            time.sleep(1.0)
            if len(unique_by_doi) >= 245:
                break

    # Sort unique papers deterministically by citations desc, then year desc
    rows_sorted = sorted(
        unique_by_doi.values(),
        key=lambda r: (-(r["citations"] or 0), -(r["year"] or 0), r["doi"]),
    )

    csv_path = OUT_DIR / "q1_200plus_corpus.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "id",
            "doi",
            "year",
            "authors",
            "title",
            "journal",
            "issn",
            "sjr_quartile",
            "sub_area",
            "citations",
            "url",
            "abstract_clean",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows_sorted, 1):
            row_out = {"id": f"Q1-{idx:03d}", **row}
            writer.writerow(row_out)

    json_payload = {
        "session": "sd-ml-vn-q1-scan200",
        "search_logs": search_logs,
        "searches": searches_for_agg,
    }
    json_path = OUT_DIR / "q1_200plus_corpus.json"
    json_path.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary statistics
    by_sub = Counter(r["sub_area"] for r in rows_sorted)
    by_journal = Counter(r["journal"] for r in rows_sorted)
    by_era = Counter()
    for r in rows_sorted:
        y = r["year"]
        if not y:
            by_era["Unknown"] += 1
        elif y <= 2005:
            by_era["<=2005"] += 1
        elif y <= 2015:
            by_era["2006-2015"] += 1
        elif y <= 2020:
            by_era["2016-2020"] += 1
        else:
            by_era["2021-2026"] += 1

    print("\n=== SUMMARY (>200 STRICT Q1 SCAN) ===")
    print(f"Total unique Q1 papers: {len(rows_sorted)}")
    print("By sub_area:", dict(by_sub))
    print("By era:", dict(by_era))
    print("Top journals:")
    for j, c in by_journal.most_common():
        print(f"  {j}: {c}")
    print("\nTop 10 cited Q1 papers in corpus:")
    for r in rows_sorted[:10]:
        print(f"  [{r['citations']} cites] ({r['year']}) {r['title']} — {r['journal']} | {r['url']}")


if __name__ == "__main__":
    run_scan()
