"""Simulated PubMed search tool for literature retrieval.

Provides a deterministic, seeded simulation of the PubMed/MEDLINE database so
that agents can demonstrate literature review capabilities without requiring
network access.
"""

from __future__ import annotations

import hashlib
import random
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Simulated paper corpus
# ---------------------------------------------------------------------------

_JOURNALS = [
    "The New England Journal of Medicine",
    "The Lancet",
    "JAMA",
    "Nature Medicine",
    "Nature Reviews Drug Discovery",
    "Journal of Clinical Oncology",
    "The Lancet Oncology",
    "British Medical Journal",
    "Annals of Internal Medicine",
    "Clinical Pharmacology & Therapeutics",
    "Drug Safety",
    "Journal of Medicinal Chemistry",
    "Molecular Pharmacology",
    "Pharmacological Reviews",
    "European Journal of Pharmacology",
]

_FIRST_NAMES = [
    "James",
    "Maria",
    "Wei",
    "Priya",
    "Ahmed",
    "Elena",
    "Hiroshi",
    "Olga",
    "Carlos",
    "Fatima",
    "Liam",
    "Yuki",
    "Aisha",
    "Marco",
]
_LAST_NAMES = [
    "Smith",
    "Chen",
    "Patel",
    "Mueller",
    "Kim",
    "Santos",
    "Nakamura",
    "Ivanova",
    "Rodriguez",
    "Al-Rashid",
    "O'Brien",
    "Tanaka",
    "Johansson",
    "Liu",
]

_STUDY_TYPES = [
    "Randomized Controlled Trial",
    "Phase III Clinical Trial",
    "Phase II Clinical Trial",
    "Meta-Analysis",
    "Systematic Review",
    "Cohort Study",
    "Case-Control Study",
    "Cross-Sectional Study",
    "Case Report",
    "In Vitro Study",
    "Animal Study",
    "Review Article",
]

_EVIDENCE_LEVELS = {
    "Randomized Controlled Trial": "1b",
    "Phase III Clinical Trial": "1b",
    "Phase II Clinical Trial": "2b",
    "Meta-Analysis": "1a",
    "Systematic Review": "1a",
    "Cohort Study": "2b",
    "Case-Control Study": "3b",
    "Cross-Sectional Study": "4",
    "Case Report": "4",
    "In Vitro Study": "5",
    "Animal Study": "5",
    "Review Article": "5",
}


def _seed_from_string(s: str) -> int:
    """Deterministic seed from a string."""
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def _generate_authors(rng: random.Random, count: int = 4) -> list[str]:
    authors = []
    for _ in range(count):
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        authors.append(f"{last} {first[0]}")
    return authors


def _generate_paper(query: str, index: int, rng: random.Random) -> dict[str, Any]:
    """Generate a single simulated paper based on query context."""
    study_type = rng.choice(_STUDY_TYPES)
    journal = rng.choice(_JOURNALS)
    year = rng.randint(2015, 2025)
    n_patients = rng.randint(50, 5000) if "Trial" in study_type else rng.randint(10, 500)
    pmid = str(rng.randint(30000000, 39999999))

    # Build a plausible title
    title_templates = [
        f"Efficacy and Safety of {query} in Patients: A {study_type}",
        f"{study_type} Evaluating {query} for Treatment Outcomes",
        f"Long-term Outcomes of {query} Therapy: {study_type}",
        f"Comparative Analysis of {query} vs Standard of Care",
        f"Pharmacokinetics and Pharmacodynamics of {query}: Phase I Results",
        f"Molecular Mechanisms Underlying {query} Activity",
        f"Real-World Evidence for {query}: A Multi-Center {study_type}",
        f"Biomarker-Guided {query} Therapy: Results from a {study_type}",
    ]
    title = rng.choice(title_templates)

    abstract_sentences = [
        f"BACKGROUND: {query} has emerged as a promising therapeutic approach.",
        f"METHODS: We conducted a {study_type.lower()} enrolling {n_patients} patients.",
        f"RESULTS: The primary endpoint was met with statistical significance (p<0.{rng.randint(1, 5):02d}).",
        f"Median follow-up was {rng.randint(6, 36)} months.",
        f"The overall response rate was {rng.randint(20, 80)}%.",
        f"Grade 3-4 adverse events occurred in {rng.randint(5, 35)}% of patients.",
        f"CONCLUSIONS: {query} demonstrates {'favorable' if rng.random() > 0.3 else 'mixed'} "
        f"efficacy and an {'acceptable' if rng.random() > 0.2 else 'concerning'} safety profile.",
    ]

    return {
        "pmid": pmid,
        "title": title,
        "authors": _generate_authors(rng),
        "journal": journal,
        "year": year,
        "study_type": study_type,
        "evidence_level": _EVIDENCE_LEVELS.get(study_type, "5"),
        "abstract": " ".join(abstract_sentences),
        "citation_count": rng.randint(0, 500),
        "n_patients": n_patients,
        "key_findings": [
            f"Primary endpoint achieved (p<0.{rng.randint(1, 5):02d})",
            f"ORR: {rng.randint(20, 80)}% vs {rng.randint(10, 50)}% control",
            f"Median PFS: {rng.uniform(3, 24):.1f} months",
            f"Most common AE: {rng.choice(['fatigue', 'nausea', 'rash', 'diarrhea'])} "
            f"({rng.randint(10, 60)}%)",
        ],
    }


class PubMedTool:
    """Simulated PubMed search and paper retrieval tool.

    All results are deterministically generated based on query hashing so
    that the same query always returns the same papers.
    """

    def search(self, query: str, max_results: int = 10) -> dict[str, Any]:
        """Search for papers matching a query string."""
        seed = _seed_from_string(query)
        rng = random.Random(seed)
        count = min(max_results, 20)

        papers = [_generate_paper(query, i, rng) for i in range(count)]
        papers.sort(key=lambda p: p["citation_count"], reverse=True)

        logger.info("pubmed.search", query=query, results=len(papers))
        return {
            "query": query,
            "total_results": rng.randint(count, count * 50),
            "returned": len(papers),
            "papers": papers,
        }

    def get_paper_details(self, pmid: str) -> dict[str, Any]:
        """Get detailed information about a single paper by PMID."""
        seed = _seed_from_string(pmid)
        rng = random.Random(seed)
        paper = _generate_paper(f"PMID-{pmid}", 0, rng)
        paper["pmid"] = pmid

        # Enrich with additional detail
        paper["methods_summary"] = (
            f"This was a {paper['study_type'].lower()} conducted across "
            f"{rng.randint(5, 80)} sites in {rng.randint(3, 15)} countries. "
            f"Patients were randomised {rng.choice(['1:1', '2:1', '3:1'])} to "
            f"treatment vs. control. The primary endpoint was "
            f"{rng.choice(['overall survival', 'progression-free survival', 'overall response rate'])}."
        )
        paper["conclusions"] = (
            f"The study {'met' if rng.random() > 0.3 else 'did not meet'} its primary endpoint. "
            f"Further investigation in a larger population is "
            f"{'warranted' if rng.random() > 0.4 else 'recommended'}."
        )
        paper["references_count"] = rng.randint(20, 80)

        logger.info("pubmed.get_details", pmid=pmid)
        return paper

    def cross_reference(self, pmid: str, direction: str = "cited_by") -> dict[str, Any]:
        """Find papers citing or referenced by a given PMID."""
        seed = _seed_from_string(f"{pmid}:{direction}")
        rng = random.Random(seed)
        count = rng.randint(3, 12)

        related = [_generate_paper(f"ref-{pmid}-{i}", i, rng) for i in range(count)]

        logger.info("pubmed.cross_reference", pmid=pmid, direction=direction, count=count)
        return {
            "source_pmid": pmid,
            "direction": direction,
            "count": count,
            "related_papers": related,
        }

    def summarize_evidence(self, pmids: list[str], focus: str) -> dict[str, Any]:
        """Produce an evidence summary across multiple papers."""
        papers = [self.get_paper_details(pmid) for pmid in pmids]

        # Aggregate evidence levels
        level_counts: dict[str, int] = {}
        total_patients = 0
        for p in papers:
            level = p.get("evidence_level", "5")
            level_counts[level] = level_counts.get(level, 0) + 1
            total_patients += p.get("n_patients", 0)

        # Determine overall evidence quality
        high_evidence = level_counts.get("1a", 0) + level_counts.get("1b", 0)
        overall_quality = (
            "HIGH"
            if high_evidence >= len(papers) * 0.5
            else "MODERATE"
            if high_evidence >= 1
            else "LOW"
        )

        logger.info("pubmed.summarize", pmids=pmids, focus=focus)
        return {
            "focus": focus,
            "papers_analysed": len(papers),
            "total_patients_across_studies": total_patients,
            "evidence_level_distribution": level_counts,
            "overall_evidence_quality": overall_quality,
            "summary": (
                f"Across {len(papers)} studies involving {total_patients} patients, "
                f"the evidence for {focus} is graded as {overall_quality}. "
                f"{high_evidence} studies provide Level 1 evidence."
            ),
            "papers": [
                {
                    "pmid": p["pmid"],
                    "title": p["title"],
                    "evidence_level": p["evidence_level"],
                    "year": p["year"],
                }
                for p in papers
            ],
        }
