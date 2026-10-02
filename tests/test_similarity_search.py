"""Tests for agent/similarity_search.py. TF-IDF only -- the embedding path is
exercised solely through its fallback, so nothing is ever downloaded."""
import builtins

import pytest

from agent import memory
from agent import similarity_search as ss


def _report(findings, name="Web Security"):
    return {"overall_score": 70, "grade": "C", "categories": [{"name": name, "score": 70, "weight": 1.0,
                                                                "findings": findings}]}


class TestBuildCorpus:
    def test_seed_entries_are_tagged_as_seed(self, tmp_db_path):
        corpus = ss.build_corpus(include_real=False)
        assert len(corpus) == len(ss.SEED_FINDINGS)
        assert {e["source"] for e in corpus} == {"seed"}

    def test_real_findings_are_tagged_with_domain_and_timestamp(self, tmp_db_path):
        memory.save_audit("https://example.com", _report([
            {"severity": "warning", "issue": "Missing HSTS header.", "recommendation": "Add it."},
        ]))
        corpus = ss.build_corpus(include_seed=False)
        assert len(corpus) == 1
        assert corpus[0]["source"] == "real_audit"
        assert corpus[0]["domain"] == "example.com"
        assert corpus[0]["timestamp"]
        assert corpus[0]["category"] == "Web Security"

    def test_returned_seed_entries_are_copies(self, tmp_db_path):
        corpus = ss.build_corpus(include_real=False)
        corpus[0]["issue"] = "mutated"
        assert ss.SEED_FINDINGS[0]["issue"] != "mutated"

    def test_non_dict_category_is_skipped(self):
        assert ss._findings_from_report({"categories": ["not a dict"]}) == []


class TestBuildIndex:
    def test_default_backend_is_tfidf(self, tmp_db_path):
        assert ss.build_index().backend == "tfidf"

    def test_empty_corpus_raises(self, tmp_db_path):
        with pytest.raises(ValueError):
            ss.build_index(include_seed=False, include_real=True)

    def test_embedding_falls_back_to_tfidf_when_package_missing(self, tmp_db_path, monkeypatch):
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "sentence_transformers":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        logs = []
        index = ss.build_index(backend="embedding", log_fn=logs.append)
        assert index.backend == "tfidf"
        assert any("falling back to TF-IDF" in m for m in logs)


class TestSearch:
    def test_most_similar_finding_ranks_first(self, tmp_db_path):
        index = ss.build_index(include_real=False)
        results = ss.search(index, "page is missing a meta description")
        assert results[0]["issue"] == "Missing meta description."
        assert results[0]["source"] == "seed"
        assert results[0]["similarity"] > 0

    def test_top_k_limits_results(self, tmp_db_path):
        index = ss.build_index(include_real=False)
        assert len(ss.search(index, "certificate", top_k=2)) == 2

    def test_category_filter(self, tmp_db_path):
        index = ss.build_index(include_real=False)
        results = ss.search(index, "alt text images", top_k=10, category="Accessibility")
        assert results
        assert {r["category"] for r in results} == {"Accessibility"}

    def test_results_sorted_by_descending_similarity(self, tmp_db_path):
        index = ss.build_index(include_real=False)
        sims = [r["similarity"] for r in ss.search(index, "ssl certificate expired", top_k=5)]
        assert sims == sorted(sims, reverse=True)

    def test_identical_findings_from_repeat_audits_are_deduplicated(self, tmp_db_path):
        # Regression: two audit dates of the same site used to fill top_k
        # with literal duplicates of the same finding.
        finding = {"severity": "critical", "issue": "Zebra stripes missing from favicon.",
                   "recommendation": "Paint the zebra stripes back on."}
        memory.save_audit("https://example.com", _report([finding]))
        memory.save_audit("https://example.com", _report([finding]))
        index = ss.build_index()
        results = ss.search(index, "zebra stripes favicon", top_k=5)
        assert [r["issue"] for r in results].count(finding["issue"]) == 1
