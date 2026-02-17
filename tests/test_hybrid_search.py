"""Tests for retrieval.hybrid_search reciprocal rank fusion."""

from retrieval.hybrid_search import reciprocal_rank_fusion


class TestReciprocalRankFusion:
    def test_single_list(self):
        results = [
            [
                {"chunk_id": "a", "content": "A"},
                {"chunk_id": "b", "content": "B"},
            ]
        ]
        merged = reciprocal_rank_fusion(results)
        assert len(merged) == 2
        assert merged[0]["chunk_id"] == "a"  # rank 0 -> higher score
        assert merged[1]["chunk_id"] == "b"

    def test_two_identical_lists_boost(self):
        list1 = [
            {"chunk_id": "a", "content": "A"},
            {"chunk_id": "b", "content": "B"},
        ]
        list2 = [
            {"chunk_id": "a", "content": "A"},
            {"chunk_id": "b", "content": "B"},
        ]
        merged = reciprocal_rank_fusion([list1, list2])
        assert merged[0]["chunk_id"] == "a"
        # a should have 2x the score contribution
        assert merged[0]["rrf_score"] > merged[1]["rrf_score"]

    def test_different_lists_merged(self):
        list1 = [{"chunk_id": "a", "content": "A"}]
        list2 = [{"chunk_id": "b", "content": "B"}]
        merged = reciprocal_rank_fusion([list1, list2])
        assert len(merged) == 2
        ids = {d["chunk_id"] for d in merged}
        assert ids == {"a", "b"}

    def test_empty_lists(self):
        merged = reciprocal_rank_fusion([[], []])
        assert merged == []

    def test_rrf_scores_attached(self):
        results = [[{"chunk_id": "a", "content": "A"}]]
        merged = reciprocal_rank_fusion(results)
        assert "rrf_score" in merged[0]
        assert merged[0]["rrf_score"] > 0

    def test_k_parameter(self):
        # Use separate dict objects to avoid mutation from shared references
        results_k10 = [[{"chunk_id": "a", "content": "A"}]]
        results_k100 = [[{"chunk_id": "a", "content": "A"}]]
        merged_k10 = reciprocal_rank_fusion(results_k10, k=10)
        merged_k100 = reciprocal_rank_fusion(results_k100, k=100)
        # Smaller k gives higher score for rank 0: 1/(k+1)
        assert merged_k10[0]["rrf_score"] > merged_k100[0]["rrf_score"]

    def test_overlapping_results_boosted(self):
        list1 = [
            {"chunk_id": "a", "content": "A"},
            {"chunk_id": "b", "content": "B"},
            {"chunk_id": "c", "content": "C"},
        ]
        list2 = [
            {"chunk_id": "c", "content": "C"},
            {"chunk_id": "a", "content": "A"},
        ]
        merged = reciprocal_rank_fusion([list1, list2])
        # Both 'a' and 'c' appear in both lists, should have higher scores
        scores = {d["chunk_id"]: d["rrf_score"] for d in merged}
        assert scores["a"] > scores["b"]  # 'a' in both, 'b' in only one
