"""Run with: python -m unittest discover -s tests -p 'test_regressions.py'."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from api.main import search_double_with_context
from api.schemas import SearchRequestWithContext
from ingestion.tasks import process_pdf_task


class ContextTests(unittest.TestCase):
    def test_context_window_respects_zero_and_defaults(self):
        for request_fields, expected in [
            ({"context_chunks": 0}, [4]),
            ({"context_chunks": 1}, [3, 4, 5]),
            ({"context_chunks": None}, list(range(1, 8))),
            ({}, list(range(1, 8))),
        ]:
            for use_elasticsearch in (False, True):
                with self.subTest(fields=request_fields, elasticsearch=use_elasticsearch):
                    hit = {"file_id": "doc", "filename": "doc.pdf", "chunk_index": 4,
                           "chunk_count": 10, "text": "Matched text"}
                    vector = Mock()
                    vector.hybrid_search.return_value = {"objects": [hit]}
                    vector.search.return_value = {"objects": [hit]}
                    elasticsearch = Mock() if use_elasticsearch else None
                    if elasticsearch is not None:
                        elasticsearch.search.return_value = {"objects": [hit]}
                    store = elasticsearch if use_elasticsearch else vector
                    store.get_chunks_by_range.side_effect = lambda _, indices: [
                        {"chunk_index": i, "text": f"Chunk {i}"} for i in indices
                    ]
                    with patch("retrieval.pipeline.rerank_cross_encoder", return_value=[hit]):
                        response = search_double_with_context(
                            SearchRequestWithContext(q="query", **request_fields),
                            vector, elasticsearch,
                        )
                    result = json.loads(response.body)
                    store.get_chunks_by_range.assert_called_once_with("doc", expected)
                    self.assertEqual(result["documents"][0]["augmented_chunk_indices"], expected)
                    self.assertEqual(result["context_chunks"], (len(expected) - 1) // 2)
                    self.assertEqual(
                        result["retrieval_mode"],
                        "rrf" if use_elasticsearch else "weaviate_hybrid",
                    )


class RerankFlagTests(unittest.TestCase):
    """`/search/double/context` ignored `rerank` until the schema gained the field."""

    def _run(self, rerank, use_elasticsearch):
        hit = {"file_id": "doc", "filename": "doc.pdf", "chunk_index": 0,
               "chunk_count": 1, "text": "Matched text"}
        vector = Mock()
        vector.hybrid_search.return_value = {"objects": [hit]}
        vector.search.return_value = {"objects": [hit]}
        elasticsearch = Mock() if use_elasticsearch else None
        if elasticsearch is not None:
            elasticsearch.search.return_value = {"objects": [hit]}
        store = elasticsearch if use_elasticsearch else vector
        store.get_chunks_by_range.side_effect = lambda _, indices: [
            {"chunk_index": i, "text": f"Chunk {i}"} for i in indices
        ]
        with patch("retrieval.pipeline.rerank_cross_encoder", return_value=[hit]) as reranker:
            search_double_with_context(
                SearchRequestWithContext(q="query", rerank=rerank),
                vector, elasticsearch,
            )
        return reranker

    def test_rerank_false_skips_the_cross_encoder(self):
        for use_elasticsearch in (False, True):
            with self.subTest(elasticsearch=use_elasticsearch):
                self.assertFalse(self._run(False, use_elasticsearch).called)

    def test_rerank_defaults_to_on(self):
        for use_elasticsearch in (False, True):
            with self.subTest(elasticsearch=use_elasticsearch):
                self.assertTrue(self._run(True, use_elasticsearch).called)


class UploadCleanupTests(unittest.TestCase):
    def run_task(self, path, extraction, store=None):
        with patch("ingestion.tasks.extract_text_from_pdf", extraction), \
             patch("ingestion.tasks.get_store", return_value=store or Mock()), \
             patch("ingestion.tasks.USE_ELASTICSEARCH", False):
            return process_pdf_task.run(str(path), "doc", "doc.pdf")

    def test_cleanup_after_extraction_errors_or_empty_text(self):
        for error in ("encrypted PDF", "no text", "invalid PDF", None):
            with self.subTest(error=error), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "upload.pdf"
                path.write_bytes(b"test")
                extraction = Mock(side_effect=ValueError(error)) if error else Mock(return_value=[])
                result = self.run_task(path, extraction)
                self.assertEqual(result["status"], "error")
                self.assertFalse(path.exists())

    def test_cleanup_after_success_or_indexing_failure(self):
        for fails in (False, True):
            with self.subTest(fails=fails), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "upload.pdf"
                path.write_bytes(b"test")
                store = Mock()
                if fails:
                    store.add_documents.side_effect = RuntimeError("index unavailable")
                result = self.run_task(path, Mock(return_value=[{"text": "hello"}]), store)
                self.assertEqual(result["status"], "error" if fails else "success")
                self.assertFalse(path.exists())

    def test_missing_upload_preserves_extraction_error(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_task(Path(directory) / "missing.pdf", Mock(side_effect=FileNotFoundError("missing upload")))
            self.assertEqual(result["status"], "error")
            self.assertIn("missing upload", result["message"])

    def test_cleanup_failure_is_logged_without_hiding_original_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upload.pdf"
            path.write_bytes(b"test")
            with patch("ingestion.tasks.os.remove", side_effect=PermissionError("denied")), \
                 self.assertLogs("ingestion.tasks", level="ERROR"):
                result = self.run_task(path, Mock(side_effect=ValueError("invalid PDF")))
            self.assertEqual(result["status"], "error")
            self.assertIn("invalid PDF", result["message"])


if __name__ == "__main__":
    unittest.main()
