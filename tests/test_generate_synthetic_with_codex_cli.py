import unittest
from pathlib import Path

from generate_synthetic_with_codex_cli import (
    CliConfig,
    build_prompt,
    chunk_records,
    schema_payload,
    validate_item,
)


class CodexCliSyntheticTests(unittest.TestCase):
    def test_build_prompt_includes_ids_and_sources(self) -> None:
        prompt = build_prompt(
            [
                {"id": "a1", "source": "هذا نص"},
                {"id": "a2", "source": "كيف الحال"},
            ]
        )
        self.assertIn("id: a1", prompt)
        self.assertIn("source: هذا نص", prompt)
        self.assertIn("id: a2", prompt)

    def test_chunk_records_respects_batch_size(self) -> None:
        batches = chunk_records([{"id": str(i), "source": "x"} for i in range(5)], 2, None)
        self.assertEqual([len(batch) for batch in batches], [2, 2, 1])

    def test_chunk_records_respects_max_batches(self) -> None:
        batches = chunk_records([{"id": str(i), "source": "x"} for i in range(10)], 2, 3)
        self.assertEqual(len(batches), 3)

    def test_schema_payload_requires_three_fields(self) -> None:
        schema = schema_payload()
        self.assertEqual(schema["type"], "array")
        self.assertEqual(schema["items"]["required"], ["id", "source", "target"])

    def test_validate_item_accepts_fully_diacritized_pair(self) -> None:
        is_valid, reason = validate_item({"id": "a1", "source": "هذا نص", "target": "هٰذَا نَصٌّ"})
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    def test_validate_item_rejects_partial_diacritization(self) -> None:
        is_valid, reason = validate_item({"id": "a1", "source": "هذا نص", "target": "هٰذَا نص"})
        self.assertFalse(is_valid)
        self.assertEqual(reason, "partial_diacritization")


if __name__ == "__main__":
    unittest.main()
