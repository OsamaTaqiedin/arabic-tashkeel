import json
import unittest

from generate_synthetic_tashkeel import (
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    make_request_payload,
    parse_response_output,
    validate_synthetic_pair,
)


class SyntheticGenerationTests(unittest.TestCase):
    def test_make_request_payload_contains_schema(self) -> None:
        payload = make_request_payload("هذا نص", DEFAULT_MODEL, DEFAULT_SYSTEM_PROMPT)
        self.assertEqual(payload["model"], DEFAULT_MODEL)
        self.assertEqual(payload["text"]["format"]["type"], "json_schema")
        self.assertTrue(payload["text"]["format"]["strict"])

    def test_parse_response_output_reads_output_text(self) -> None:
        response = {
            "output_text": json.dumps(
                {"source": "هذا نص", "target": "هٰذَا نَصٌّ"},
                ensure_ascii=False,
            )
        }
        parsed = parse_response_output(response)
        self.assertEqual(parsed["target"], "هٰذَا نَصٌّ")

    def test_validate_synthetic_pair_accepts_fully_diacritized_target(self) -> None:
        is_valid, reason = validate_synthetic_pair("هذا نص", "هٰذَا نَصٌّ")
        self.assertTrue(is_valid)
        self.assertIsNone(reason)

    def test_validate_synthetic_pair_rejects_partial_diacritization(self) -> None:
        is_valid, reason = validate_synthetic_pair("هذا نص", "هٰذَا نص")
        self.assertFalse(is_valid)
        self.assertEqual(reason, "partial_diacritization")

    def test_validate_synthetic_pair_rejects_source_mismatch(self) -> None:
        is_valid, reason = validate_synthetic_pair("هذا نص", "هٰذَا نَصُّهُ")
        self.assertFalse(is_valid)
        self.assertEqual(reason, "source_target_mismatch")


if __name__ == "__main__":
    unittest.main()
