import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from import_chat_synthetic_batches import merge_batches, validate_item


TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


@contextmanager
def workspace_tempdir():
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TEMP_ROOT / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


class ImportChatSyntheticBatchTests(unittest.TestCase):
    def test_validate_item_accepts_fully_diacritized_pair(self) -> None:
        ok, reason = validate_item({"source": "هذا نص", "target": "هٰذَا نَصٌّ"})
        self.assertTrue(ok)
        self.assertIsNone(reason)

    def test_validate_item_rejects_mismatch(self) -> None:
        ok, reason = validate_item({"source": "هذا نص", "target": "هٰذَا نَصُّهُ"})
        self.assertFalse(ok)
        self.assertEqual(reason, "source_target_mismatch")

    def test_merge_batches_filters_invalid_and_duplicate_records(self) -> None:
        with workspace_tempdir() as temp_dir:
            batch_path = temp_dir / "batch_001.json"
            batch_path.write_text(
                json.dumps(
                    [
                        {"id": "a1", "source": "هذا نص", "target": "هٰذَا نَصٌّ"},
                        {"id": "a2", "source": "هذا نص", "target": "هٰذَا نص"},
                        {"id": "a1", "source": "هذا نص", "target": "هٰذَا نَصٌّ"},
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            accepted, rejected = merge_batches(temp_dir, "chat_manual")
            self.assertEqual(len(accepted), 1)
            self.assertEqual(len(rejected), 2)
            self.assertEqual(accepted[0]["generator_mode"], "chat_manual")


if __name__ == "__main__":
    unittest.main()
