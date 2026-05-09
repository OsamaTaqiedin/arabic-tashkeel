import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path

from build_tashkeela_dataset import (
    all_arabic_words_have_diacritics,
    BuilderConfig,
    Document,
    arabic_character_count,
    arabic_words,
    build_examples,
    likely_header_noise,
    normalize_text,
    run,
    segment_sentences,
    split_documents,
    strip_diacritics,
)

TEST_TEMP_ROOT = Path(__file__).resolve().parent / ".tmp"


@contextmanager
def workspace_tempdir():
    TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    path = TEST_TEMP_ROOT / f"case_{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=False)
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class StripDiacriticsTests(unittest.TestCase):
    def test_removes_shadda_and_vowels(self) -> None:
        self.assertEqual(strip_diacritics("النَّصُّ"), "النص")

    def test_removes_tanween_sukun_and_dagger_alif(self) -> None:
        self.assertEqual(strip_diacritics("هٰذَا كِتَابٌ لَمْ يُكْتَبْ"), "هذا كتاب لم يكتب")


class SegmentationTests(unittest.TestCase):
    def test_splits_on_sentence_punctuation(self) -> None:
        text = "هٰذَا نَصٌّ. هَلْ يَعْمَلُ؟ نَعَمْ!"
        sentences = [sentence for _, sentence in segment_sentences(text)]
        self.assertEqual(sentences, ["هٰذَا نَصٌّ.", "هَلْ يَعْمَلُ؟", "نَعَمْ!"])

    def test_splits_on_blank_lines(self) -> None:
        text = "هٰذَا سَطْرٌ\n\nذٰلِكَ سَطْرٌ آخَرُ"
        sentences = [sentence for _, sentence in segment_sentences(text)]
        self.assertEqual(sentences, ["هٰذَا سَطْرٌ", "ذٰلِكَ سَطْرٌ آخَرُ"])


class FilteringTests(unittest.TestCase):
    def test_arabic_words_extracts_letter_tokens(self) -> None:
        self.assertEqual(arabic_words("هٰذَا نَصٌّ 123 test"), ["هٰذَا", "نَصٌّ"])

    def test_fully_diacritized_sentence_is_accepted_by_word_check(self) -> None:
        self.assertTrue(all_arabic_words_have_diacritics("هٰذَا نَصٌّ مُشَكَّلٌ"))

    def test_partially_diacritized_sentence_is_rejected_by_word_check(self) -> None:
        self.assertFalse(all_arabic_words_have_diacritics("هٰذَا نص مُشَكَّلٌ"))

    def test_header_noise_detected_near_start(self) -> None:
        self.assertTrue(likely_header_noise("الكتاب : عنوان", 0))

    def test_normalize_text_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text("أ  ب\t\tج\n\n د "), "أ ب ج\n\nد")

    def test_arabic_character_count(self) -> None:
        self.assertEqual(arabic_character_count("abc هٰذَا 123"), 5)


class SplitTests(unittest.TestCase):
    def test_split_documents_is_deterministic(self) -> None:
        documents = [
            Document(path=Path(f"doc{i}.txt"), relative_path=f"doc{i}.txt", domain="classical")
            for i in range(20)
        ]
        config = BuilderConfig(
            input_dir=Path("input"),
            output_dir=Path("output"),
            train_ratio=0.8,
            validation_ratio=0.1,
            test_ratio=0.1,
            min_chars=8,
            max_chars=400,
            seed=42,
            rejected_path=None,
        )
        first = split_documents(documents, config)
        second = split_documents(documents, config)
        self.assertEqual(first, second)


class IntegrationTests(unittest.TestCase):
    def test_build_examples_filters_headers_and_keeps_diacritized_content(self) -> None:
        with workspace_tempdir() as temp_dir:
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text(
                "الكتاب : عنوان.\n\n"
                "هٰذَا نَصٌّ مُشَكَّلٌ.\n"
                "سَطْرٌ ثَانٍ.\n",
                encoding="utf-8",
            )
            document = Document(
                path=file_path,
                relative_path="sample.txt",
                domain="classical",
            )
            config = BuilderConfig(
                input_dir=Path(temp_dir),
                output_dir=Path(temp_dir) / "out",
                train_ratio=0.8,
                validation_ratio=0.1,
                test_ratio=0.1,
                min_chars=4,
                max_chars=200,
                seed=7,
                rejected_path=None,
            )
            accepted, rejected = build_examples(document, config)
            self.assertEqual(len(accepted), 2)
            self.assertTrue(
                any(record["reason"] in {"header_noise", "no_diacritics_removed"} for record in rejected)
            )
            self.assertEqual(accepted[0]["source"], "هذا نص مشكل.")

    def test_build_examples_rejects_partially_diacritized_sentence(self) -> None:
        with workspace_tempdir() as temp_dir:
            file_path = Path(temp_dir) / "sample.txt"
            file_path.write_text(
                "هٰذَا نَصٌّ مُشَكَّلٌ.\n"
                "هٰذَا نص مُشَكَّلٌ.\n",
                encoding="utf-8",
            )
            document = Document(
                path=file_path,
                relative_path="sample.txt",
                domain="classical",
            )
            config = BuilderConfig(
                input_dir=Path(temp_dir),
                output_dir=Path(temp_dir) / "out",
                train_ratio=0.8,
                validation_ratio=0.1,
                test_ratio=0.1,
                min_chars=4,
                max_chars=200,
                seed=7,
                rejected_path=None,
            )
            accepted, rejected = build_examples(document, config)
            self.assertEqual(len(accepted), 1)
            self.assertTrue(any(record["reason"] == "partial_diacritization" for record in rejected))

    def test_run_writes_non_leaking_splits(self) -> None:
        with workspace_tempdir() as temp_dir:
            root = Path(temp_dir)
            corpus = root / "texts.txt"
            corpus.mkdir()
            (corpus / "msa").mkdir()
            (corpus / "book1.txt").write_text("هٰذَا نَصٌّ أَوَّلٌ.\n", encoding="utf-8")
            (corpus / "book2.txt").write_text("ذٰلِكَ نَصٌّ ثَانٍ.\n", encoding="utf-8")
            (corpus / "msa" / "news.txt").write_text("خَبَرٌ جَدِيدٌ.\n", encoding="utf-8")
            config = BuilderConfig(
                input_dir=corpus,
                output_dir=root / "dataset",
                train_ratio=0.34,
                validation_ratio=0.33,
                test_ratio=0.33,
                min_chars=4,
                max_chars=200,
                seed=3,
                rejected_path="rejected.jsonl",
            )

            manifest = run(config)

            seen_files = {}
            for split in ("train", "validation", "test"):
                split_path = config.output_dir / f"{split}.jsonl"
                self.assertTrue(split_path.exists())
                with split_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        record = json.loads(line)
                        self.assertEqual(record["split"], split)
                        self.assertEqual(record["source"], strip_diacritics(record["target"]))
                        self.assertNotEqual(record["source"], record["target"])
                        seen_files.setdefault(record["source_file"], split)
                        self.assertEqual(seen_files[record["source_file"]], split)

            manifest_path = config.output_dir / "dataset_manifest.json"
            self.assertTrue(manifest_path.exists())
            saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(saved_manifest["documents_per_split"], manifest["documents_per_split"])


if __name__ == "__main__":
    unittest.main()
