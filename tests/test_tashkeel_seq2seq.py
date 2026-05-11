import json
import shutil
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import skipUnless

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from tashkeel_seq2seq.dataset import (
    CharVocabulary,
    TashkeelDataset,
    build_char_vocab,
    collate_batch,
    load_jsonl,
)
from tashkeel_seq2seq.metrics import compute_grouped_metrics, exact_match_rate, levenshtein_distance
from tashkeel_seq2seq.metrics import case_ending_error_rate, diacritic_error_rate, word_error_rate


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


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


class DatasetTests(unittest.TestCase):
    def test_load_jsonl_reads_expected_schema(self) -> None:
        with workspace_tempdir() as temp_dir:
            path = temp_dir / "train.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": "1",
                        "source": "هذا نص",
                        "target": "هٰذَا نَصٌّ",
                        "domain": "msa",
                        "source_file": "a.txt",
                        "split": "train",
                    }
                ],
            )
            examples = load_jsonl(path)
            self.assertEqual(len(examples), 1)
            self.assertEqual(examples[0].domain, "msa")

    def test_vocab_round_trip(self) -> None:
        vocab = CharVocabulary(["<pad>", "<bos>", "<eos>", "<unk>", "ه", "ذ", "ا"])
        text = "هذا"
        encoded = vocab.encode(text, add_bos=True, add_eos=True)
        decoded = vocab.decode(encoded)
        self.assertEqual(decoded, text)

    def test_build_char_vocab_contains_characters(self) -> None:
        with workspace_tempdir() as temp_dir:
            path = temp_dir / "train.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "id": "1",
                        "source": "هذا",
                        "target": "هٰذَا",
                        "domain": "msa",
                        "source_file": "a.txt",
                        "split": "train",
                    }
                ],
            )
            examples = load_jsonl(path)
            vocab = build_char_vocab(examples)
            self.assertIn("ه", vocab.stoi)


@skipUnless(torch is not None, "PyTorch is not installed")
class TorchDatasetTests(unittest.TestCase):
    def test_collate_shapes(self) -> None:
        vocab = CharVocabulary(["<pad>", "<bos>", "<eos>", "<unk>", "ه", "ذ", "ا", "ن", "ص"])
        dataset = TashkeelDataset(
            [
                type("Example", (), {"id": "1", "source": "هذا", "target": "هذا", "domain": "msa", "source_file": "a", "split": "train"})(),
                type("Example", (), {"id": "2", "source": "نص", "target": "نص", "domain": "classical", "source_file": "b", "split": "train"})(),
            ],
            vocab,
        )
        batch = collate_batch([dataset[0], dataset[1]], vocab.pad_id)
        self.assertEqual(tuple(batch["source_ids"].shape), (2, 4))
        self.assertEqual(tuple(batch["target_ids"].shape), (2, 5))


class MetricsTests(unittest.TestCase):
    def test_levenshtein_distance(self) -> None:
        self.assertEqual(levenshtein_distance("abc", "axc"), 1)

    def test_exact_match_rate(self) -> None:
        self.assertEqual(exact_match_rate(["a", "b"], ["a", "c"]), 0.5)

    def test_compute_grouped_metrics(self) -> None:
        metrics = compute_grouped_metrics(
            predictions=["a", "b", "c"],
            targets=["a", "x", "c"],
            domains=["classical", "msa", "classical"],
        )
        self.assertIn("overall", metrics)
        self.assertIn("classical", metrics)
        self.assertIn("msa", metrics)

    def test_diacritic_error_rate(self) -> None:
        prediction = "هٰذَا كَتَابٌ"
        target = "هٰذَا كِتَابٌ"
        self.assertAlmostEqual(diacritic_error_rate([prediction], [target]), 1 / 7)

    def test_word_error_rate(self) -> None:
        prediction = "هٰذَا كَتَابٌ"
        target = "هٰذَا كِتَابٌ"
        self.assertAlmostEqual(word_error_rate([prediction], [target]), 1 / 2)

    def test_case_ending_error_rate(self) -> None:
        prediction = "هٰذَا كِتَابٍ"
        target = "هٰذَا كِتَابٌ"
        self.assertAlmostEqual(case_ending_error_rate([prediction], [target]), 1 / 2)


@skipUnless(torch is not None, "PyTorch is not installed")
class ModelAndTrainingTests(unittest.TestCase):
    def test_forward_decode_and_smoke_train(self) -> None:
        from tashkeel_seq2seq.train import TrainingConfig, build_dataloaders, build_model, evaluate, train_model

        with workspace_tempdir() as temp_dir:
            train_path = temp_dir / "train.jsonl"
            validation_path = temp_dir / "validation.jsonl"
            test_path = temp_dir / "test.jsonl"
            records = [
                {
                    "id": "1",
                    "source": "هذا نص",
                    "target": "هٰذَا نَصٌّ",
                    "domain": "msa",
                    "source_file": "a.txt",
                    "split": "train",
                },
                {
                    "id": "2",
                    "source": "ذلك كتاب",
                    "target": "ذٰلِكَ كِتَابٌ",
                    "domain": "classical",
                    "source_file": "b.txt",
                    "split": "train",
                },
                {
                    "id": "3",
                    "source": "هذا كتاب",
                    "target": "هٰذَا كِتَابٌ",
                    "domain": "msa",
                    "source_file": "c.txt",
                    "split": "validation",
                },
                {
                    "id": "4",
                    "source": "ذلك نص",
                    "target": "ذٰلِكَ نَصٌّ",
                    "domain": "classical",
                    "source_file": "d.txt",
                    "split": "test",
                },
            ]
            write_jsonl(train_path, [records[0], records[1]])
            write_jsonl(validation_path, [records[2]])
            write_jsonl(test_path, [records[3]])

            config = TrainingConfig(
                train_path=str(train_path),
                validation_path=str(validation_path),
                test_path=str(test_path),
                checkpoint_dir=str(temp_dir / "artifacts"),
                embedding_dim=16,
                encoder_hidden_size=8,
                decoder_hidden_size=16,
                batch_size=2,
                learning_rate=1e-3,
                epochs=1,
                teacher_forcing_ratio=1.0,
                gradient_clip=1.0,
                dropout=0.0,
                max_decode_length=16,
                min_frequency=1,
                num_workers=0,
                device="cpu",
                use_amp=False,
                resume_from_latest=False,
                limit_train_examples=None,
                limit_validation_examples=None,
                limit_test_examples=None,
                run_test=True,
            )

            vocab, train_loader, validation_loader, _ = build_dataloaders(config)
            batch = next(iter(train_loader))
            model = build_model(config, len(vocab), vocab.pad_id, vocab.bos_id, vocab.eos_id)
            logits = model(batch["source_ids"], batch["source_lengths"], batch["target_ids"], teacher_forcing_ratio=1.0)
            self.assertEqual(logits.size(0), 2)
            decoded = model.greedy_decode(batch["source_ids"], batch["source_lengths"], max_decode_length=16)
            self.assertEqual(decoded.dim(), 2)

            criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.pad_id)
            evaluation = evaluate(model, validation_loader, criterion, vocab, torch.device("cpu"), 16, False)
            self.assertIn("overall", evaluation["metrics"])
            self.assertIn("msa", evaluation["metrics"])

            result = train_model(config)
            self.assertEqual(result["best_epoch"], 1)
            self.assertTrue((temp_dir / "artifacts" / "best.pt").exists())
            self.assertTrue((temp_dir / "artifacts" / "history.json").exists())
            self.assertIn("epoch_seconds", result["history"][0])

    def test_limits_and_resume_from_latest(self) -> None:
        from tashkeel_seq2seq.train import TrainingConfig, build_dataloaders, train_model

        with workspace_tempdir() as temp_dir:
            train_path = temp_dir / "train.jsonl"
            validation_path = temp_dir / "validation.jsonl"
            test_path = temp_dir / "test.jsonl"
            records = []
            for index in range(6):
                records.append(
                    {
                        "id": str(index),
                        "source": f"هذا نص {index}",
                        "target": f"هٰذَا نَصٌّ {index}",
                        "domain": "msa" if index % 2 == 0 else "classical",
                        "source_file": f"{index}.txt",
                        "split": "train",
                    }
                )
            write_jsonl(train_path, records)
            write_jsonl(validation_path, [dict(records[0], split="validation")])
            write_jsonl(test_path, [dict(records[1], split="test")])

            config = TrainingConfig(
                train_path=str(train_path),
                validation_path=str(validation_path),
                test_path=str(test_path),
                checkpoint_dir=str(temp_dir / "artifacts"),
                embedding_dim=8,
                encoder_hidden_size=4,
                decoder_hidden_size=8,
                batch_size=2,
                learning_rate=1e-3,
                epochs=1,
                teacher_forcing_ratio=1.0,
                gradient_clip=1.0,
                dropout=0.0,
                max_decode_length=12,
                min_frequency=1,
                num_workers=0,
                device="cpu",
                use_amp=False,
                resume_from_latest=False,
                limit_train_examples=3,
                limit_validation_examples=1,
                limit_test_examples=1,
                run_test=False,
            )

            _, train_loader, validation_loader, test_loader = build_dataloaders(config)
            self.assertEqual(len(train_loader.dataset), 3)
            self.assertEqual(len(validation_loader.dataset), 1)
            self.assertEqual(len(test_loader.dataset), 1)

            first_result = train_model(config)
            self.assertEqual(first_result["best_epoch"], 1)

            resumed_config = TrainingConfig(
                train_path=config.train_path,
                validation_path=config.validation_path,
                test_path=config.test_path,
                checkpoint_dir=config.checkpoint_dir,
                embedding_dim=config.embedding_dim,
                encoder_hidden_size=config.encoder_hidden_size,
                decoder_hidden_size=config.decoder_hidden_size,
                batch_size=config.batch_size,
                learning_rate=config.learning_rate,
                epochs=2,
                teacher_forcing_ratio=config.teacher_forcing_ratio,
                gradient_clip=config.gradient_clip,
                dropout=config.dropout,
                max_decode_length=config.max_decode_length,
                min_frequency=config.min_frequency,
                num_workers=config.num_workers,
                device=config.device,
                use_amp=False,
                resume_from_latest=True,
                limit_train_examples=config.limit_train_examples,
                limit_validation_examples=config.limit_validation_examples,
                limit_test_examples=config.limit_test_examples,
                run_test=False,
            )
            resumed_result = train_model(resumed_config)
            self.assertEqual(len(resumed_result["history"]), 2)
            self.assertEqual(resumed_result["history"][-1]["epoch"], 2)

    def test_simple_model_smoke_train(self) -> None:
        from tashkeel_seq2seq.simple_train import SimpleTrainingConfig, train_model

        with workspace_tempdir() as temp_dir:
            train_path = temp_dir / "train.jsonl"
            validation_path = temp_dir / "validation.jsonl"
            test_path = temp_dir / "test.jsonl"
            records = [
                {
                    "id": "1",
                    "source": "هذا نص",
                    "target": "هٰذَا نَصٌّ",
                    "domain": "msa",
                    "source_file": "a.txt",
                    "split": "train",
                },
                {
                    "id": "2",
                    "source": "ذلك نص",
                    "target": "ذٰلِكَ نَصٌّ",
                    "domain": "classical",
                    "source_file": "b.txt",
                    "split": "validation",
                },
                {
                    "id": "3",
                    "source": "هذا كتاب",
                    "target": "هٰذَا كِتَابٌ",
                    "domain": "msa",
                    "source_file": "c.txt",
                    "split": "test",
                },
            ]
            write_jsonl(train_path, [records[0]])
            write_jsonl(validation_path, [records[1]])
            write_jsonl(test_path, [records[2]])

            config = SimpleTrainingConfig(
                train_path=str(train_path),
                validation_path=str(validation_path),
                test_path=str(test_path),
                checkpoint_dir=str(temp_dir / "simple_artifacts"),
                embedding_dim=8,
                hidden_size=16,
                batch_size=1,
                learning_rate=1e-3,
                epochs=1,
                teacher_forcing_ratio=1.0,
                max_decode_length=16,
                device="cpu",
                run_test=True,
            )
            result = train_model(config)
            self.assertTrue((temp_dir / "simple_artifacts" / "best.pt").exists())
            self.assertIn("history", result)


if __name__ == "__main__":
    unittest.main()
