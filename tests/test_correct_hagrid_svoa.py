import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "correct_hagrid_svoa.py"


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def run_converter(directory: Path, rows, annotations):
    annotation_root = directory / "annotations"
    for relative, value in annotations.items():
        write_json(annotation_root / relative, value)
    dataset = directory / "dataset.jsonl"
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    output = directory / "corrected.jsonl"
    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dataset-jsonl",
            str(dataset),
            "--hagrid-annotations-dir",
            str(annotation_root),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in output.read_text().splitlines()]


class CorrectHagridSvoaTest(unittest.TestCase):
    def test_official_gesture_replaces_only_action_without_interpretation(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            row = {
                "image_id": "hagrid:little_finger/abc.jpg",
                "image_path": "cache/abc.jpg",
                "subject": "woman",
                "action": "pose",
                "object": "none",
                "adjuncts": "room",
            }
            result = run_converter(
                directory,
                [row],
                {
                    "train/little_finger.json": {
                        "abc.jpg": {
                            "labels": ["little_finger"],
                            "bboxes": [[0.1, 0.2, 0.3, 0.4]],
                            "united_label": "no_gesture",
                            "user_id": "user-7",
                            "meta": {"age": 30},
                        }
                    }
                },
            )[0]
            self.assertEqual(result["action"], "little finger")
            self.assertEqual(result["subject"], "woman")
            self.assertEqual(result["object"], "none")
            self.assertEqual(result["adjuncts"], "room")
            self.assertEqual(result["hagrid_primary_label"], "little_finger")
            self.assertEqual(result["hagrid_labels"], ["little_finger"])
            self.assertEqual(result["hagrid_user_id"], "user-7")
            self.assertEqual(result["split"], "train")
            self.assertEqual(result["vlm_original"]["action"], "pose")

    def test_no_gesture_preserves_complete_vlm_label(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            row = {
                "image_id": "hagrid:no_gesture/plain.png",
                "image_path": "cache/plain.jpg",
                "subject": "person",
                "action": "wave",
                "object": "unknown",
                "adjuncts": "office",
            }
            result = run_converter(
                directory,
                [row],
                {
                    "val/no_gesture.json": {
                        "plain.png": {
                            "labels": ["no_gesture"],
                            "bboxes": [[0, 0, 1, 1]],
                            "user_id": "u1",
                        }
                    }
                },
            )[0]
            for field in ("subject", "action", "object", "adjuncts"):
                self.assertEqual(result[field], row[field])
            self.assertIsNone(result["hagrid_primary_label"])
            self.assertEqual(result["annotation_source"], "vlm")
            self.assertEqual(result["corrected_fields"], [])

    def test_united_label_has_priority(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            row = {
                "image_id": "hagrid:hand_heart2/heart.jpg",
                "image_path": "cache/heart.jpg",
                "subject": "person",
                "action": "hold",
                "object": "none",
                "adjuncts": "none",
            }
            result = run_converter(
                directory,
                [row],
                {
                    "test/hand_heart2.json": {
                        "heart.jpg": {
                            "labels": ["palm", "palm"],
                            "bboxes": [[0, 0, 0.4, 0.4], [0, 0, 0.5, 0.5]],
                            "united_label": "hand_heart2",
                        }
                    }
                },
            )[0]
            self.assertEqual(result["action"], "hand heart2")
            self.assertEqual(result["hagrid_primary_label"], "hand_heart2")

    def test_distinct_hands_use_largest_box_and_keep_all_labels(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            row = {
                "image_id": "hagrid:like/twohands.jpg",
                "image_path": "cache/twohands.jpg",
                "subject": "person",
                "action": "gesture",
                "object": "none",
                "adjuncts": "camera",
            }
            result = run_converter(
                directory,
                [row],
                {
                    "train/like.json": {
                        "twohands.jpg": {
                            "labels": ["no_gesture", "like", "three_gun"],
                            "bboxes": [
                                [0, 0, 0.9, 0.9],
                                [0, 0, 0.2, 0.2],
                                [0, 0, 0.5, 0.4],
                            ],
                        }
                    }
                },
            )[0]
            self.assertEqual(result["action"], "three gun")
            self.assertEqual(
                result["hagrid_labels"],
                ["no_gesture", "like", "three_gun"],
            )
            self.assertEqual(
                result["hagrid_non_no_labels"], ["like", "three_gun"]
            )

    def test_non_hagrid_row_is_unchanged(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            row = {
                "image_id": "swig:one.jpg",
                "image_path": "one.jpg",
                "subject": "dog",
                "action": "run",
                "object": "none",
                "adjuncts": "park",
            }
            result = run_converter(
                directory,
                [row],
                {
                    "train/like.json": {
                        "unused.jpg": {"labels": ["like"], "bboxes": [[0, 0, 1, 1]]}
                    }
                },
            )[0]
            self.assertEqual(result, row)


if __name__ == "__main__":
    unittest.main()
