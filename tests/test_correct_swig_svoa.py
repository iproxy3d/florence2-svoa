import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "correct_swig_svoa.py"


def write_json(path: Path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class CorrectSwigSvoaTest(unittest.TestCase):
    def test_corrects_only_confident_fields_and_preserves_other_sources(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            swig = directory / "SWiG_jsons"
            swig.mkdir()
            official = {
                "drinking_1.jpg": {
                    "verb": "drinking",
                    "bb": {
                        "agent": [0, 0, 100, 200],
                        "liquid": [40, 50, 80, 100],
                        "container": [35, 45, 85, 110],
                        "place": [-1, -1, -1, -1],
                    },
                    "frames": [
                        {
                            "agent": "n-woman",
                            "liquid": "n-water",
                            "container": "n-glass",
                            "place": "n-kitchen",
                        },
                        {
                            "agent": "n-woman",
                            "liquid": "n-water",
                            "container": "n-glass",
                            "place": "n-kitchen",
                        },
                        {
                            "agent": "n-person",
                            "liquid": "n-juice",
                            "container": "n-glass",
                            "place": "n-home",
                        },
                    ],
                }
            }
            write_json(swig / "train.json", official)
            write_json(swig / "dev.json", {})
            write_json(swig / "test.json", {})
            write_json(
                swig / "imsitu_space.json",
                {
                    "verbs": {
                        "drinking": {
                            "order": ["agent", "liquid", "container", "place"],
                            "roles": {
                                "agent": {
                                    "framenet": "ingestor",
                                    "def": "The entity doing the drink action",
                                },
                                "liquid": {
                                    "framenet": "ingestibles",
                                    "def": "The entity that the agent is drinking",
                                },
                                "container": {"framenet": "source", "def": "source"},
                                "place": {"framenet": "place", "def": "location"},
                            },
                        }
                    },
                    "nouns": {
                        "n-woman": {"gloss": ["woman", "adult female"]},
                        "n-water": {"gloss": ["water"]},
                        "n-kitchen": {"gloss": ["kitchen"]},
                    },
                },
            )
            dataset = directory / "dataset.jsonl"
            rows = [
                {
                    "image_id": "swig:images_512/drinking_1.jpg",
                    "image_path": "cache/swig/one.jpg",
                    "subject": "person",
                    "action": "hold",
                    "object": "glass",
                    "adjuncts": "indoors",
                    "teacher_model": "teacher",
                },
                {
                    "image_id": "gqa:2",
                    "image_path": "cache/gqa/two.jpg",
                    "subject": "dog",
                    "action": "run",
                    "object": "none",
                    "adjuncts": "park",
                },
            ]
            dataset.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            action_map = directory / "actions.json"
            write_json(action_map, {"drinking": "drink"})
            output = directory / "corrected.jsonl"

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-jsonl",
                    str(dataset),
                    "--swig-json-dir",
                    str(swig),
                    "--output",
                    str(output),
                    "--action-map",
                    str(action_map),
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            result = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(
                [
                    result[0]["subject"],
                    result[0]["action"],
                    result[0]["object"],
                    result[0]["adjuncts"],
                ],
                ["woman", "drink", "water", "kitchen"],
            )
            self.assertEqual(result[0]["split"], "train")
            self.assertEqual(result[0]["vlm_original"]["action"], "hold")
            self.assertEqual(result[1], rows[1])

    def test_disagreement_preserves_vlm_value(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            swig = directory / "SWiG_jsons"
            swig.mkdir()
            write_json(
                swig / "train.json",
                {
                    "looking_9.jpg": {
                        "verb": "looking",
                        "bb": {"agent": [0, 0, 10, 10], "item": [20, 20, 30, 30]},
                        "frames": [
                            {"agent": "n1", "item": "n2"},
                            {"agent": "n3", "item": "n4"},
                            {"agent": "n5", "item": "n6"},
                        ],
                    }
                },
            )
            write_json(swig / "dev.json", {})
            write_json(swig / "test.json", {})
            write_json(
                swig / "imsitu_space.json",
                {
                    "verbs": {
                        "looking": {
                            "order": ["agent", "item"],
                            "roles": {
                                "agent": {"framenet": "agent", "def": "actor"},
                                "item": {"framenet": "theme", "def": "target"},
                            },
                        }
                    },
                    "nouns": {f"n{i}": {"gloss": [f"word{i}"]} for i in range(1, 7)},
                },
            )
            dataset = directory / "dataset.jsonl"
            row = {
                "image_id": "swig:looking_9.jpg",
                "image_path": "one.jpg",
                "subject": "person",
                "action": "look",
                "object": "screen",
                "adjuncts": "room",
            }
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = directory / "corrected.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-jsonl",
                    str(dataset),
                    "--swig-json-dir",
                    str(swig),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            corrected = json.loads(output.read_text())
            self.assertEqual(corrected["subject"], "person")
            self.assertEqual(corrected["object"], "screen")
            self.assertEqual(corrected["action"], "look")

    def test_ungrounded_object_does_not_replace_visible_vlm_target(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            swig = directory / "SWiG_jsons"
            swig.mkdir()
            write_json(
                swig / "train.json",
                {
                    "carrying_3.jpg": {
                        "verb": "carrying",
                        "bb": {"agent": [0, 0, 10, 10], "item": [-1, -1, -1, -1]},
                        "frames": [
                            {"agent": "n1", "item": "n2"},
                            {"agent": "n1", "item": "n2"},
                            {"agent": "n1", "item": "n2"},
                        ],
                    }
                },
            )
            write_json(swig / "dev.json", {})
            write_json(swig / "test.json", {})
            write_json(
                swig / "imsitu_space.json",
                {
                    "verbs": {
                        "carrying": {
                            "order": ["agent", "item"],
                            "roles": {
                                "agent": {"framenet": "agent", "def": "actor"},
                                "item": {"framenet": "theme", "def": "target"},
                            },
                        }
                    },
                    "nouns": {"n1": {"gloss": ["woman"]}, "n2": {"gloss": ["bag"]}},
                },
            )
            dataset = directory / "dataset.jsonl"
            row = {
                "image_id": "swig:carrying_3.jpg",
                "image_path": "one.jpg",
                "subject": "person",
                "action": "carry",
                "object": "box",
                "adjuncts": "none",
            }
            dataset.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = directory / "corrected.jsonl"
            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--dataset-jsonl",
                    str(dataset),
                    "--swig-json-dir",
                    str(swig),
                    "--output",
                    str(output),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            corrected = json.loads(output.read_text())
            self.assertEqual(corrected["subject"], "woman")
            self.assertEqual(corrected["object"], "box")


if __name__ == "__main__":
    unittest.main()
