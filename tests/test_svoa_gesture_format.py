import unittest

from scripts.infer_florence2_svoa import parse_prediction
from scripts.prepare_svoa_dataset import validate_svoa
from scripts.train_florence2_svoa import normalize_prediction, normalize_word


class SvoaGestureFormatTest(unittest.TestCase):
    def test_prepare_accepts_multiword_action_only(self):
        value = validate_svoa(
            {
                "subject": "person",
                "action": "thumb index2",
                "object": "none",
                "adjuncts": "camera",
            }
        )
        self.assertEqual(value["action"], "thumb index2")
        with self.assertRaises(ValueError):
            validate_svoa(
                {
                    "subject": "two people",
                    "action": "like",
                    "object": "none",
                    "adjuncts": "camera",
                }
            )

    def test_training_normalizer_preserves_action_spaces(self):
        self.assertEqual(
            normalize_word("  little   finger ", "action", 1),
            "little finger",
        )
        with self.assertRaises(ValueError):
            normalize_word("two people", "subject", 1)
        self.assertEqual(
            normalize_prediction("person|little finger|none|camera"),
            "person|little finger|none|camera",
        )

    def test_inference_parser_preserves_action_spaces(self):
        result = parse_prediction(
            "person|thumb   index|none|camera",
            ("subject", "action", "object", "adjuncts"),
        )
        self.assertTrue(result["valid_format"])
        self.assertEqual(result["action"], "thumb index")


if __name__ == "__main__":
    unittest.main()
