import unittest

from florence2_svoa import SVOAEvent, parse_generation


class SchemaTest(unittest.TestCase):
    def test_pipe_roundtrip(self):
        event = SVOAEvent.from_text("person|thumb index2|none|camera")
        self.assertEqual(event.action, "thumb index2")
        self.assertEqual(event.to_text(), "person|thumb index2|none|camera")

    def test_json_output(self):
        result = parse_generation(
            '{"subject":"dog","action":"run","object":"none","adjuncts":"park"}'
        )
        self.assertTrue(result.valid)
        self.assertEqual(result.event.subject, "dog")

    def test_invalid_output_is_not_filled_in(self):
        result = parse_generation("person|run|park")
        self.assertFalse(result.valid)
        self.assertIsNone(result.event)


if __name__ == "__main__":
    unittest.main()
