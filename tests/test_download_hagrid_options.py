import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.download_svoa_images import (
    HAGRID_GESTURES,
    StateDB,
    build_parser,
    hagrid_archive_url,
    hagrid_selected_success_count,
    register_manual_images,
)


class DownloadHagridOptionsTest(unittest.TestCase):
    def test_default_is_only_no_gesture(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.hagrid_gestures, ["no_gesture"])
        self.assertIn("like", HAGRID_GESTURES)
        self.assertIn("thumb_index2", HAGRID_GESTURES)

    def test_archive_template_preserves_official_class_name(self):
        args = argparse.Namespace(
            hagrid_url=None,
            hagrid_gestures=["little_finger"],
            hagrid_url_template="https://example.test/{gesture}.zip",
        )
        self.assertEqual(
            hagrid_archive_url(args, "little_finger"),
            "https://example.test/little_finger.zip",
        )

    def test_custom_url_requires_one_selected_class(self):
        args = argparse.Namespace(
            hagrid_url="https://example.test/custom.zip",
            hagrid_gestures=["like", "ok"],
            hagrid_url_template="https://example.test/{gesture}.zip",
        )
        with self.assertRaises(ValueError):
            hagrid_archive_url(args, "like")

    def test_selected_count_ignores_other_classes(self):
        with tempfile.TemporaryDirectory() as raw_directory:
            db = StateDB(Path(raw_directory) / "state.sqlite3")
            try:
                now = "2020-01-01T00:00:00+00:00"
                db.connection.executemany(
                    """
                    INSERT INTO downloads(
                        dataset, image_id, source_ref, status, attempts,
                        created_at, updated_at
                    ) VALUES('hagrid', ?, 'test', 'done', 1, ?, ?)
                    """,
                    [
                        ("like/a.jpg", now, now),
                        ("ok/b.jpg", now, now),
                        ("dislike/c.jpg", now, now),
                    ],
                )
                db.connection.commit()
                self.assertEqual(
                    hagrid_selected_success_count(db, ["like", "ok"]), 2
                )
            finally:
                db.close()

    def test_manual_class_directory_counts_toward_gesture_quota(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            image_path = root / "hagrid" / "little_finger" / "abc.jpg"
            image_path.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), "white").save(image_path)
            db = StateDB(root / "state.sqlite3")
            try:
                self.assertEqual(register_manual_images("hagrid", root, db), 1)
                self.assertEqual(
                    hagrid_selected_success_count(db, ["little_finger"]), 1
                )
                self.assertIn(
                    "little_finger/manual:little_finger/abc.jpg",
                    db.successful_ids("hagrid"),
                )
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
