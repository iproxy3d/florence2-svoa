import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from zipfile import ZipInfo

from PIL import Image

from scripts.download_svoa_images import (
    DownloadedImage,
    StateDB,
    build_parser,
    download_gqa,
    download_hagrid,
    download_remote_zip,
    ordered_image_members,
)


def jpeg_bytes():
    stream = BytesIO()
    Image.new("RGB", (8, 8), "white").save(stream, format="JPEG")
    return stream.getvalue()


def zip_info(name, offset):
    info = ZipInfo(name)
    info.header_offset = offset
    return info


class FakeArchive:
    def __init__(self, fail_on_first_open=False):
        self.items = [
            zip_info("late.jpg", 200),
            zip_info("early.jpg", 0),
        ]
        self.opened = []
        self.fail_on_first_open = fail_on_first_open
        self.closed = False

    def infolist(self):
        return self.items

    def open(self, name):
        self.opened.append(name)
        if self.fail_on_first_open:
            self.fail_on_first_open = False
            raise OSError("transient connection reset")
        return BytesIO(jpeg_bytes())

    def close(self):
        self.closed = True


class DownloadResumeTest(unittest.TestCase):
    def test_zip_uses_physical_offsets(self):
        infos = [
            zip_info("three.jpg", 30),
            zip_info("one.jpg", 10),
            zip_info("subfolder/", 0),
            zip_info("two.jpg", 20),
            zip_info("not_image.txt", 15),
        ]
        result = ordered_image_members(infos, seed=17, block_size=0)
        self.assertEqual(
            [info.filename for info in result],
            ["one.jpg", "two.jpg", "three.jpg"],
        )

    def test_block_shuffle_never_splits_physical_neighbors(self):
        infos = [
            zip_info(f"{index}.jpg", index * 10)
            for index in range(12)
        ]
        names = [
            info.filename
            for info in ordered_image_members(infos, seed=17, block_size=3)
        ]
        blocks = [names[index:index + 3] for index in range(0, 12, 3)]
        self.assertCountEqual(
            blocks,
            [
                ["0.jpg", "1.jpg", "2.jpg"],
                ["3.jpg", "4.jpg", "5.jpg"],
                ["6.jpg", "7.jpg", "8.jpg"],
                ["9.jpg", "10.jpg", "11.jpg"],
            ],
        )
        self.assertEqual(
            names,
            [
                info.filename
                for info in ordered_image_members(infos, seed=17, block_size=3)
            ],
        )

    def test_zip_retry_reopens_archive_and_keeps_saved_files(self):
        first = FakeArchive(fail_on_first_open=True)
        recovered = FakeArchive()
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            db = StateDB(root / "state.sqlite3")
            try:
                with (
                    patch(
                        "scripts.download_svoa_images.open_remote_archive",
                        side_effect=[(first, None), (recovered, None)],
                    ) as opener,
                    patch("scripts.download_svoa_images.time.sleep"),
                ):
                    download_remote_zip(
                        "swig", db, root, target=2, seed=17,
                        archive_url="https://example.test/a.zip", max_errors=20,
                        shuffle_block_size=0, member_retries=2,
                    )
                self.assertEqual(opener.call_count, 2)
                self.assertEqual(first.opened, ["early.jpg"])
                self.assertEqual(recovered.opened, ["early.jpg", "late.jpg"])
                self.assertTrue(first.closed)
                self.assertTrue(recovered.closed)
                self.assertEqual(db.success_count("swig"), 2)
                self.assertEqual(db.failed_count("swig"), 0)
                self.assertEqual(len(list((root / "swig").glob("*.jpg"))), 2)
                with patch(
                    "scripts.download_svoa_images.open_remote_archive",
                    side_effect=AssertionError("already cached"),
                ):
                    download_remote_zip(
                        "swig", db, root, target=2, seed=17,
                        archive_url="https://example.test/a.zip", max_errors=20,
                    )
            finally:
                db.close()

    def test_hagrid_continues_after_one_broken_class(self):
        args = build_parser().parse_args([
            "--hagrid-gestures", "like", "ok",
            "--hagrid-limit-per-gesture", "1",
        ])
        visited = []
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            db = StateDB(root / "state.sqlite3")
            try:
                def fake_download(dataset, db_arg, output_dir, target,
                                  seed, archive_url, max_errors, **kwargs):
                    gesture = kwargs["image_prefix"].rstrip("/")
                    visited.append(gesture)
                    if gesture == "like":
                        raise ConnectionError("like.zip unavailable")
                    destination = root / "ok.jpg"
                    destination.write_bytes(jpeg_bytes())
                    db_arg.record_success(DownloadedImage(
                        dataset="hagrid", image_id="ok/test.jpg",
                        source_ref=archive_url,
                        local_path=destination, width=8, height=8,
                    ))

                with patch(
                    "scripts.download_svoa_images.download_remote_zip",
                    side_effect=fake_download,
                ):
                    with self.assertRaisesRegex(RuntimeError, "like"):
                        download_hagrid(args, db, root)
                self.assertEqual(visited, ["like", "ok"])
                self.assertEqual(db.success_count("hagrid"), 1)
            finally:
                db.close()

    def test_gqa_retries_old_failed_rows_without_rewinding_cursor(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            db = StateDB(root / "state.sqlite3")
            try:
                old_image = root / "old.jpg"
                old_image.write_bytes(jpeg_bytes())
                db.record_success(DownloadedImage(
                    dataset="gqa", image_id="seen", source_ref="test",
                    local_path=old_image, width=8, height=8,
                ))
                db.record_failure(
                    "gqa", "retry", "hf://old", OSError("HTTP 429")
                )
                db.set_offset("gqa", 4)
                rows = [
                    {
                        "row_idx": 0,
                        "row": {
                            "id": "seen",
                            "image": {"src": "https://example.test/seen.jpg"},
                        },
                    },
                    {
                        "row_idx": 1,
                        "row": {
                            "id": "retry",
                            "image": {"src": "https://example.test/retry.jpg"},
                        },
                    },
                ]
                with (
                    patch(
                        "scripts.download_svoa_images.dataset_viewer_rows",
                        return_value=rows,
                    ) as viewer,
                    patch(
                        "scripts.download_svoa_images.fetch_remote_image",
                        return_value=Image.new("RGB", (8, 8), "white"),
                    ) as image_fetcher,
                ):
                    download_gqa(
                        db, root, target=2, dataset_name="test/data",
                        viewer_url="https://example.test/rows", timeout=1,
                        max_errors=50, max_consecutive_errors=5,
                    )
                self.assertEqual(viewer.call_args.args[3], 0)
                self.assertEqual(image_fetcher.call_count, 1)
                self.assertEqual(db.get_offset("gqa"), 4)
                self.assertEqual(db.success_count("gqa"), 2)
                self.assertEqual(db.failed_count("gqa"), 0)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
