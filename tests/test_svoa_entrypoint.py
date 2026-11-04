import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import svoa


class SvoaEntrypointTest(unittest.TestCase):
    def test_public_parser_has_exactly_four_commands(self):
        parser = svoa.build_parser()
        action = next(
            item
            for item in parser._actions
            if item.__class__.__name__ == "_SubParsersAction"
        )
        self.assertEqual(
            set(action.choices), {"download", "prepare", "train", "infer"}
        )

    def test_download_balances_hagrid_inside_total_dataset_budget(self):
        args = svoa.build_parser().parse_args(
            ["download", "--limit-per-dataset", "5000"]
        )
        with tempfile.TemporaryDirectory() as raw_directory:
            args.output_dir = Path(raw_directory)
            with patch.object(svoa, "run_script") as run, patch.object(
                svoa, "download_swig_annotations"
            ):
                svoa.command_download(args, [])
        command = [str(value) for value in run.call_args.args[1]]
        index = command.index("--hagrid-limit-per-gesture")
        self.assertEqual(command[index + 1], "147")

    def test_prepare_reports_missing_download_instead_of_deep_traceback(self):
        args = svoa.build_parser().parse_args(["prepare"])
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            args.download_dir = root / "missing"
            args.work_dir = root / "work"
            with self.assertRaisesRegex(FileNotFoundError, "svoa.py download"):
                svoa.command_prepare(args, [])


if __name__ == "__main__":
    unittest.main()
