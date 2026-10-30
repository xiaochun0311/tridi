from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from tridi.utils.exp import init_exp


class ExperimentRootRegression(unittest.TestCase):
    def test_cli_output_root_matches_created_experiment_for_himo_and_legacy(self):
        for dataset in ("himo", "grab"):
            with self.subTest(dataset=dataset), TemporaryDirectory() as temp:
                root = Path(temp)
                config = root / "base.yaml"
                config.write_text("env:\n  experiments_folder: " + str(root / "default") + "\n")
                cfg = init_exp(SimpleNamespace(config=[str(config)], overrides=[
                    "env.experiments_folder=" + str(root / "selected"),
                    "run.name=output_root_regression", "run.job=inspect", f"run.datasets=[{dataset}]"]))
                experiment = Path(cfg.run.path)
                self.assertTrue(experiment.is_dir())
                self.assertEqual(experiment.parent.resolve(), (root / "selected").resolve())
                self.assertTrue((experiment / "checkpoints").is_dir())
                self.assertTrue((experiment / "sample_config.yaml").is_file())


if __name__ == "__main__":
    unittest.main()
