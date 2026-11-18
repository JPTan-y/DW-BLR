import csv
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulation import generate_simulation_data as generator


class SimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = generator.read_seed_table()
        cls.task = cls.tasks[0]
        cls.train, cls.tests = generator.generate_task_data(cls.task, n_test=80)

    def test_default_protocol(self):
        protocol = generator.simulation_protocol()
        self.assertEqual(len(protocol["tasks"]), 300)
        self.assertEqual(protocol["sample_sizes"], [1500, 3000, 5000])
        self.assertEqual(protocol["repetitions"], list(range(1, 101)))
        self.assertEqual(protocol["environments"], list(generator.ENVIRONMENTS))
        self.assertFalse(protocol["seed_filtering_by_this_script"])
        json.dumps(protocol, allow_nan=False)

    def test_protocol_subset_and_order(self):
        protocol = generator.simulation_protocol(
            n_test=80, sample_sizes=[5000, 1500], repetitions=[2, 1])
        self.assertEqual(
            [(t["n_train"], t["replication"]) for t in protocol["tasks"]],
            [(1500, 1), (1500, 2), (5000, 1), (5000, 2)])

    def test_protocol_rejects_invalid_options(self):
        for options in (
            dict(n_test=1), dict(n_jobs=0), dict(n_jobs=generator.safe_worker_limit()+1),
            dict(sample_sizes=[]), dict(sample_sizes=[100]),
            dict(sample_sizes=[1500.5]), dict(repetitions=[0]),
            dict(repetitions=[101]), dict(repetitions=[True]),
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                generator.simulation_protocol(**options)

    def test_seed_schema(self):
        with generator.DEFAULT_SEED_FILE.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            self.assertEqual(rows.fieldnames,
                ["n_train", "replication", "train_seed"] +
                [f"test_seed_{r:g}" for r in generator.ENVIRONMENTS])
            values = list(rows)
        self.assertEqual(len(values), 300)
        self.assertEqual(self.task["train_seed"], 2960789117)
        seeds = [s for task in self.tasks for s in [task["train_seed"], *task["test_seeds"]]]
        self.assertEqual(len(seeds), len(set(seeds)))

    def test_generated_shapes_and_environments(self):
        self.assertEqual(self.train["X"].shape, (1500, 30))
        self.assertEqual(self.train["y"].shape, (1500,))
        self.assertEqual(list(self.tests), list(generator.ENVIRONMENTS))
        for environment, data in self.tests.items():
            self.assertEqual(data["X"].shape, (80, 30))
            self.assertEqual(data["y"].shape, (80,))
            self.assertEqual(float(data["environment"]), environment)
            self.assertTrue(np.isfinite(data["X"]).all())
            self.assertTrue(np.isin(data["y"], [0, 1]).all())

    def test_data_stream_is_repeatable_and_global_rng_independent(self):
        np.random.seed(123)
        before = np.random.get_state()
        repeated = generator.generate_dataset(
            self.task["n_train"], 2., self.task["train_seed"])
        after = np.random.get_state()
        np.testing.assert_array_equal(self.train["X"], repeated["X"])
        np.testing.assert_array_equal(self.train["y"], repeated["y"])
        for a, b in zip(before, after):
            np.testing.assert_equal(a, b)

    def test_summary_is_json_serializable_and_hashes_observations(self):
        summary = generator.describe_task_data(self.task, self.train, self.tests)
        json.dumps(summary, allow_nan=False)
        self.assertEqual(len(summary["datasets"]), 9)
        self.assertEqual(summary["n_test"], 80)
        self.assertTrue(summary["independent_r2_verified"])
        first = summary["datasets"][0]
        self.assertEqual(first["n"], 1500)
        self.assertEqual(first["seed"], self.task["train_seed"])
        self.assertEqual(first["positive_count"], int(self.train["y"].sum()))
        self.assertEqual(first["X_sha256"],
            hashlib.sha256(np.ascontiguousarray(self.train["X"]).tobytes()).hexdigest())

    def test_duplicate_or_missing_test_seeds_are_rejected(self):
        for seeds in (self.task["test_seeds"][:-1],
                      [self.task["train_seed"], *self.task["test_seeds"][1:]]):
            with self.assertRaises(ValueError):
                generator.generate_task_data(dict(self.task, test_seeds=seeds), 80)

    def test_summary_rejects_missing_or_leaking_environments(self):
        missing = dict(self.tests)
        missing.pop(2.)
        with self.assertRaises(ValueError):
            generator.describe_task_data(self.task, self.train, missing)
        leaking = dict(self.tests)
        leaking[2.] = dict(self.tests[2.], X=self.train["X"][:80].copy())
        with self.assertRaises(ValueError):
            generator.describe_task_data(self.task, self.train, leaking)

    def test_seed_reader_rejects_missing_fields_and_duplicate_tasks(self):
        with generator.DEFAULT_SEED_FILE.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            fields, rows = reader.fieldnames, list(reader)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seeds.csv"
            for columns, values in ((fields[1:], rows), (fields, rows + [rows[0]])):
                with path.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(values)
                with self.assertRaises(ValueError):
                    generator.read_seed_table(path)


if __name__ == "__main__":
    unittest.main()
