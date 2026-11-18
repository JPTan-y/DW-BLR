from pathlib import Path
from decimal import Decimal, InvalidOperation
import argparse
import csv
import hashlib
import json
import os
import platform
import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

FEATURES = [f"S{i}" for i in range(1, 6)] + [f"V{i}" for i in range(1, 26)]
BETA = np.r_[1/3, -2/3, 1., -1/3, 2/3, np.zeros(25)]
ENVIRONMENTS = (-3., -2.5, -2., -1.5, 1.5, 2., 2.5, 3.)
SAMPLE_SIZES = (1500, 3000, 5000)
DEFAULT_SEED_FILE = Path(__file__).resolve().parent / "seed" / "seed_table.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "simulation_data"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def covariance():
    value = np.full((30, 30), .2)
    value[:15, :15] = .5
    value[:5, :5] = .8
    np.fill_diagonal(value, 1.)
    if not np.allclose(value, value.T) or np.linalg.eigvalsh(value).min() <= 0:
        raise ValueError("Covariance must be symmetric positive definite")
    return value


def _integer(value, label, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(f"Invalid {label}: {value!r}")
    try:
        number = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise ValueError(f"Invalid {label}: {value!r}") from exc
    if not number.is_finite() or number != number.to_integral_value():
        raise ValueError(f"{label} must be an integer")
    result = int(number)
    if result < minimum or (maximum is not None and result > maximum):
        raise ValueError(f"{label} is out of range")
    return result


def read_seed_table(path=DEFAULT_SEED_FILE, expected_repeats=100):
    expected_repeats = _integer(expected_repeats, "expected_repeats", 1)
    test_columns = [f"test_seed_{r:g}" for r in ENVIRONMENTS]
    required = ["n_train", "replication", "train_seed"] + test_columns
    tasks = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        rows = csv.DictReader(stream)
        if rows.fieldnames is None:
            raise ValueError("Empty seed table")
        header = [str(value).strip() for value in rows.fieldnames]
        if any(header.count(key) != 1 for key in required):
            raise ValueError("Seed table columns must contain each required field exactly once: " + str(required))
        rows.fieldnames = header
        for row_number, record in enumerate(rows, 2):
            if all(value is None or isinstance(value, str) and not value.strip() for value in record.values()):
                continue
            if None in record:
                raise ValueError(f"row {row_number}: too many CSV fields")
            n = _integer(record["n_train"], f"row {row_number}: n_train", 2)
            rep = _integer(record["replication"], f"row {row_number}: replication", 1)
            task = dict(n_train=n, replication=rep)
            task["train_seed"] = _integer(record["train_seed"], f"row {row_number}: train_seed", 0, 2**32-1)
            task["test_seeds"] = [_integer(record[key], f"row {row_number}: {key}", 0, 2**32-1) for key in test_columns]
            tasks.append(task)
    keys = [(t["n_train"], t["replication"]) for t in tasks]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate sample-size/replication rows")
    if sorted({t["n_train"] for t in tasks}) != list(SAMPLE_SIZES):
        raise ValueError(f"Expected sample sizes {SAMPLE_SIZES}")
    for n in SAMPLE_SIZES:
        if sorted(t["replication"] for t in tasks if t["n_train"] == n) != list(range(1, expected_repeats+1)):
            raise ValueError(f"n={n} must contain repetitions 1 through {expected_repeats}")
    data_seeds = [seed for t in tasks for seed in [t["train_seed"]]+t["test_seeds"]]
    if len(set(data_seeds)) != len(data_seeds):
        raise ValueError("Training and test seeds must be distinct")
    return sorted(tasks, key=lambda t: (t["n_train"], t["replication"]))


def generate_dataset(n, r, seed):
    n = _integer(n, "n", 2)
    seed = _integer(seed, "seed", 0, 2**32-1)
    r = float(r)
    if not np.isfinite(r) or abs(r) < 1:
        raise ValueError("Rejection environment requires finite |r| >= 1")
    sx, sy = np.random.SeedSequence(seed).spawn(2)
    rx, ry = np.random.default_rng(sx), np.random.default_rng(sy)
    parts = []
    remaining = n
    proposed = accepted = 0
    sigma = covariance()
    with threadpool_limits(limits=1):
        while remaining:
            batch = min(max(5000, 5*remaining), 2000*n-proposed)
            if batch <= 0:
                raise RuntimeError("Rejection sampling candidate budget exhausted")
            x = rx.multivariate_normal(np.zeros(30), sigma, size=batch, check_valid="raise")
            h = x@BETA + .5*np.prod(x[:, :3], axis=1)
            distance = np.abs(h[:, None] - np.sign(r)*x[:, [9, 14]]).sum(axis=1)
            keep = np.log(np.maximum(rx.random(batch), np.finfo(float).tiny)) < -5*np.log(abs(r))*distance
            accepted += int(keep.sum())
            proposed += batch
            take = x[keep][:remaining]
            parts.append(take)
            remaining -= len(take)
        x = np.vstack(parts)
        h = x@BETA + .5*np.prod(x[:, :3], axis=1)
        probability = expit(-.5+h)
    return dict(X=x, y=ry.binomial(1, probability).astype(int), true_probability=probability,
                h=h, seed=np.uint32(seed), environment=np.float64(r),
                candidate_count=np.int64(proposed), acceptance_rate=np.float64(accepted/proposed))


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    os.replace(temporary, path)


def _ensure_dataset(path, n, r, seed, expected_hash=None):
    if path.exists():
        if expected_hash is not None and sha256(path) != expected_hash:
            raise ValueError(f"Saved data hash mismatch: {path}")
        with np.load(path, allow_pickle=False) as data:
            if data["X"].shape != (n, 30) or data["y"].shape != (n,) or int(data["seed"]) != seed or float(data["environment"]) != r:
                raise ValueError(f"Saved data does not match requested seed/configuration: {path}")
            if not np.isfinite(data["X"]).all() or not np.isin(data["y"], [0, 1]).all():
                raise ValueError(f"Invalid saved observations: {path}")
        return sha256(path)
    data = generate_dataset(n, r, seed)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **data)
    os.replace(temporary, path)
    return sha256(path)


def generate_task(task, output_dir, n_test=2000):
    folder = Path(output_dir)/f"n{task['n_train']}"/f"rep_{task['replication']:03d}"
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder/"data_record.json"
    old = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    if old and (old["task"] != task or old["n_test"] != n_test):
        raise ValueError(f"Data configuration changed: {folder}")
    specs = [("train.npz", task["n_train"], 2., task["train_seed"])]
    specs += [(f"test_{i:02d}.npz", n_test, r, seed) for i, (r, seed) in enumerate(zip(ENVIRONMENTS, task["test_seeds"]))]
    hashes = {}
    for name, n, r, seed in specs:
        hashes[name] = _ensure_dataset(folder/name, n, r, seed, old.get("sha256", {}).get(name))
    with np.load(folder/"train.npz") as train, np.load(folder/"test_05.npz") as test:
        count = min(task["n_train"], n_test)
        if np.array_equal(train["X"][:count], test["X"][:count]):
            raise ValueError("The r=2 test set duplicates the training data")
    record = dict(task=task, n_test=n_test, environments=list(ENVIRONMENTS), sha256=hashes,
                  independent_r2_verified=True)
    _atomic_json(record_path, record)
    return record


def safe_worker_limit():
    import psutil
    logical = os.cpu_count() or 1
    physical = psutil.cpu_count(logical=False) or logical
    return max(1, min(int(.85*logical), int(.85*physical)))


def simulation_protocol(seed_file=DEFAULT_SEED_FILE, n_test=2000,
                        sample_sizes=SAMPLE_SIZES, repetitions=None, n_jobs=1):
    n_test = _integer(n_test, "n_test", 2)
    n_jobs = _integer(n_jobs, "n_jobs", 1)
    if n_jobs > safe_worker_limit():
        raise ValueError(f"n_jobs exceeds 85% CPU limit: {safe_worker_limit()}")
    wanted_sizes = sorted({_integer(n, "sample size", 2) for n in sample_sizes})
    wanted_reps = sorted({_integer(rep, "replication", 1, 100)
                          for rep in (range(1, 101) if repetitions is None else repetitions)})
    if not wanted_sizes or not set(wanted_sizes).issubset(SAMPLE_SIZES) or not wanted_reps:
        raise ValueError("Requested sample sizes/repetitions are outside the seed table")
    tasks = [task for task in read_seed_table(seed_file)
             if task["n_train"] in wanted_sizes and task["replication"] in wanted_reps]
    if len(tasks) != len(wanted_sizes) * len(wanted_reps):
        raise ValueError("The seed table does not cover every requested task")
    return dict(schema_version=2, seed_table_sha256=sha256(seed_file),
                generator_sha256=sha256(__file__), n_test=n_test,
                sample_sizes=wanted_sizes, repetitions=wanted_reps,
                features=FEATURES, covariance=covariance().tolist(), beta=BETA.tolist(),
                intercept=-.5, interaction_coefficient=.5, r_train=2.,
                environments=list(ENVIRONMENTS), tasks=tasks, data_storage="memory",
                seed_filtering_by_this_script=False)


def generate_task_data(task, n_test=2000):
    n_test = _integer(n_test, "n_test", 2)
    n_train = _integer(task["n_train"], "n_train", 2)
    train_seed = _integer(task["train_seed"], "train_seed", 0, 2**32-1)
    test_seeds = [_integer(seed, "test_seed", 0, 2**32-1) for seed in task["test_seeds"]]
    if len(test_seeds) != len(ENVIRONMENTS):
        raise ValueError("Expected one seed per test environment")
    if len(set([train_seed, *test_seeds])) != len(ENVIRONMENTS) + 1:
        raise ValueError("Training and test seeds must be distinct")
    train = generate_dataset(n_train, 2., train_seed)
    tests = {r: generate_dataset(n_test, r, seed)
             for r, seed in zip(ENVIRONMENTS, test_seeds)}
    count = min(n_train, n_test)
    if np.array_equal(train["X"][:count], tests[2.]["X"][:count]):
        raise ValueError("The r=2 test set duplicates the training data")
    return train, tests


def describe_task_data(task, train, tests):
    if set(tests) != set(ENVIRONMENTS):
        raise ValueError("Expected all eight test environments")
    n_test = len(tests[ENVIRONMENTS[0]]["y"])
    if len(train["y"]) != task["n_train"] or any(len(d["y"]) != n_test for d in tests.values()):
        raise ValueError("Dataset sizes do not match the task")
    count = min(len(train["y"]), n_test)
    if np.array_equal(train["X"][:count], tests[2.]["X"][:count]):
        raise ValueError("The r=2 test set duplicates the training data")

    def describe(role, data):
        X, y = np.asarray(data["X"]), np.asarray(data["y"])
        if X.shape != (len(y), len(FEATURES)) or y.ndim != 1:
            raise ValueError("Unexpected dataset shape")
        if not np.isfinite(X).all() or not np.isin(y, [0, 1]).all():
            raise ValueError("Invalid observations")
        return dict(role=role, n=len(y), r=float(data["environment"]),
                    seed=int(data["seed"]), positive_count=int(y.sum()),
                    candidate_count=int(data["candidate_count"]),
                    acceptance_rate=float(data["acceptance_rate"]),
                    X_sha256=hashlib.sha256(np.ascontiguousarray(X).tobytes()).hexdigest(),
                    y_sha256=hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest())

    json_task = {key: int(value) if isinstance(value, (int, np.integer)) else value
                 for key, value in task.items() if key != "test_seeds"}
    json_task["test_seeds"] = [int(seed) for seed in task["test_seeds"]]
    return dict(task=json_task, data_storage="memory", n_test=n_test,
                independent_r2_verified=True,
                datasets=[describe("train", train)] +
                         [describe("test", tests[r]) for r in ENVIRONMENTS])


def generate_all_data(seed_file=DEFAULT_SEED_FILE, output_dir=DEFAULT_OUTPUT, *, n_test=2000,
                      sample_sizes=SAMPLE_SIZES, repetitions=None, n_jobs=1):
    n_test = _integer(n_test, "n_test", 2)
    n_jobs = _integer(n_jobs, "n_jobs", 1)
    if n_jobs > safe_worker_limit():
        raise ValueError(f"n_jobs exceeds 85% CPU limit: {safe_worker_limit()}")
    tasks = read_seed_table(seed_file)
    wanted_sizes = set(int(v) for v in sample_sizes)
    wanted_reps = set(range(1, 101) if repetitions is None else (int(v) for v in repetitions))
    if not wanted_sizes or not wanted_sizes.issubset(SAMPLE_SIZES) or not wanted_reps or not wanted_reps.issubset(range(1, 101)):
        raise ValueError("Requested sample sizes/repetitions are outside the seed table")
    tasks = [t for t in tasks if t["n_train"] in wanted_sizes and t["replication"] in wanted_reps]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = dict(schema_version=1, seed_table_sha256=sha256(seed_file), generator_sha256=sha256(__file__),
                    n_test=n_test, sample_sizes=sorted(wanted_sizes), repetitions=sorted(wanted_reps),
                    features=FEATURES, covariance=covariance().tolist(), beta=BETA.tolist(),
                    intercept=-.5, interaction_coefficient=.5, r_train=2., environments=list(ENVIRONMENTS),
                    acceptance="exp(-5*log(abs(r))*(abs(h-sign(r)*V5)+abs(h-sign(r)*V10)))",
                    tasks=tasks, seed_filtering_by_this_script=False)
    manifest_path = output/"data_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != protocol:
        raise ValueError("Existing data directory uses a different protocol; choose a new output directory")
    _atomic_json(manifest_path, protocol)
    if n_jobs == 1:
        records = []
        for i, task in enumerate(tasks, 1):
            records.append(generate_task(task, output, n_test))
            print(f"Data {i}/{len(tasks)}: n={task['n_train']}, repetition={task['replication']}", flush=True)
    else:
        from joblib import Parallel, delayed, parallel_config
        with parallel_config(backend="loky", inner_max_num_threads=1):
            records = Parallel(n_jobs=n_jobs, verbose=10)(delayed(generate_task)(t, output, n_test) for t in tasks)
    _atomic_json(output/"generation_audit.json", dict(datasets=len(records)*9, tasks=len(records),
                 all_independent_r2=True, records=records,
                 runtime=dict(python=platform.python_version(), numpy=np.__version__)))
    return protocol


def load_task_data(task, data_dir):
    folder = Path(data_dir)/f"n{task['n_train']}"/f"rep_{task['replication']:03d}"
    record = json.loads((folder/"data_record.json").read_text(encoding="utf-8"))
    if record["task"] != task:
        raise ValueError("Data record/seed table mismatch")
    result = {}
    for name, digest in record["sha256"].items():
        if sha256(folder/name) != digest:
            raise ValueError(f"Data hash mismatch: {folder/name}")
        with np.load(folder/name, allow_pickle=False) as arrays:
            result[name] = {key: arrays[key].copy() for key in arrays.files}
    return result["train.npz"], {r: result[f"test_{i:02d}.npz"] for i,r in enumerate(ENVIRONMENTS)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--first", type=int, default=100)
    args = parser.parse_args()
    generate_all_data(args.seeds, args.output, n_test=args.n_test, n_jobs=args.workers,
                      repetitions=range(1, args.first+1))
