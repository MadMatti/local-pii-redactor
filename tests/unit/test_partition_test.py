import copy
import hashlib
import json

import pytest

from scripts.evaluation.partition_test import PartitionError, main, partition_test


def _row(sample_id, *, negative=False):
    spans = [] if negative else [{"type": "EMAIL", "text": "private@example.test", "start": 0, "end": 20}]
    return {
        "sample_id": sample_id,
        "messages": [
            {"role": "user", "content": "private@example.test"},
            {"role": "assistant", "content": json.dumps({"entities": spans})},
        ],
        "metadata": {"is_negative": negative, "entity_spans": spans},
    }


def _write(path, rows, *, sort_keys=False):
    path.write_text(
        "".join(json.dumps(row, sort_keys=sort_keys) + "\n" for row in rows),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inputs(tmp_path, rows=None, prior=None):
    rows = [_row("one"), _row("two", negative=True), _row("three")] if rows is None else rows
    prior = [rows[2], rows[0]] if prior is None else prior
    dataset = tmp_path / "full.jsonl"
    previous = tmp_path / "prior.jsonl"
    digest = _write(dataset, rows)
    _write(previous, prior, sort_keys=True)
    return dataset, previous, tmp_path / "views", digest


def _run(inputs):
    dataset, previous, output, digest = inputs
    return partition_test(dataset, previous, output, expected_dataset_sha256=digest)


def test_partition_is_disjoint_exhaustive_ordered_and_content_preserving(tmp_path):
    inputs = _inputs(tmp_path)
    manifest = _run(inputs)
    output = inputs[2]
    groups = {
        name: [json.loads(line) for line in (output / item["path"]).read_text().splitlines()]
        for name, item in manifest["outputs"].items()
    }
    assert [row["sample_id"] for row in groups["previously_used"]] == ["one", "three"]
    assert [row["sample_id"] for row in groups["not_previously_used"]] == ["two"]
    assert groups["previously_used"] == [_row("one"), _row("three")]
    assert manifest["outputs"]["previously_used"]["entity_counts"] == {"EMAIL": 2}
    assert manifest["outputs"]["not_previously_used"]["negative_count"] == 1
    assert manifest["exhaustive"] and manifest["disjoint"]
    assert "not an untouched holdout" in manifest["limitation"]
    for item in manifest["outputs"].values():
        assert item["sha256"] == hashlib.sha256((output / item["path"]).read_bytes()).hexdigest()


def test_partition_rejects_missing_prior_ids(tmp_path):
    inputs = _inputs(tmp_path, prior=[_row("absent")])
    with pytest.raises(PartitionError, match="not a subset"):
        _run(inputs)
    assert not inputs[2].exists()


@pytest.mark.parametrize("change", ["messages", "metadata"])
def test_partition_rejects_changed_content_for_shared_ids(tmp_path, change):
    full_row = _row("one")
    changed = copy.deepcopy(full_row)
    if change == "messages":
        changed["messages"][0]["content"] = "changed confidential text"
    else:
        changed["metadata"]["extra"] = "changed confidential metadata"
    with pytest.raises(PartitionError, match="changed record content"):
        _run(_inputs(tmp_path, rows=[full_row], prior=[changed]))


@pytest.mark.parametrize("duplicate_in", ["full", "prior"])
def test_partition_rejects_duplicate_ids(tmp_path, duplicate_in):
    row = _row("secret-id")
    inputs = _inputs(
        tmp_path,
        rows=[row, row] if duplicate_in == "full" else [row],
        prior=[row, row] if duplicate_in == "prior" else [row],
    )
    with pytest.raises(PartitionError, match="duplicate sample_id") as exc:
        _run(inputs)
    assert "secret-id" not in str(exc.value)


def test_partition_checks_frozen_hash_before_writing(tmp_path):
    dataset, previous, output, _ = _inputs(tmp_path)
    with pytest.raises(PartitionError, match="SHA-256"):
        partition_test(dataset, previous, output, expected_dataset_sha256="0" * 64)
    assert not output.exists()


def test_partition_rerun_verifies_identical_bytes_without_rewriting(tmp_path):
    inputs = _inputs(tmp_path)
    first = _run(inputs)
    before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in inputs[2].iterdir()}
    assert _run(inputs) == first
    after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in inputs[2].iterdir()}
    assert before == after


@pytest.mark.parametrize("tamper", ["output", "manifest", "input"])
def test_partition_refuses_to_replace_changed_outputs_or_inputs(tmp_path, tamper):
    inputs = _inputs(tmp_path)
    _run(inputs)
    if tamper == "input":
        _write(inputs[1], [_row("one")])
    else:
        target = inputs[2] / ("manifest.json" if tamper == "manifest" else "previously_used.jsonl")
        target.write_text("tampered\n", encoding="utf-8")
    before = {p.name: p.read_bytes() for p in inputs[2].iterdir()}
    with pytest.raises(PartitionError, match="existing output differs"):
        _run(inputs)
    assert {p.name: p.read_bytes() for p in inputs[2].iterdir()} == before


def test_partition_cli_logs_only_counts_paths_and_sanitized_errors(tmp_path, capsys):
    dataset, previous, output, digest = _inputs(tmp_path)
    args = [
        "--dataset", str(dataset), "--previously-used-dataset", str(previous),
        "--output-dir", str(output), "--expected-dataset-sha256", digest,
    ]
    assert main(args) == 0
    previous.write_text('{"sample_id":"private@example.test",broken}\n', encoding="utf-8")
    assert main(args) == 1
    captured = capsys.readouterr()
    assert "private@example.test" not in captured.out + captured.err
    assert "invalid JSON" in captured.err


def test_partition_accepts_empty_prior_and_existing_empty_directory(tmp_path):
    inputs = _inputs(tmp_path, prior=[])
    inputs[2].mkdir()
    manifest = _run(inputs)
    assert manifest["outputs"]["previously_used"]["record_count"] == 0
    assert manifest["outputs"]["not_previously_used"]["record_count"] == 3


def test_length_partition_labels_and_complement_are_explicit(tmp_path):
    dataset, reference, output, digest = _inputs(tmp_path)
    manifest = partition_test(dataset, reference, output, expected_dataset_sha256=digest, partition_role="length_filter")
    assert manifest["partition_role"] == "length_filter"
    assert set(manifest["outputs"]) == {"retained", "excluded"}
    assert manifest["outputs"]["excluded"]["record_count"] == 1
    assert manifest["outputs"]["excluded"]["negative_count"] == 1
    assert "no tokenizer" in manifest["limitation"]
    assert [json.loads(s)["sample_id"] for s in (output / "excluded.jsonl").read_text().splitlines()] == ["two"]
    with pytest.raises(PartitionError, match="existing output differs"):
        partition_test(dataset, reference, output, expected_dataset_sha256=digest)
