import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from pii_redactor.deployment import diagnostics as d
from scripts.deployment import diagnose_pi as cli


MEMINFO = "MemTotal: 8000000 kB\nMemAvailable: 6000000 kB\nSwapTotal: 1024 kB\nSwapFree: 1024 kB\n"


@pytest.fixture
def pi(tmp_path, monkeypatch):
    files = {"proc/meminfo": MEMINFO,
             "etc/os-release": 'ID=debian\nVERSION_ID="12"\nHOSTNAME=private-canary\n',
             "sys/firmware/devicetree/base/model": "Raspberry Pi 4 Model B Rev 1.4\x00",
             "proc/cpuinfo": "Serial: private-canary\n"}
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    monkeypatch.setattr(d.platform, "system", lambda: "Linux")
    monkeypatch.setattr(d.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(d.platform, "python_version", lambda: "3.11.15")
    monkeypatch.setattr(d.struct, "calcsize", lambda _: 8)
    monkeypatch.setattr(d.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(d.shutil, "disk_usage", lambda _: SimpleNamespace(total=64 * 2**30, free=20 * 2**30))
    monkeypatch.setattr(d, "firmware_query", lambda command: "temp=45.5'C\n" if command == "measure_temp" else "throttled=0x0\n")
    return tmp_path


def test_meminfo_uses_available_not_free_and_binary_kilobytes():
    memory = d.parse_meminfo(MEMINFO + "MemFree: 1 kB\n")
    assert memory["available_bytes"] == 6000000 * 1024
    assert memory["total_bytes"] == 8000000 * 1024
    assert memory["swap_used_bytes"] == 0


@pytest.mark.parametrize("text", ["", MEMINFO.replace("MemAvailable", "Missing"),
    MEMINFO.replace("6000000", "9000000"), MEMINFO.replace("kB", "MB"),
    MEMINFO.replace("SwapFree: 1024", "SwapFree: 2048"), MEMINFO + "MemTotal: 1 kB\n",
    MEMINFO.replace("6000000", "-1")])
def test_rejects_missing_malformed_or_inconsistent_memory(text):
    with pytest.raises(ValueError):
        d.parse_meminfo(text)


def test_current_and_historical_throttling_are_distinct():
    result = d.parse_throttled("throttled=0x50005\n")
    assert result["active"] == ["undervoltage", "throttled"]
    assert result["historical"] == ["undervoltage", "throttled"]
    assert result["unknown_bits"] == "0x0"
    assert d.parse_throttled("throttled=0x10")["unknown_bits"] == "0x10"
    assert d.parse_throttled("throttled=0x50000")["active"] == []


@pytest.mark.parametrize("text", ["", "0x0", "throttled=0x100000000", "throttled=0x0 private-canary"])
def test_bad_firmware_responses_are_not_healthy(text):
    with pytest.raises(ValueError):
        d.parse_throttled(text)


@pytest.mark.parametrize("text", ["", "temp=nan'C", "temp=151'C", "temp=40'F", "temp=40'C private-canary"])
def test_temperature_requires_valid_celsius(text):
    with pytest.raises(ValueError):
        d.parse_temperature(text)


def test_complete_inventory_is_never_benchmark_or_deployment_approval(pi):
    report = d.collect_diagnostics(pi, root=pi)
    assert report["status"] == "diagnostics_complete"
    assert report["benchmark_ready"] is False and report["deployment_approved"] is False
    assert report["issues"] == []
    assert report["temperature_celsius"] == 45.5
    assert report["host"]["os"] == {"id": "debian", "version_id": "12"}
    assert "private-canary" not in json.dumps(report)
    assert str(pi) not in json.dumps(report)


def test_32_bit_python_or_kernel_is_not_compatible(pi, monkeypatch):
    monkeypatch.setattr(d.struct, "calcsize", lambda _: 4)
    assert "64_bit_arm_kernel_and_python_required" in d.collect_diagnostics(pi, root=pi)["issues"]
    monkeypatch.setattr(d.struct, "calcsize", lambda _: 8)
    monkeypatch.setattr(d.platform, "machine", lambda: "armv7l")
    assert "64_bit_arm_kernel_and_python_required" in d.collect_diagnostics(pi, root=pi)["issues"]


def test_non_linux_host_skips_firmware_commands(pi, monkeypatch):
    monkeypatch.setattr(d.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(d, "firmware_query", lambda _: pytest.fail("must not invoke firmware on Mac"))
    report = d.collect_diagnostics(pi, root=pi)
    assert report["status"] == "review_required"
    assert report["memory"] is None and report["temperature_celsius"] is None
    assert "linux_required" in report["issues"]


def test_missing_counters_remain_unknown(pi, monkeypatch):
    (pi / "proc/meminfo").write_text("invalid private-canary")
    monkeypatch.setattr(d, "firmware_query", lambda _: None)
    report = d.collect_diagnostics(pi, root=pi)
    assert report["memory"] is None and report["throttling"] is None
    assert report["temperature_celsius"] is None
    assert len(report["issues"]) == 3
    assert "private-canary" not in json.dumps(report)


def test_unknown_hardware_and_all_firmware_warnings_require_review(pi, monkeypatch):
    (pi / "sys/firmware/devicetree/base/model").write_text("Raspberry Pi 5 Model B Rev 1.0\x00")
    monkeypatch.setattr(d, "firmware_query", lambda command: "temp=45'C" if command == "measure_temp" else "throttled=0x10011")
    report = d.collect_diagnostics(pi, root=pi)
    assert report["status"] == "review_required"
    assert set(report["issues"]) == {"planned_pi4_hardware_not_confirmed", "active_power_or_thermal_flags",
        "historical_power_or_thermal_flags_require_review", "unknown_throttling_bits_require_review"}


def test_swap_use_is_reported_without_mutating_swap(pi):
    (pi / "proc/meminfo").write_text(MEMINFO.replace("SwapFree: 1024", "SwapFree: 512"))
    report = d.collect_diagnostics(pi, root=pi)
    assert report["memory"]["swap_used_bytes"] == 512 * 1024
    assert "swap_in_use_requires_review" in report["issues"]


def test_os_identity_parsing_never_evaluates_shell_code():
    with pytest.raises(ValueError):
        d.parse_os_release('ID="$(touch private-canary)"\nVERSION_ID=12')
    with pytest.raises(ValueError):
        d.parse_os_release('ID=debian\nID=raspbian\nVERSION_ID=12')


def test_firmware_is_allowlisted_bounded_and_has_no_shell(monkeypatch):
    monkeypatch.setattr(d.shutil, "which", lambda _: "/usr/bin/vcgencmd")
    calls = []
    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="throttled=0x0\n")
    monkeypatch.setattr(d.subprocess, "run", run)
    assert d.firmware_query("get_throttled") == "throttled=0x0\n"
    assert calls[0][0] == ["/usr/bin/vcgencmd", "get_throttled"]
    assert calls[0][1]["timeout"] == 5 and calls[0][1].get("shell", False) is False
    with pytest.raises(ValueError):
        d.firmware_query("get_throttled 0xffff")
    assert len(calls) == 1


def test_firmware_timeout_or_permission_failure_never_echoes_output(monkeypatch, capsys):
    monkeypatch.setattr(d.shutil, "which", lambda _: "/usr/bin/vcgencmd")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("private-canary", 5, output="private-canary")
    monkeypatch.setattr(d.subprocess, "run", timeout)
    assert d.firmware_query("get_throttled") is None
    monkeypatch.setattr(d.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=1, stdout="private-canary"))
    assert d.firmware_query("get_throttled") is None
    assert "private-canary" not in str(capsys.readouterr())


def test_cli_writes_review_report_and_refuses_overwrite_or_escape(pi, monkeypatch, capsys):
    monkeypatch.setattr(cli, "PROJECT_ROOT", pi)
    monkeypatch.setattr(d.platform, "machine", lambda: "armv7l")
    monkeypatch.setattr(cli, "collect_diagnostics", lambda storage: d.collect_diagnostics(storage, root=pi))
    output = pi / "evaluation/results/pi-preflight/environment.json"
    args = ["--storage-path", str(pi), "--output", str(output)]
    assert cli.main(args) == 2
    before = output.read_bytes()
    assert json.loads(before)["status"] == "review_required"
    assert cli.main(args) == 1 and output.read_bytes() == before
    assert cli.main(["--output", str(pi / "outside.json")]) == 1
    link = pi / "evaluation/results/link.json"
    link.symlink_to(output)
    assert cli.main(["--output", str(link)]) == 1
    assert "private-canary" not in str(capsys.readouterr())


def test_cli_json_stdout_and_safe_error(pi, monkeypatch, capsys):
    monkeypatch.setattr(cli, "collect_diagnostics", lambda storage: d.collect_diagnostics(storage, root=pi))
    assert cli.main(["--storage-path", str(pi)]) == 0
    assert json.loads(capsys.readouterr().out)["benchmark_ready"] is False
    assert cli.main(["--storage-path", str(pi / "private-canary")]) == 1
    assert "private-canary" not in str(capsys.readouterr())


def test_imports_and_help_work_without_site_packages_or_ml_frameworks():
    script = Path(__file__).resolve().parents[2] / "scripts/deployment/diagnose_pi.py"
    result = subprocess.run([sys.executable, "-I", "-S", str(script), "--help"],
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0
    assert "--storage-path" in result.stdout
    assert not result.stderr
