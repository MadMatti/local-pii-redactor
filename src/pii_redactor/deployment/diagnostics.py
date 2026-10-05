"""Standard-library-only Pi inventory; successful collection is not approval."""

from __future__ import annotations

import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
from datetime import UTC, datetime
from pathlib import Path


MEMORY_FIELDS = ("MemTotal", "MemAvailable", "SwapTotal", "SwapFree")
THROTTLE_FLAGS = ("undervoltage", "arm_frequency_capped", "throttled", "soft_temperature_limit")


def parse_meminfo(text: str) -> dict[str, int]:
    values = {}
    for line in text.splitlines():
        name = line.split(":", 1)[0]
        if name not in MEMORY_FIELDS:
            continue
        match = re.fullmatch(r"(\w+):\s*(\d+)\s+kB\s*", line)
        if not match or name in values:
            raise ValueError("invalid memory counters")
        values[name] = int(match[2]) * 1024
    if (set(values) != set(MEMORY_FIELDS) or values["MemTotal"] <= 0
            or values["MemAvailable"] > values["MemTotal"]
            or values["SwapFree"] > values["SwapTotal"]):
        raise ValueError("missing or inconsistent memory counters")
    return {"total_bytes": values["MemTotal"], "available_bytes": values["MemAvailable"],
            "swap_total_bytes": values["SwapTotal"], "swap_free_bytes": values["SwapFree"],
            "swap_used_bytes": values["SwapTotal"] - values["SwapFree"]}


def parse_throttled(text: str) -> dict:
    match = re.fullmatch(r"throttled=(0x[0-9a-fA-F]{1,8})\s*", text)
    if not match:
        raise ValueError("invalid throttling response")
    mask = int(match[1], 16)
    known = 0xF000F
    return {"mask": hex(mask),
            "active": [name for bit, name in enumerate(THROTTLE_FLAGS) if mask & (1 << bit)],
            "historical": [name for bit, name in enumerate(THROTTLE_FLAGS) if mask & (1 << (bit + 16))],
            "unknown_bits": hex(mask & ~known)}


def parse_temperature(text: str) -> float:
    match = re.fullmatch(r"temp=(-?\d{1,3}(?:\.\d+)?)'C\s*", text)
    if not match or not -40 <= float(match[1]) <= 150:
        raise ValueError("invalid temperature response")
    return float(match[1])


def _read_text(path: Path) -> str | None:
    try:
        with path.open(encoding="utf-8") as handle:
            text = handle.read(65537)
        return text if len(text) <= 65536 else None
    except (OSError, UnicodeError):
        return None


def parse_os_release(text: str) -> dict[str, str]:
    """Whitelist distro identifiers; never serialize the whole configuration."""
    result = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator and key in {"ID", "VERSION_ID"}:
            parts = shlex.split(value, comments=False)
            if len(parts) != 1 or not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", parts[0]) or key in result:
                raise ValueError("invalid OS identification")
            result[key] = parts[0]
    if set(result) != {"ID", "VERSION_ID"}:
        raise ValueError("missing OS identification")
    return {"id": result["ID"], "version_id": result["VERSION_ID"]}


def firmware_query(command: str) -> str | None:
    """Only these read-only firmware commands are allowed; no sudo or shell."""
    if command not in {"get_throttled", "measure_temp"}:
        raise ValueError("firmware command is not allowed")
    binary = shutil.which("vcgencmd")
    if binary is None:
        return None
    try:
        result = subprocess.run([binary, command], stdin=subprocess.DEVNULL,
                                capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.SubprocessError, UnicodeError):
        return None
    if result.returncode != 0 or len(result.stdout) > 128:
        return None
    return result.stdout


def collect_diagnostics(storage_path: Path, *, root: Path = Path("/")) -> dict:
    """Read public hardware counters only; root injection supports offline tests."""
    if not storage_path.is_dir():
        raise ValueError("storage path must be an existing directory")
    disk = shutil.disk_usage(storage_path)
    system = platform.system()
    architecture = platform.machine().lower()
    pointer_bits = struct.calcsize("P") * 8
    cores = os.cpu_count()
    issues = []
    if system != "Linux":
        issues.append("linux_required")
    if architecture not in {"aarch64", "arm64"} or pointer_bits != 64:
        issues.append("64_bit_arm_kernel_and_python_required")
    if cores is None or cores < 1:
        issues.append("cpu_count_unavailable")
    model = os_info = memory = temperature = throttling = None
    if system == "Linux":
        for relative in ("sys/firmware/devicetree/base/model", "proc/device-tree/model"):
            candidate = _read_text(root / relative)
            if candidate is not None:
                candidate = candidate.rstrip("\x00\n")
                if re.fullmatch(r"Raspberry Pi [A-Za-z0-9 .+-]{1,80}", candidate):
                    model = candidate
                    break
        os_text = _read_text(root / "etc/os-release")
        if os_text is None:
            os_text = _read_text(root / "usr/lib/os-release")
        try:
            os_info = parse_os_release(os_text or "")
        except ValueError:
            issues.append("os_identification_unavailable_or_invalid")
        try:
            memory = parse_meminfo(_read_text(root / "proc/meminfo") or "")
        except ValueError:
            issues.append("memory_counters_unavailable_or_invalid")
    if model is None or not model.startswith("Raspberry Pi 4 Model B "):
        issues.append("planned_pi4_hardware_not_confirmed")
    # Do not execute a coincidentally named command on a non-Pi development host.
    if model is not None and system == "Linux":
        try:
            temperature = parse_temperature(firmware_query("measure_temp") or "")
        except ValueError:
            issues.append("temperature_unavailable_or_invalid")
        try:
            throttling = parse_throttled(firmware_query("get_throttled") or "")
        except ValueError:
            issues.append("throttling_unavailable_or_invalid")
        if throttling is not None:
            if throttling["active"]:
                issues.append("active_power_or_thermal_flags")
            if throttling["historical"]:
                issues.append("historical_power_or_thermal_flags_require_review")
            if throttling["unknown_bits"] != "0x0":
                issues.append("unknown_throttling_bits_require_review")
    if memory is not None and memory["swap_used_bytes"]:
        issues.append("swap_in_use_requires_review")
    if disk.free == 0:
        issues.append("no_free_storage")
    return {"schema_version": 1,
            "collected_at_utc": datetime.now(UTC).isoformat(),
            "status": "review_required" if issues else "diagnostics_complete",
            "benchmark_ready": False, "deployment_approved": False,
            "host": {"system": system, "architecture": architecture,
                     "python_pointer_bits": pointer_bits, "python_version": platform.python_version(),
                     "logical_cpu_count": cores, "device_model": model, "os": os_info},
            "memory": memory, "storage": {"total_bytes": disk.total, "free_bytes": disk.free},
            "temperature_celsius": temperature, "throttling": throttling,
            "issues": issues,
            "human_inputs_required": ["power_supply_rating", "cooling", "storage_medium",
                                      "benchmark_authorization", "device_specific_resource_stop_limits"],
            "limitations": [
                "One idle snapshot does not prove runtime memory fit, stable power, cooling, or sustained performance.",
                "Historical throttling flags do not imply an active fault; unknown telemetry never means zero.",
                "No hostname, username, serial number, network address, credentials, or document content is collected.",
                "This command does not install, transfer, benchmark, change firmware flags, or start services."]}
