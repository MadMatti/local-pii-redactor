# Raspberry Pi deployment

Read-only hardware diagnostics are implemented. Model transfer, native builds,
benchmark execution, and service installation are **not yet approved or run**.
Only Q8_0 is currently eligible for a proposed benchmark; Q4/Q5 failed the
validation gates. See [the preparation plan](benchmark-plan.md).

## Collect the target Pi's inventory

Run this on the intended Pi from a checkout of this repository, using Python
3.11. No package installation, MLX, PyTorch, network, model, or root access is
required by the diagnostic command:

```bash
python3.11 scripts/deployment/diagnose_pi.py \
  --storage-path . \
  --output evaluation/results/pi-preflight-20261005/environment.json
```

Use an existing `--storage-path` on the intended model filesystem; it is inspected
but not written. Pick a fresh output filename for each snapshot. Reports are
new-file-only under the Git-ignored `evaluation/results/` directory. Omit
`--output` to print the same JSON instead of writing a report.

The snapshot includes OS identity, ARM architecture, Python pointer width, CPU
count, Pi model, RAM available/total, swap use, filesystem capacity/free space,
temperature, and current/historical firmware throttling flags. The only external
commands are the read-only `vcgencmd measure_temp` and `vcgencmd get_throttled`,
with five-second timeouts. It does not invoke SSH, sudo, an installer, a model
server, firmware setters, or network discovery. Missing firmware access remains
unknown; the tool does not try to install it or escalate privileges.

The report omits hostname, username, serial number, network addresses, the storage
path, environment variables, credentials, and document content. It records
hardware/environment metadata, so keep raw reports local and review before sharing.

Exit codes:

- `0`: all required inventory fields collected for the planned 64-bit Pi 4,
  without the diagnostic warnings below. **Not benchmark readiness or approval.**
- `2`: report completed but requires review: unsupported OS/hardware/architecture,
  32-bit Python, unknown telemetry, active/historical/unknown firmware flags,
  swap in use, or zero free storage. This is expected on the development Mac.
- `1`: invalid paths, existing report, permission failure, or another collection
  failure. Raw exception/command output is not printed.

`benchmark_ready` and `deployment_approved` remain false even on a healthy Pi.
One snapshot cannot establish memory fit, adequate cooling, sustained performance,
or power-supply quality. Temperature and free-resource stop limits still need a
device-specific approved policy. A 64-bit ARM kernel with 32-bit Python does not
pass the architecture check. A different Pi model requires its own review.

## Human information still needed

Provide the Pi model/RAM, OS version, cooling, power-supply rating, storage medium,
and intended SSH host/user if remote work is authorized. Do not send passwords or
private keys. Explicit approval of a Q8-only hardware benchmark is separate from
production-model selection and service deployment.

## Counter interpretation

RAM availability uses Linux `MemAvailable`, not `MemFree`; the former estimates
memory available without swapping. Counter units are converted from kernel kB
to bytes. See [Linux `/proc` documentation](https://www.kernel.org/doc/html/latest/filesystems/proc.html#meminfo).

Firmware flags distinguish current conditions from historical occurrences; a
past undervoltage/throttling flag does not mean an active fault. Unknown bits or
missing readings require review, not an assumed healthy state. Definitions are
from [the official Raspberry Pi `vcgencmd` documentation](https://www.raspberrypi.com/documentation/computers/os.html#vcgencmd).
