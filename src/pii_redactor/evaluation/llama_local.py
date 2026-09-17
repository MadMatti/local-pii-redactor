"""Private loopback-only llama.cpp evaluation transport; no ML framework imports."""

from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import time
from contextlib import contextmanager
from urllib.error import HTTPError, URLError
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener


class LocalRuntimeError(RuntimeError):
    """A local runtime check failed; messages must not include prompt contents."""


class NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise LocalRuntimeError("local model endpoint attempted an HTTP redirect")


class LocalClient:
    def __init__(self, port, key):
        if type(port) is not int or not 1 <= port <= 65535 or not key:
            raise LocalRuntimeError("invalid local endpoint configuration")
        self.origin = f"http://127.0.0.1:{port}"
        self.key = key
        self.opener = build_opener(ProxyHandler({}), NoRedirects())

    def request(self, endpoint, body=None, *, timeout=120):
        if endpoint not in {"/health", "/props", "/tokenize", "/apply-template", "/completion"}:
            raise LocalRuntimeError("endpoint is not permitted for frozen evaluation")
        if endpoint == "/props" and body is not None:
            raise LocalRuntimeError("runtime properties are read-only")
        request = Request(self.origin + endpoint,
                          data=json.dumps(body).encode() if body is not None else None,
                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        try:
            with self.opener.open(request, timeout=timeout) as response:
                if response.geturl() != self.origin + endpoint:
                    raise LocalRuntimeError("local model endpoint changed")
                result = json.loads(response.read(8 * 1024 * 1024 + 1))
        except (HTTPError, URLError, OSError, ValueError) as exc:
            raise LocalRuntimeError(f"local model request failed ({type(exc).__name__})") from None
        if not isinstance(result, dict):
            raise LocalRuntimeError("local model response is not an object")
        return result


def generation_request(tokens, max_tokens, seed):
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise LocalRuntimeError("prompt must contain token IDs")
    return {"prompt": tokens, "n_predict": max_tokens, "seed": seed,
            "temperature": 0.0, "samplers": ["temperature"],
            "repeat_penalty": 1.0, "repeat_last_n": 0,
            "presence_penalty": 0.0, "frequency_penalty": 0.0,
            "cache_prompt": False, "stream": False, "return_tokens": True,
            "stop": [], "id_slot": 0}


@contextmanager
def local_server(binary, model, *, context, threads, startup_timeout=180):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    client = LocalClient(port, secrets.token_urlsafe(32))
    command = [str(binary), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
               "--ctx-size", str(context), "--parallel", "1", "--threads", str(threads),
               "--batch-size", "512", "--ubatch-size", "512", "--gpu-layers", "all",
               "--fit", "off", "--flash-attn", "on", "--cache-type-k", "f16", "--cache-type-v", "f16",
               "--no-context-shift", "--no-cache-prompt", "--offline", "--no-webui",
               "--no-ui-mcp-proxy", "--no-slots", "--log-disable", "--jinja", "--reasoning", "off"]
    # Do not inherit server configuration, credentials, or arbitrary network/model overrides.
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("LLAMA_", "HF_", "HUGGING_FACE_", "GGML_"))}
    env["LLAMA_API_KEY"] = client.key
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + startup_timeout
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise LocalRuntimeError("owned local model server exited during startup")
            try:
                if client.request("/health", timeout=2).get("status") == "ok":
                    break
            except LocalRuntimeError:
                pass
            time.sleep(0.25)
        else:
            raise LocalRuntimeError("local model server startup timed out")
        if process.poll() is not None:
            raise LocalRuntimeError("owned local model server is not running")
        yield client, command
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
