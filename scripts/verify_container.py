"""Smoke-test an already built local image on an ephemeral loopback port."""

import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from uuid import uuid4

# Standard library only: CI runs this with the runner's Python, without project dependencies.
ROOT = Path(__file__).resolve().parents[1]


def docker(*arguments: str) -> str:
    return subprocess.check_output(["docker", *arguments], text=True).strip()


def verify(image: str) -> None:
    name = f"attrition-smoke-{uuid4().hex[:12]}"
    started = False
    try:
        docker("run", "--detach", "--name", name, "--publish", "127.0.0.1::8000", image)
        started = True
        address = docker("port", name, "8000/tcp")
        base = f"http://{address}"
        for _ in range(60):
            try:
                with urllib.request.urlopen(base + "/health", timeout=2) as response:
                    health = json.load(response)
                assert health["status"] == "ok" and health["model_available"] is True and health["run_id"]
                break
            except (urllib.error.URLError, OSError):
                time.sleep(1)
        else:
            raise TimeoutError("Container did not become ready.")
        payload = (ROOT / "examples/employee.json").read_bytes()
        request = urllib.request.Request(base + "/predict", data=payload,
                                         headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=10) as response:
            result = json.load(response)
        assert 0 <= result["attrition_probability"] <= 1
        assert 0 <= result["threshold"] <= 1
        expected = "Yes" if result["attrition_probability"] >= result["threshold"] else "No"
        assert result["prediction"] == expected
        assert result["run_id"] == health["run_id"]
        checks = (
            "import os; from pathlib import Path; from attrition.models.artifacts import load_bundle; "
            "assert os.getuid() != 0; assert not Path('/app/data/raw').exists(); "
            "assert not Path('/app/src/attrition/models/train.py').exists(); "
            "assert load_bundle().run_id"
        )
        docker("exec", name, "python", "-c", checks)
        print(f"Container health, prediction, bundle, non-root user, and serving-only contents verified: {image}")
    finally:
        if started:
            try:
                print(docker("logs", name))
            finally:
                docker("rm", "--force", name)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="ibmhratt-dev:local")
    verify(parser.parse_args().image)


if __name__ == "__main__":
    main()
