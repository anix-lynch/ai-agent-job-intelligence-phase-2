"""Load secrets from env, global config, or project .env."""

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_secret(key: str, default: str = None) -> str:
    if key in os.environ:
        return os.environ[key]
    global_secrets = Path.home() / ".config" / "secrets" / "global.env"
    if global_secrets.exists():
        try:
            with open(global_secrets, "r") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(f"{key}="):
                        return line.split("=", 1)[1].strip("'\"")
        except (PermissionError, IOError):
            pass
    env_file = _PROJECT_ROOT / ".env"
    if env_file.exists():
        with open(env_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(f"{key}="):
                    return line.split("=", 1)[1].strip("'\"")
    if default is not None:
        return default
    raise ValueError(
        f"Secret '{key}' not found in environment, global.env, or .env file"
    )


def load_secrets_to_env():
    for secrets_file in [
        Path.home() / ".config" / "secrets" / "global.env",
        _PROJECT_ROOT / ".env",
    ]:
        if secrets_file.exists():
            try:
                with open(secrets_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            os.environ[key.strip()] = value.strip("'\"")
            except (PermissionError, IOError):
                continue


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python get_secret.py <secret_key>")
        sys.exit(1)
    try:
        print(get_secret(sys.argv[1]))
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
