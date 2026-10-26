"""Generate and inspect FedMed API keys.

    python -m app.security.keys generate --label ops --role admin
    python -m app.security.keys generate --label reader --role read
    python -m app.security.keys cert --out secrets

Only the *hash* is printed for configuration. The raw key is shown once, because it
cannot be recovered afterwards — that is the point of storing only the hash.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .auth import Scope, generate_key, hash_key
from .tls import generate_self_signed_certificate, inspect_certificate


def _generate(args: argparse.Namespace) -> int:
    raw = generate_key()
    digest = hash_key(raw)

    print(f"Generated API key for label '{args.label}' with role '{args.role}'.")
    print()
    print("  Raw key (copy it now; it is not stored and cannot be shown again):")
    print(f"    {raw}")
    print()
    print("  Add this entry to FEDMED_API_KEYS (store the HASH, not the key):")
    print(f"    {args.label}:{args.role}:{digest}")
    print()
    print("  Then send it on requests as:")
    print(f"    X-API-Key: <raw key>")
    if args.env_file:
        env_path = Path(args.env_file)
        existing = env_path.read_text(encoding="utf-8") if env_path.exists() else ""
        prefix = "" if not existing else ("\n" if not existing.endswith("\n") else "")
        entry = f"{args.label}:{args.role}:{digest}"
        if "FEDMED_API_KEYS=" in existing:
            print()
            print(f"  {env_path} already defines FEDMED_API_KEYS - append manually:")
            print(f"    {entry}")
        else:
            env_path.parent.mkdir(parents=True, exist_ok=True)
            env_path.write_text(
                existing + prefix + f"FEDMED_API_KEYS={entry}\n", encoding="utf-8"
            )
            print()
            print(f"  Written to {env_path} (git-ignored).")
    return 0


def _cert(args: argparse.Namespace) -> int:
    out = Path(args.out)
    cert_path, key_path = generate_self_signed_certificate(
        out / "server.crt",
        out / "server.key",
        hostnames=tuple(args.hostname),
        valid_days=args.days,
    )
    print(f"Wrote self-signed certificate: {cert_path}")
    print(f"Wrote private key            : {key_path}")
    print()
    info = inspect_certificate(cert_path)
    print(f"  {info.describe()}")
    print()
    print("Serve the API over HTTPS with:")
    print(
        f"  python -m uvicorn app.main:app --host 127.0.0.1 --port 8443 "
        f"--ssl-certfile {cert_path} --ssl-keyfile {key_path}"
    )
    print()
    print("This certificate is self-signed, so no client trusts it by default.")
    return 0


def _inspect(args: argparse.Namespace) -> int:
    info = inspect_certificate(Path(args.cert))
    print(info.describe())
    return 0 if info.valid else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.security.keys",
        description="Generate FedMed API keys and development TLS certificates.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    generate = sub.add_parser("generate", help="Generate an API key.")
    generate.add_argument("--label", required=True, help="Name used to attribute the key.")
    generate.add_argument(
        "--role", required=True, choices=[scope.value for scope in Scope], help="Scope of the key."
    )
    generate.add_argument(
        "--env-file",
        default=None,
        help="Optional path to append the hash to, e.g. backend/.env.",
    )
    generate.set_defaults(func=_generate)

    cert = sub.add_parser("cert", help="Generate a development TLS certificate.")
    cert.add_argument("--out", default="secrets", help="Directory for the certificate and key.")
    cert.add_argument(
        "--hostname",
        action="append",
        default=["localhost", "127.0.0.1"],
        help="Hostname or IP to include in the SAN list. Repeatable.",
    )
    cert.add_argument("--days", type=int, default=365, help="Validity in days.")
    cert.set_defaults(func=_cert)

    inspect = sub.add_parser("inspect", help="Inspect an existing certificate.")
    inspect.add_argument("--cert", required=True)
    inspect.set_defaults(func=_inspect)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
