"""Transport security: TLS certificates for the API and the federated gRPC path.

What this module really does:

    * generates a self-signed certificate + key (RSA 2048, SANs for localhost and
      127.0.0.1) using the `cryptography` library
    * inspects an existing certificate and reports whether it parses, its subject,
      its validity window, and whether it is currently valid
    * returns the SSL arguments uvicorn needs to serve HTTPS
    * returns the `Certificates` object Flower needs for TLS on the gRPC path

What it does not do: it does not make anything trustworthy. A self-signed
certificate is not trusted by any client by default. This is a development
mechanism — see docs/SECURITY.md for what production would require.
"""

from __future__ import annotations

import datetime as dt
import ipaddress
from dataclasses import dataclass
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

DEFAULT_VALID_DAYS = 365
DEFAULT_KEY_SIZE = 2048


@dataclass(frozen=True)
class CertificateInfo:
    """What an on-disk certificate actually is."""

    path: str
    exists: bool
    valid: bool
    subject: str | None = None
    issuer: str | None = None
    not_before: str | None = None
    not_after: str | None = None
    days_remaining: int | None = None
    is_self_signed: bool | None = None
    san: list[str] = None  # type: ignore[assignment]
    error: str | None = None

    def __post_init__(self) -> None:
        if self.san is None:
            object.__setattr__(self, "san", [])

    def describe(self) -> str:
        if not self.exists:
            return f"No certificate at {self.path}."
        if not self.valid:
            return f"Certificate at {self.path} could not be read: {self.error}"
        return (
            f"{self.subject} (self-signed: {self.is_self_signed}), "
            f"valid until {self.not_after} ({self.days_remaining} days)"
        )


def generate_self_signed_certificate(
    cert_path: Path,
    key_path: Path,
    hostnames: tuple[str, ...] = ("localhost", "127.0.0.1"),
    valid_days: int = DEFAULT_VALID_DAYS,
    key_size: int = DEFAULT_KEY_SIZE,
) -> tuple[Path, Path]:
    """Write a self-signed certificate and an unencrypted private key.

    The key is written with owner-only permissions where the platform supports it.
    It is NOT encrypted with a passphrase, which is a deliberate simplifying choice
    for a local development certificate and is recorded as a limitation.

    Refuses to overwrite an existing certificate or key, so a certificate a user
    is already using cannot be silently replaced.
    """
    cert_path = Path(cert_path)
    key_path = Path(key_path)
    for path in (cert_path, key_path):
        if path.exists():
            raise FileExistsError(
                f"{path} already exists. Delete it deliberately or choose another path; "
                "this function will not overwrite a certificate in use."
            )

    key = rsa.generate_private_key(public_exponent=65537, key_size=key_size)
    now = dt.datetime.now(dt.timezone.utc)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "FedMed development"),
            x509.NameAttribute(NameOID.COMMON_NAME, hostnames[0] if hostnames else "localhost"),
        ]
    )

    san_entries: list[x509.GeneralName] = []
    for host in hostnames:
        try:
            san_entries.append(x509.IPAddress(ipaddress.ip_address(host)))
        except ValueError:
            san_entries.append(x509.DNSName(host))

    certificate = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(days=valid_days))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    cert_path.parent.mkdir(parents=True, exist_ok=True)
    cert_path.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    key_path.write_bytes(
        key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    try:
        # Best effort: restrict the key to the owner. A no-op on filesystems that
        # do not support POSIX modes.
        key_path.chmod(0o600)
    except OSError:
        pass

    return cert_path, key_path


def inspect_certificate(path: Path) -> CertificateInfo:
    """Read a certificate and report what it is, without trusting it."""
    path = Path(path)
    if not path.exists():
        return CertificateInfo(path=str(path), exists=False, valid=False)

    try:
        data = path.read_bytes()
        certificate = x509.load_pem_x509_certificate(data)
    except Exception as exc:  # noqa: BLE001 - any parse failure is "not usable"
        return CertificateInfo(
            path=str(path),
            exists=True,
            valid=False,
            error=f"{type(exc).__name__}: {exc}",
        )

    now = dt.datetime.now(dt.timezone.utc)
    not_after = certificate.not_valid_after_utc
    not_before = certificate.not_valid_before_utc

    san: list[str] = []
    try:
        extension = certificate.extensions.get_extension_for_class(x509.SubjectAlternativeName)
        san = [str(name.value) for name in extension.value]
    except x509.ExtensionNotFound:
        san = []

    return CertificateInfo(
        path=str(path),
        exists=True,
        valid=not_before <= now <= not_after,
        subject=certificate.subject.rfc4514_string(),
        issuer=certificate.issuer.rfc4514_string(),
        not_before=not_before.isoformat(),
        not_after=not_after.isoformat(),
        days_remaining=(not_after - now).days,
        is_self_signed=certificate.issuer == certificate.subject,
        san=san,
    )


def uvicorn_ssl_kwargs(cert_path: Path, key_path: Path) -> dict[str, str]:
    """The keyword arguments uvicorn needs to serve HTTPS.

    Raises FileNotFoundError rather than returning paths that do not exist, because
    silently falling back to plain HTTP after a TLS configuration error is exactly
    the failure mode that makes an operator believe traffic is encrypted.
    """
    cert_path = Path(cert_path)
    key_path = Path(key_path)
    if not cert_path.exists():
        raise FileNotFoundError(f"TLS certificate not found: {cert_path}.")
    if not key_path.exists():
        raise FileNotFoundError(f"TLS key not found: {key_path}.")
    return {"ssl_certfile": str(cert_path), "ssl_keyfile": str(key_path)}


def flower_certificates(ca_path: Path, cert_path: Path, key_path: Path):
    """Build Flower's `Certificates` object for TLS on the gRPC path.

    Imported lazily so that this module can be used (for certificate generation and
    inspection) without Flower installed.
    """
    from flwr.common import Config  # noqa: F401  (imported for parity with Flower)
    from flwr.common.grpc import GRPC_MAX_MESSAGE_LENGTH  # noqa: F401
    from flwr.common.typing import Certificates  # type: ignore[attr-defined]

    for label, path in (("CA", ca_path), ("certificate", cert_path), ("key", key_path)):
        if not Path(path).exists():
            raise FileNotFoundError(f"TLS {label} not found: {path}.")

    return Certificates(
        ca_certificate=Path(ca_path).read_bytes(),
        client_certificate=Path(cert_path).read_bytes(),
        client_private_key=Path(key_path).read_bytes(),
    )
