"""RFC 8785 (JCS) canonical JSON serialization and content digests.

Every persisted experiment identity (manifest hashes, contract hashes,
trial IDs, scheduler ordering keys, summary input hashes) is a SHA-256
over the JCS form of a JSON object, so two writers can only agree on an
identity when they agree on the exact configuration bytes.

Digests for different purposes are domain-separated: the hashed object
embeds a purpose string, so a trial ID can never collide with a
scheduler ordering key computed from the same fields.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any


class CanonicalizationError(ValueError):
    """The value cannot be represented in canonical JSON."""


def _format_float(value: float) -> str:
    """ECMAScript Number::toString for a finite float (RFC 8785 §3.2.2.3).

    Python's repr() already yields the shortest round-tripping digits;
    this reformats them with ECMAScript's decimal/exponent layout rules.
    """
    if value == 0.0:
        return "0"  # covers -0.0, which JCS serializes as "0"

    mantissa, exponent = repr(float(value)), 0
    if "e" in mantissa:
        mantissa, _, exp_text = mantissa.partition("e")
        exponent = int(exp_text)
    sign = ""
    if mantissa.startswith("-"):
        sign, mantissa = "-", mantissa[1:]
    if "." in mantissa:
        integer_part, _, fraction_part = mantissa.partition(".")
    else:
        integer_part, fraction_part = mantissa, ""
    digits = (integer_part + fraction_part).lstrip("0")
    # decimal point position n: value = 0.<digits> * 10**n
    point = len(integer_part.lstrip("0")) if integer_part.strip("0") else (
        -(len(fraction_part) - len(fraction_part.lstrip("0")))
    )
    point += exponent
    digits = digits.rstrip("0")
    k = len(digits)

    if k <= point <= 21:
        body = digits + "0" * (point - k)
    elif 0 < point <= 21:
        body = digits[:point] + "." + digits[point:]
    elif -6 < point <= 0:
        body = "0." + "0" * (-point) + digits
    else:
        exp = point - 1
        exp_str = f"e+{exp}" if exp >= 0 else f"e-{-exp}"
        body = (digits if k == 1 else digits[0] + "." + digits[1:]) + exp_str
    return sign + body


def _serialize(value: Any, out: list) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, int):
        # RFC 8785 numbers are IEEE 754 doubles; larger integers would
        # silently lose precision and break cross-language agreement.
        if abs(value) > 2**53:
            raise CanonicalizationError(
                f"Integer {value} exceeds IEEE 754 exact range"
            )
        out.append(str(value))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError(
                "NaN and Infinity cannot be canonicalized"
            )
        out.append(_format_float(value))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _serialize(item, out)
        out.append("]")
    elif isinstance(value, dict):
        for key in value:
            if not isinstance(key, str):
                raise CanonicalizationError(
                    f"Object keys must be strings, got {type(key).__name__}"
                )
        out.append("{")
        # RFC 8785 sorts property names by UTF-16 code units.
        for i, key in enumerate(
            sorted(value, key=lambda k: k.encode("utf-16-be"))
        ):
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _serialize(value[key], out)
        out.append("}")
    else:
        raise CanonicalizationError(
            f"Type {type(value).__name__} cannot be canonicalized"
        )


def jcs_canonicalize(value: Any) -> str:
    """Return the RFC 8785 canonical JSON text of a JSON-compatible value."""
    out: list = []
    _serialize(value, out)
    return "".join(out)


def jcs_sha256(value: Any) -> str:
    """Hex SHA-256 of the UTF-8 canonical JSON encoding of `value`."""
    return hashlib.sha256(jcs_canonicalize(value).encode("utf-8")).hexdigest()


def canonical_object_digest(domain: str, payload: dict) -> str:
    """Domain-separated digest: identical payloads hashed for different
    purposes (trial identity vs. schedule ordering) never collide."""
    return jcs_sha256({"domain": domain, "payload": payload})


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
