"""UUIDv7 generator (RFC 9562).

Mirrors ``sdks/typescript/src/uuid.ts``. Used for request ids; the spaturzu
gateway extracts the timestamp prefix into ``requests.created_at`` so
retries (same id) hit the same row via ``ON CONFLICT (project_id, id,
created_at)``.

Layout: 48 bits unix_ts_ms (big-endian) || 4 bits version (0111) ||
12 bits random || 2 bits variant (10) || 62 bits random.

Hand-rolled rather than relying on ``uuid.uuid7()`` because that landed
in Python 3.13. Customers may be on 3.10–3.12; the helper has no
dependencies beyond the standard library.
"""

from __future__ import annotations

import os
import time
import uuid


def uuid7() -> str:
    ms = int(time.time() * 1000)
    buf = bytearray(16)
    # First 6 bytes: big-endian unix_ts_ms. The value fits in 48 bits
    # until ~year 8921; ``int.to_bytes`` raises ``OverflowError`` if
    # somehow exceeded, surfacing the corruption rather than silently
    # truncating.
    buf[0:6] = ms.to_bytes(6, "big")
    # 10 random bytes for the rest; version + variant nibbles below
    # overwrite four of these bits.
    buf[6:16] = os.urandom(10)
    # Version 7 in the high nibble of byte 6.
    buf[6] = (buf[6] & 0x0F) | 0x70
    # Variant 10x in the top two bits of byte 8.
    buf[8] = (buf[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(buf)))
