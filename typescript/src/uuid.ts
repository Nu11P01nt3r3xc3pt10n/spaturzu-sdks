import { randomBytes } from "node:crypto";

// UUIDv7 generator (RFC 9562).
//
// Layout: 48 bits of unix_ts_ms (big-endian) || 4 bits version (0111)
//   || 12 bits random || 2 bits variant (10) || 62 bits random.
//
// Why v7 for the SDK's request ids: the spaturzu gateway extracts the
// timestamp prefix and writes it to `requests.created_at`. Retries reuse
// the same id, so they produce the same created_at, so
// `ON CONFLICT (project_id, id, created_at) DO NOTHING` deduplicates
// correctly on the TimescaleDB-bound hypertable. (See feedback memory
// `request_dedup_uuidv7_upsert`.)
//
// Side benefit: time-ordered ids give B-tree insert locality and avoid
// the random-page-split pattern that hurts v4 UUID PKs under write load.
export function uuidv7(): string {
  const ms = Date.now();
  const buf = Buffer.alloc(16);
  // Big-endian 48-bit timestamp into bytes 0-5. writeUIntBE supports up
  // to 6-byte values; Date.now() is well within 2^48 until the year ~8921.
  buf.writeUIntBE(ms, 0, 6);
  // 10 random bytes fill the rest. The version + variant nibbles below
  // overwrite four of these bits.
  randomBytes(10).copy(buf, 6);
  // Version 7 in the high nibble of byte 6. writeUInt8/readUInt8 dodges
  // the `noUncheckedIndexedAccess` complaint about `buf[i]` being
  // `number | undefined`.
  buf.writeUInt8((buf.readUInt8(6) & 0x0f) | 0x70, 6);
  // Variant 10x in the top two bits of byte 8.
  buf.writeUInt8((buf.readUInt8(8) & 0x3f) | 0x80, 8);
  const hex = buf.toString("hex");
  return (
    hex.slice(0, 8) +
    "-" +
    hex.slice(8, 12) +
    "-" +
    hex.slice(12, 16) +
    "-" +
    hex.slice(16, 20) +
    "-" +
    hex.slice(20, 32)
  );
}
