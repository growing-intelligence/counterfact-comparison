"""Verify this certificate offline, without contacting anyone: the Ed25519 signature, the sha256 of every file it lists,
and the Merkle root over those hashes. Needs Python 3 and one library: pip install cryptography
    python verify_certificate.py CounterFact-certificate-SIGNED.json [proof_dir]
proof_dir defaults to the directory holding the certificate. Exit status 0 = everything checks out."""
import base64, hashlib, json, os, sys
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
cert_path = sys.argv[1]; root = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(cert_path))
cert = json.load(open(cert_path, encoding="utf-8")); sig = cert["signature"]; ok = True
# 1. signature. The canonical bytes are the certificate without 'signature' and 'self_sha256', serialised with sorted keys,
#    compact separators and ensure_ascii=True (non-ASCII escaped). Serialising any other way gives different bytes.
core = {k: v for k, v in cert.items() if k not in ("signature", "self_sha256")}
msg = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
pub = base64.b64decode(sig["public_key_b64"])
try:
    Ed25519PublicKey.from_public_bytes(pub).verify(base64.b64decode(sig["signature_b64"]), msg); print("signature: VALID (Ed25519, key " + sig["public_key_b64"] + ")")
except Exception as e: ok = False; print("signature: INVALID —", e)
h = hashlib.sha256(msg).hexdigest()
print("self_sha256:", "matches" if h == cert.get("self_sha256") else f"MISMATCH (computed {h})"); ok &= h == cert.get("self_sha256")
# 2. every listed file
files = cert["file_hashes_sha256"]; missing = []; bad = []
for rel, want in sorted(files.items()):
    p = os.path.join(root, rel)
    if not os.path.exists(p): missing.append(rel); continue
    d = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 24), b""): d.update(c)
    if d.hexdigest() != want: bad.append(rel)
print(f"files: {len(files) - len(missing) - len(bad)} of {len(files)} verified" + (f"; MISSING {missing}" if missing else "") + (f"; MISMATCH {bad}" if bad else ""))
ok &= not bad
# 3. merkle root over the file hashes
merkle = hashlib.sha256("".join(f"{k}:{files[k]}\n" for k in sorted(files)).encode()).hexdigest()
print("merkle_root:", "matches" if merkle == cert["merkle_root_sha256"] else f"MISMATCH (computed {merkle})"); ok &= merkle == cert["merkle_root_sha256"]
print("\nRESULT:", "OK" if ok and not missing else ("OK (signature and merkle valid; some files not present here)" if ok else "FAILED"))
sys.exit(0 if ok else 1)
