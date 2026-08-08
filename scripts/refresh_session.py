"""Refresh the DeepSeek session headlessly from the Chrome profile copy.

Called periodically by a systemd timer. If the cached session is still fresh
it's a no-op; otherwise it re-captures the token from the profile. Exits 0 on
success, 1 only if there is NO usable session at all (needs manual re-login).

Token-only sessions (created with `python -m deepseek.auth --token <TOKEN>`)
are kept as-is: we never overwrite a working session with a failed refresh.
"""
import sys
import time
from pathlib import Path

ROOT = Path("/var/www/Deepseek-API")
sys.path.insert(0, str(ROOT))

from deepseek.auth import Session, LoginRequired, SESSION_MAX_AGE

PROFILE_DIR = Path("/var/www/Deepseek-API/session/profile-copy")
SESSION_FILE = ROOT / "session" / "session.json"


def main() -> int:
    cached = Session.load(SESSION_FILE)
    if cached and cached.age < SESSION_MAX_AGE:
        print(f"REFRESH_SKIP: session fresh ({cached.age:.0f}s old, token={cached.token[:12]}...)")
        return 0

    try:
        from deepseek.auth import _headless_refresh
        session = _headless_refresh(PROFILE_DIR)
    except Exception as e:
        print(f"REFRESH_ERROR: {e}")
        # If we have a working session (even stale), keep it rather than bricking.
        if cached:
            print("REFRESH_KEEP: retaining existing session")
            return 0
        return 1

    if session is None:
        if cached:
            print("REFRESH_KEEP: profile has no token — retaining existing session")
            return 0
        print("REFRESH_FAILED: no token in profile — run `python -m deepseek.auth` "
              "or `python -m deepseek.auth --token <TOKEN>`")
        return 1

    session.save(SESSION_FILE)
    print(f"REFRESH_OK: token={session.token[:12]}... cookies={len(session.cookies)} age={session.age:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())