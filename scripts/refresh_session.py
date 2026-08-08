"""Refresh the DeepSeek session headlessly from the Chrome profile copy.

Called periodically by a systemd timer. If the cached session is still fresh
it's a no-op; otherwise it re-captures the token from the profile. Exits 0 on
success, 1 if the profile no longer yields a token (needs manual re-login).
"""
import sys
from pathlib import Path

ROOT = Path("/var/www/Deepseek-API")
sys.path.insert(0, str(ROOT))

from deepseek.auth import get_session, LoginRequired

PROFILE_DIR = Path("/var/www/Deepseek-API/session/profile-copy")
SESSION_FILE = ROOT / "session" / "session.json"

def main() -> int:
    try:
        # max_age=0 forces a headless refresh from the profile copy.
        session = get_session(
            profile_dir=PROFILE_DIR,
            session_file=SESSION_FILE,
            max_age=0,
            allow_interactive=False,
        )
    except LoginRequired:
        print("REFRESH_FAILED: no token in profile — manual re-login needed")
        return 1
    except Exception as e:
        print(f"REFRESH_ERROR: {e}")
        return 1

    session.save(SESSION_FILE)
    print(f"REFRESH_OK: token={session.token[:12]}... cookies={len(session.cookies)} age={session.age:.0f}s")
    return 0

if __name__ == "__main__":
    sys.exit(main())