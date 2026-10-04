"""Hub boundary and no-clobber publication regression tests."""
import asyncio
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx
from starlette.responses import PlainTextResponse

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "web"))
from services.access import BackendAccessMiddleware
from services.output_files import new_output
from jupyter_config import setup_mona_track


class AccessOutputTests(unittest.TestCase):
    def test_hub_job_endpoints_reject_other_owner(self):
        import web.app as app
        import state
        from fastapi import HTTPException
        async def run():
            with patch.object(state, "JUPYTER_MODE", True), patch.object(state, "resolve_identity", return_value="alice"), \
                    patch.dict(app.training_jobs, {"foreign": {"username": "bob"}}), \
                    patch.dict(app.background_jobs, {"foreign-bg": {"username": "bob"}}):
                for endpoint, value in [(app.get_training_status, "foreign"), (app.cancel_training, "foreign"),
                                        (app.get_job_status, "foreign-bg"), (app.get_active_jobs, "bob"),
                                        (app.list_user_jobs, "bob")]:
                    with self.assertRaises(HTTPException) as error:
                        await endpoint(value)
                    self.assertEqual(error.exception.status_code, 403)
                self.assertNotIn("cancel_requested", app.training_jobs["foreign"])
        asyncio.run(run())

    def test_failed_merge_preserves_existing_output(self):
        import web.app as app
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing.mp4"
            output.write_bytes(b"user-edited-video")
            with patch.dict(app.background_jobs, {"merge-test": {}}), \
                    patch.dict(app.sessions, {"merge-user": {"files": {}, "detect_files": {}}}), \
                    patch.object(app, "save_background_jobs"):
                app._run_merge_job("merge-test", "merge-user", ["missing"], output, 30, True)
                self.assertEqual(app.background_jobs["merge-test"]["status"], "failed")
                self.assertEqual(output.read_bytes(), b"user-edited-video")
                self.assertEqual(list(Path(directory).iterdir()), [output])

    def test_proxy_secret_is_unique_and_shared_only_with_backend(self):
        first, second = setup_mona_track(), setup_mona_track()
        secret = first["environment"]["MONA_TRACK_PROXY_TOKEN"]
        self.assertEqual(first["request_headers_override"]["X-Mona-Proxy-Token"], secret)
        self.assertNotEqual(secret, second["environment"]["MONA_TRACK_PROXY_TOKEN"])
        self.assertNotIn(secret, " ".join(first["command"]))

    def test_http_boundary_and_local_origin(self):
        async def run():
            for hub, token in [(True, "secret"), (True, ""), (False, "")]:
                app = BackendAccessMiddleware(PlainTextResponse("ok"), hub_mode=hub, proxy_token=token)
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://local") as client:
                    response = await client.get("/")
                    self.assertEqual(response.status_code, 403 if hub else 200)
                    if hub:
                        response = await client.get("/", headers=[(b"X-Mona-Proxy-Token", b"\xff")])
                        self.assertEqual(response.status_code, 403)
                    response = await client.get("/", headers={"X-Mona-Proxy-Token": "secret"})
                    self.assertEqual(response.status_code, 403 if hub and not token else 200)
                    self.assertEqual((await client.get("/", headers={"Origin": "https://evil.test"})).status_code, 403)
                    if not hub:
                        self.assertEqual((await client.get("/", headers={"Origin": "http://local"})).status_code, 200)
        asyncio.run(run())

    def test_websocket_denied_before_accept(self):
        async def run():
            messages = []
            async def send(message):
                messages.append(message)
            async def unexpected(*args):
                self.fail("Unauthenticated WebSocket reached backend")
            await BackendAccessMiddleware(unexpected, hub_mode=True, proxy_token="secret")(
                {"type": "websocket", "headers": []}, unexpected, send)
            self.assertEqual(messages, [{"type": "websocket.close", "code": 1008}])
        asyncio.run(run())

    def test_atomic_publication_conflict_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "output.csv"
            with new_output(target) as temporary:
                temporary.write_text("complete")
                self.assertFalse(target.exists())
            self.assertEqual(target.read_text(), "complete")
            with self.assertRaises(FileExistsError):
                with new_output(target) as temporary:
                    temporary.write_text("overwrite")
            self.assertEqual(target.read_text(), "complete")
            other = Path(directory) / "failed.csv"
            with self.assertRaises(RuntimeError):
                with new_output(other) as temporary:
                    temporary.write_text("partial")
                    raise RuntimeError("failed")
            self.assertFalse(other.exists())
            self.assertEqual(list(Path(directory).iterdir()), [target])
