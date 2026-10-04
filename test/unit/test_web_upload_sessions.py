"""Capability-bound chunk uploads; fixtures and persistence are temporary only."""
import asyncio
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import os

from fastapi import HTTPException
from PIL import Image
from starlette.datastructures import UploadFile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "web"))
import state
from routers import files


class Body:
    def __init__(self, token, blocks):
        self.headers = {"x-upload-token": token}
        self.blocks = blocks
        self.reads = 0

    async def stream(self):
        for block in self.blocks:
            self.reads += 1
            yield block


class Multipart:
    def __init__(self, file):
        self.file = file

    async def form(self, **limits):
        return {"username": "owner", "file": self.file}


class WatchedFile(UploadFile):
    def __init__(self, content):
        memory = BytesIO(content)
        memory._rolled = False  # Match Starlette's in-memory spooled-file path.
        super().__init__(memory, filename="image.png")
        self.read_sizes = []

    async def read(self, size=-1):
        self.read_sizes.append(size)
        return await super().read(size)


class UploadSessionTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="upload-test-")))
        self.stack.enter_context(patch.object(state, "DATA_DIR", root))
        self.stack.enter_context(patch.object(state, "JUPYTER_MODE", False))
        self.stack.enter_context(patch.dict(state.users, {"owner": {}, "other": {}}, clear=True))
        self.stack.enter_context(patch.dict(state.sessions, {
            name: {"files": {}, "samples": {}, "models": [], "detect_files": {}}
            for name in ("owner", "other")}, clear=True))
        self.stack.enter_context(patch.dict(files._uploads, {}, clear=True))
        image = BytesIO()
        Image.new("L", (2, 3), 12).save(image, format="PNG")
        self.content = image.getvalue()

    def start(self, size=None):
        return asyncio.run(files.upload_start(files.ChunkUploadStart(
            username="owner", filename="sample.png", total_size=len(self.content) if size is None else size)))

    def chunk(self, upload, content=None, offset=0, owner="owner", token=None):
        request = Body(upload["upload_token"] if token is None else token,
                       [self.content if content is None else content])
        return asyncio.run(files.upload_chunk(upload["upload_id"], request, offset, owner))

    def complete(self, upload, **kwargs):
        data = dict(username="owner", upload_id=upload["upload_id"],
                    upload_token=upload["upload_token"], filename="sample.png")
        data.update(kwargs)
        return asyncio.run(files.upload_complete(files.ChunkUploadComplete(**data)))

    def reject(self, status, fn):
        with self.assertRaises(HTTPException) as caught:
            fn()
        self.assertEqual(caught.exception.status_code, status)

    def test_sequential_upload_finalizes_and_revokes_capability(self):
        upload = self.start()
        self.assertEqual(len(upload["upload_id"]), 32)
        self.assertGreaterEqual(len(upload["upload_token"]), 40)
        self.chunk(upload, self.content[:10])
        self.chunk(upload, self.content[10:], offset=10)
        info = self.complete(upload)
        self.assertEqual(Path(info["path"]).read_bytes(), self.content)
        self.assertEqual((info["width"], info["height"]), (2, 3))
        self.assertEqual(tuple(info["source_version"].values()), files._file_signature(Path(info["path"])))
        self.assertIn(info["id"], state.sessions["owner"]["files"])
        self.assertNotIn(info["id"], files._uploads)
        self.reject(404, lambda: self.chunk(upload))
        self.reject(404, lambda: self.complete(upload))
        self.assertEqual(Path(info["path"]).read_bytes(), self.content)

    def test_wrong_owner_missing_bad_token_and_unknown_id_rejected_before_read(self):
        upload = self.start()
        for owner, token, upload_id in [
            ("other", upload["upload_token"], upload["upload_id"]),
            ("owner", "", upload["upload_id"]),
            ("owner", "bad", upload["upload_id"]),
            ("owner", upload["upload_token"], "unknown")]:
            request = Body(token, [self.content])
            self.reject(404, lambda: asyncio.run(files.upload_chunk(upload_id, request, 0, owner)))
            self.assertEqual(request.reads, 0)
        self.reject(404, lambda: self.complete(upload, username="other"))
        self.reject(404, lambda: self.complete(upload, upload_token=""))

    def test_offsets_and_incomplete_or_mismatched_completion(self):
        upload = self.start()
        self.reject(400, lambda: self.chunk(upload, offset=-1))
        self.reject(409, lambda: self.chunk(upload, offset=2**40))
        self.reject(409, lambda: self.complete(upload))
        self.chunk(upload, self.content[:5])
        self.reject(409, lambda: self.chunk(upload, offset=0))
        self.chunk(upload, self.content[5:], offset=5)
        self.reject(400, lambda: self.complete(upload, filename="different.png"))
        self.reject(400, lambda: self.complete(upload, normalize=False))
        self.assertEqual(self.complete(upload)["filename"], "sample.png")

    def test_limits_reject_before_file_mutation(self):
        for size in (0, -1, files.MAX_UPLOAD_SIZE + 1):
            self.reject(413, lambda: self.start(size))
        upload = self.start()
        path = files._uploads[upload["upload_id"]]["path"]
        self.reject(413, lambda: self.chunk(upload, self.content + b"extra"))
        self.assertEqual(path.stat().st_size, 0)
        with patch.object(files, "MAX_CHUNK_SIZE", 4):
            request = Body(upload["upload_token"], [b"1234", b"5", b"not read"])
            self.reject(413, lambda: asyncio.run(files.upload_chunk(upload["upload_id"], request, 0, "owner")))
            self.assertEqual(request.reads, 2)
        self.reject(400, lambda: self.chunk(upload, b""))
        self.assertEqual(path.stat().st_size, 0)

    def test_expiry_removes_only_own_temporary_file(self):
        upload = self.start()
        record = files._uploads[upload["upload_id"]]
        self.chunk(upload, self.content[:4])
        original = state.get_user_dir("owner") / "original.png"
        original.write_bytes(self.content)
        state.sessions["owner"]["files"]["linked"] = {"path": str(original), "server_path": True}
        with patch.object(files.time, "monotonic", return_value=record["updated"] + files.UPLOAD_TTL):
            self.reject(404, lambda: self.chunk(upload, b"x", offset=4))
        self.assertFalse(record["path"].exists())
        self.assertEqual(original.read_bytes(), self.content)

    def test_existing_completed_upload_is_not_a_chunk_target(self):
        path = state.get_user_dir("owner") / "uploads" / "legacy.png"
        path.write_bytes(self.content)
        self.reject(404, lambda: asyncio.run(files.upload_chunk("legacy", Body("guess", [b"bad"]), 0, "owner")))
        self.assertEqual(path.read_bytes(), self.content)

    def test_capacity_and_restart_fail_closed(self):
        uploads = [self.start() for _ in range(8)]
        self.reject(429, lambda: self.start())
        files._uploads.clear()  # In-flight capabilities deliberately do not survive restart.
        self.reject(404, lambda: self.chunk(uploads[0]))
        self.reject(404, lambda: self.complete(uploads[0]))

    def test_concurrent_same_offset_allows_one_writer(self):
        upload = self.start()

        async def run():
            ready = asyncio.Event()
            count = 0

            class RacingBody(Body):
                async def stream(inner):
                    nonlocal count
                    count += 1
                    if count == 2:
                        ready.set()
                    await ready.wait()
                    yield self.content

            return await asyncio.gather(*[
                files.upload_chunk(upload["upload_id"], RacingBody(upload["upload_token"], []), 0, "owner")
                for _ in range(2)], return_exceptions=True)

        results = asyncio.run(run())
        self.assertEqual(sum(isinstance(result, dict) for result in results), 1)
        errors = [result for result in results if isinstance(result, HTTPException)]
        self.assertEqual([error.status_code for error in errors], [409])
        self.assertEqual(Path(self.complete(upload)["path"]).read_bytes(), self.content)

    def test_invalid_image_not_published(self):
        upload = self.start(size=4)
        self.chunk(upload, b"oops")
        self.reject(400, lambda: self.complete(upload))
        self.assertEqual(state.sessions["owner"]["files"], {})
        self.assertFalse((state.get_user_dir("owner") / "uploads" / (upload["upload_id"] + ".png")).exists())

    def test_multipart_reads_bounded_chunks_and_publishes_valid_snapshot(self):
        file = WatchedFile(self.content)
        with patch.object(files, "MAX_CHUNK_SIZE", 7):
            info = asyncio.run(files.upload_file(Multipart(file)))
        self.assertGreater(len(file.read_sizes), 2)
        self.assertEqual(set(file.read_sizes), {7})
        self.assertTrue(file.file.closed)
        self.assertEqual(len(info["id"]), 32)
        self.assertEqual(Path(info["path"]).read_bytes(), self.content)
        self.assertEqual(tuple(info["source_version"].values()), files._file_signature(Path(info["path"])))
        self.assertEqual(list(Path(info["path"]).parent.glob("*.part")), [])

    def test_preview_persists_changed_dimensions_even_with_same_frame_count(self):
        info = asyncio.run(files.upload_file(Multipart(WatchedFile(self.content))))
        Image.new("L", (4, 5), 20).save(info["path"])
        with patch.object(state, "save_user_session", wraps=state.save_user_session) as save:
            result = asyncio.run(files.get_frame("owner", info["id"], 0))
        save.assert_called_once_with("owner")
        self.assertEqual((result["width"], result["height"]), (4, 5))
        self.assertEqual(tuple(info["source_version"].values()), files._file_signature(Path(info["path"])))

    def test_multipart_oversize_invalid_and_empty_clean_only_temporary_files(self):
        uploads = state.get_user_dir("owner") / "uploads"
        original = uploads / "original.png"
        original.write_bytes(self.content)
        for content, maximum, expected in [(self.content, 10, 413), (b"oops", 100, 400), (b"", 100, 400)]:
            file = WatchedFile(content)
            with patch.object(files, "MAX_UPLOAD_SIZE", maximum), patch.object(files, "MAX_CHUNK_SIZE", 7):
                self.reject(expected, lambda: asyncio.run(files.upload_file(Multipart(file))))
            self.assertTrue(file.file.closed)
            self.assertEqual(set(uploads.iterdir()), {original})
            self.assertEqual(original.read_bytes(), self.content)

    def test_atomic_publish_race_never_overwrites_existing_destination(self):
        upload = self.start()
        self.chunk(upload)
        real_link = os.link
        destinations = []

        def race(source, destination, **kwargs):
            Path(destination).write_bytes(b"other result")
            destinations.append(Path(destination))
            return real_link(source, destination, **kwargs)

        with patch.object(files.os, "link", side_effect=race):
            self.reject(409, lambda: self.complete(upload))
            self.reject(409, lambda: asyncio.run(files.upload_file(Multipart(WatchedFile(self.content)))))
        self.assertEqual(len(destinations), 2)
        self.assertTrue(all(path.read_bytes() == b"other result" for path in destinations))
        self.assertEqual(state.sessions["owner"]["files"], {})

    def test_invalid_tdms_not_published_or_registered(self):
        upload = asyncio.run(files.upload_start(files.ChunkUploadStart(
            username="owner", filename="bad.tdms", total_size=4)))
        self.chunk(upload, b"oops")

        def invalid(path, info):
            info["error"] = "Invalid TDMS"
            return info

        with patch.object(files, "parse_tdms_info", side_effect=invalid):
            self.reject(400, lambda: self.complete(upload, filename="bad.tdms"))
            file = WatchedFile(b"oops")
            file.filename = "bad.tdms"
            self.reject(400, lambda: asyncio.run(files.upload_file(Multipart(file))))
        self.assertIn(upload["upload_id"], files._uploads)
        self.assertEqual(state.sessions["owner"]["files"], {})
        self.assertEqual(list((state.get_user_dir("owner") / "uploads").glob("*.tdms")), [])

    def test_replaced_temporary_file_cannot_write_or_expire_external_target(self):
        upload = self.start()
        record = files._uploads[upload["upload_id"]]
        outside = state.get_user_dir("owner") / "original.png"
        outside.write_bytes(self.content)
        record["path"].unlink()
        record["path"].symlink_to(outside)
        self.reject(409, lambda: self.chunk(upload))
        with patch.object(files.time, "monotonic", return_value=record["updated"] + files.UPLOAD_TTL):
            files._expire_uploads()
        self.assertEqual(outside.read_bytes(), self.content)
        self.assertTrue(record["path"].is_symlink())


if __name__ == "__main__":
    unittest.main()
