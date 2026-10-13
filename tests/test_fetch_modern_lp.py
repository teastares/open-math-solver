#!/usr/bin/env python3
"""离线验证现代 LP 下载的身份、边界、缓存及压缩完整性。"""

import bz2
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(sys.argv[1]).resolve()
sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("fetch_modern_lp", SCRIPT)
FETCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FETCH)


class Response(io.BytesIO):
    def __init__(self, data, url, with_length=True):
        super().__init__(data)
        self.url = url
        self.headers = {"Content-Length": str(len(data))} if with_length else {}

    def geturl(self):
        return self.url


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="modern lp fetch ")
        self.root = Path(self.temporary.name)
        self.output = self.root / "models"
        self.lock_path = self.root / "lock.json"
        self.model = b"NAME TEST\nROWS\n N OBJ\nCOLUMNS\n X OBJ 1\nENDATA\n"
        self.compressed = bz2.compress(self.model)
        self.lock = {"schema_version": 1, "role": "development_smoke", "models": [{
            "name": "model.mps", "url": FETCH.SOURCE + "model.mps.bz2",
            "bytes": len(self.model), "sha256": hashlib.sha256(self.model).hexdigest(),
            "compressed_bytes": len(self.compressed), "compressed_sha256": hashlib.sha256(self.compressed).hexdigest()}]}
        self.write_lock()

    def tearDown(self):
        self.temporary.cleanup()

    def write_lock(self):
        self.lock_path.write_text(json.dumps(self.lock), encoding="utf-8")

    def opener(self, url, **kwargs):
        return Response(self.compressed, url)

    def fetch(self, opener=None):
        return FETCH.fetch(self.lock_path, self.output, opener=opener or self.opener)

    def assert_no_partial(self):
        self.assertEqual(list(self.root.rglob("*.partial")), [])

    def change_compressed(self, value):
        self.compressed = value
        self.lock["models"][0]["compressed_bytes"] = len(value)
        self.lock["models"][0]["compressed_sha256"] = hashlib.sha256(value).hexdigest()
        self.write_lock()

    def test_valid_download_and_cached_revalidation(self):
        result = self.fetch()
        self.assertEqual((result["new_files"], result["new_archives"]), (1, 1))
        self.assertEqual((self.output / "model.mps").read_bytes(), self.model)
        self.assert_no_partial()

    def test_cached_run_does_not_download(self):
        self.fetch()
        opener = mock.Mock(side_effect=AssertionError("缓存复核不能联网"))
        result = self.fetch(opener)
        self.assertEqual((result["new_files"], result["new_archives"]), (0, 0))
        opener.assert_not_called()
        self.assert_no_partial()

    def test_bad_compressed_hash_is_not_published(self):
        self.lock["models"][0]["compressed_sha256"] = "0" * 64
        self.write_lock()
        with self.assertRaises(ValueError):
            self.fetch()
        self.assertFalse((self.output / ".archives" / "model.mps.bz2").exists())
        self.assertFalse((self.output / "model.mps").exists())
        self.assert_no_partial()

    def test_bad_model_hash_is_not_published(self):
        self.lock["models"][0]["sha256"] = "0" * 64
        self.write_lock()
        with self.assertRaises(ValueError):
            self.fetch()
        self.assertFalse((self.output / "model.mps").exists())
        self.assert_no_partial()

    def test_bad_paths_urls_and_duplicate_names_are_rejected_before_network(self):
        initial = copy.deepcopy(self.lock)
        for name in ["../escape.mps", "sub/model.mps", "sub\\model.mps", "D:escape.mps", "CON.mps", "/escape.mps"]:
            self.lock = copy.deepcopy(initial)
            self.lock["models"][0]["name"] = name
            self.write_lock()
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.fetch()
            self.assertFalse(self.output.exists())
        self.lock = copy.deepcopy(initial)
        self.lock["models"][0]["url"] = "http://127.0.0.1/model.mps.bz2"
        self.write_lock()
        with self.assertRaises(ValueError):
            self.fetch()
        self.lock = copy.deepcopy(initial)
        self.lock["models"].append(copy.deepcopy(self.lock["models"][0]))
        self.write_lock()
        with self.assertRaises(ValueError):
            self.fetch()
        self.assertFalse(self.output.exists())

    def test_total_budget_is_checked_before_network(self):
        for field, size in [("compressed_bytes", FETCH.COMPRESSED_LIMIT + 1), ("bytes", FETCH.EXPANDED_LIMIT + 1)]:
            previous = self.lock["models"][0][field]
            self.lock["models"][0][field] = size
            self.write_lock()
            opener = mock.Mock(side_effect=AssertionError("超限清单不能联网"))
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.fetch(opener)
            opener.assert_not_called()
            self.assertFalse(self.output.exists())
            self.lock["models"][0][field] = previous

    def test_actual_download_budget_is_enforced_without_content_length(self):
        with self.assertRaises(ValueError):
            self.fetch(lambda url, **kwargs: Response(self.compressed + b"extra", url, with_length=False))
        self.assertFalse((self.output / ".archives" / "model.mps.bz2").exists())
        self.assert_no_partial()

    def test_expansion_budget_and_complete_bz2_are_enforced(self):
        for compressed in [self.compressed[:-3], self.compressed + b"extra",
                           self.compressed + bz2.compress(b"second stream"), bz2.compress(b"a" * 10000)]:
            with self.subTest(size=len(compressed)), tempfile.TemporaryDirectory(dir=self.root) as directory:
                self.output = Path(directory) / "models"
                self.change_compressed(compressed)
                with self.assertRaises((ValueError, OSError)):
                    self.fetch()
                self.assertFalse((self.output / "model.mps").exists())
                self.assert_no_partial()

    def test_existing_modified_files_are_never_overwritten(self):
        self.fetch()
        target = self.output / "model.mps"
        target.write_bytes(b"user content")
        opener = mock.Mock(side_effect=AssertionError("现有文件冲突不能联网"))
        with self.assertRaises(ValueError):
            self.fetch(opener)
        self.assertEqual(target.read_bytes(), b"user content")
        opener.assert_not_called()

    def test_redirect_and_network_failure_leave_no_partial(self):
        for opener in [lambda url, **kwargs: Response(self.compressed, "https://example.com/changed"),
                       mock.Mock(side_effect=OSError("模拟连接中断"))]:
            with self.subTest(opener=opener), self.assertRaises((ValueError, OSError)):
                self.fetch(opener)
            self.assertFalse((self.output / ".archives" / "model.mps.bz2").exists())
            self.assert_no_partial()

    def test_publication_conflict_preserves_existing_file(self):
        target = self.root / "target.mps"
        temporary = self.root / "temporary.mps"
        target.write_bytes(b"user content")
        temporary.write_bytes(self.model)
        with self.assertRaises(ValueError):
            FETCH.publish(temporary, target, FETCH.file_identity(temporary))
        self.assertEqual(target.read_bytes(), b"user content")


if __name__ == "__main__":
    unittest.main()
