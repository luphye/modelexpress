# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for MX_PUBLISH_CHECKSUM / MX_VERIFY_CHECKSUM wiring.

The fingerprint itself is covered by the safe_checksum tests in
test_tensor_utils; these pin the publish gate and the receive-side compare.
"""

import pytest
import torch

from modelexpress.tensor_utils import safe_checksum
from modelexpress.types import ManifestMismatchError, TensorDescriptor


class TestBuildTensorProtos:
    def _build(self, tensors, monkeypatch, env):
        from modelexpress.metadata.publish import build_tensor_protos

        monkeypatch.setenv("MX_PUBLISH_CHECKSUM", env)
        return build_tensor_protos(tensors, device_id=0, global_rank=0)

    def test_default_publishes_safe_checksum(self, monkeypatch):
        from modelexpress.metadata.publish import build_tensor_protos

        monkeypatch.delenv("MX_PUBLISH_CHECKSUM", raising=False)
        w = torch.randn(4, 4)
        (proto,) = build_tensor_protos({"w": w}, device_id=0, global_rank=0)
        assert proto.HasField("checksum")
        assert proto.checksum == safe_checksum(w)

    def test_opt_out_and_zero_element_leave_field_unset(self, monkeypatch):
        tensors = {"w": torch.randn(4), "empty": torch.empty(0)}
        assert all(not p.HasField("checksum") for p in self._build(tensors, monkeypatch, "0"))
        (empty,) = self._build({"empty": torch.empty(0)}, monkeypatch, "1")
        assert not empty.HasField("checksum")

    def test_env_defaults_on(self, monkeypatch):
        from modelexpress import envs

        for name in ("MX_PUBLISH_CHECKSUM", "MX_VERIFY_CHECKSUM"):
            monkeypatch.delenv(name, raising=False)
            assert getattr(envs, name) is True


class TestVerifyReceivedChecksums:
    @staticmethod
    def _descrs(tensors, with_checksum=True):
        return [
            TensorDescriptor(
                name=n,
                addr=t.data_ptr(),
                size=t.numel() * t.element_size(),
                device_id=0,
                dtype=str(t.dtype),
                checksum=safe_checksum(t) if with_checksum else "",
            )
            for n, t in tensors.items()
        ]

    @staticmethod
    def _manager():
        from modelexpress.nixl_transfer import NixlTransferManager

        return NixlTransferManager("verify-test", device_id=0)

    def test_match_passes(self):
        mgr = self._manager()
        local = {"w": torch.randn(8, 8)}
        mgr._verify_received_checksums(self._descrs(local), local)

    def test_corrupted_bytes_raise_and_name_tensor(self):
        mgr = self._manager()
        local = {"w": torch.randn(8, 8)}
        src = self._descrs(local)
        local["w"][0, 0] += 1.0  # the "transfer" delivered different bytes
        with pytest.raises(ManifestMismatchError, match="checksum mismatch.*'w'"):
            mgr._verify_received_checksums(src, local)

    def test_no_published_digests_warn_and_skip(self, caplog):
        mgr = self._manager()
        local = {"w": torch.randn(8)}
        with caplog.at_level("WARNING", logger="modelexpress.nixl_transfer"):
            # Must not raise: rolling upgrades load from digest-less sources.
            mgr._verify_received_checksums(self._descrs(local, False), local)
        assert any("no digests" in r.message for r in caplog.records)

    def test_unmatched_and_empty_skipped(self):
        mgr = self._manager()
        local = {"w": torch.randn(8)}
        # "ghost" has no local tensor and "empty" publishes no digest;
        # the name-diff machinery owns the former, not the checksum gate.
        src = [
            TensorDescriptor(
                name="w", addr=0, size=32, device_id=0,
                dtype="torch.float32", checksum=safe_checksum(local["w"]),
            ),
            TensorDescriptor(
                name="ghost", addr=0, size=8, device_id=0,
                dtype="torch.float32", checksum="deadbeef",
            ),
        ]
        mgr._verify_received_checksums(src, local)
