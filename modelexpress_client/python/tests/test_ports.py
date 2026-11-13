# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for client base-port auto-assignment (modelexpress.ports).

The ModelExpress client derives its per-rank listener ports as
``MX_METADATA_PORT + device_id`` and ``MX_WORKER_GRPC_PORT + device_id``.
Each bind site auto-picks one block of consecutive free ports for its base
variable right before binding, so the probe-to-bind window stays minimal.
Each base anchors 500 8-aligned candidate blocks; a time-seeded random
pick staggers concurrently starting instances, and the probe scan skips
occupied blocks -- including blocks this process already reserved for the
other variable, which nothing binds yet.
"""

from __future__ import annotations

import random
from unittest.mock import patch

import pytest

from modelexpress.ports import (
    CLIENT_PORTS_PER_INSTANCE,
    METADATA_PORT_BASE,
    PORT_BLOCK_CANDIDATES,
    WORKER_GRPC_PORT_BASE,
    ensure_client_port_block,
    find_free_port_block,
)
from modelexpress import envs

MX_VARS = ("MX_METADATA_PORT", "MX_WORKER_GRPC_PORT")
RANGE_TOP = PORT_BLOCK_CANDIDATES * CLIENT_PORTS_PER_INSTANCE


class TestFindFreePortBlock:
    def _find(self, free_fn, count, port_range, search_start=None, exclude=()):
        with patch(
            "modelexpress.ports._is_port_available", side_effect=free_fn
        ):
            return find_free_port_block(count, port_range, search_start, exclude)

    def test_picks_lowest_fully_free_block(self):
        # Ports 4000..4009 are busy, so the aligned blocks at 4000 and 4008
        # are spoiled and the first fully-free block starts at 4016.
        port = self._find(lambda p: p >= 4010, 8, (4000, 6000))
        assert port == 4016

    def test_skips_block_interrupted_by_single_busy_port(self):
        # 4007 is busy, spoiling the aligned block at 4000; the next aligned
        # block at 4008 is fully free.
        port = self._find(
            lambda p: 4000 <= p <= 4006 or p >= 4008, 8, (4000, 6000)
        )
        assert port == 4008

    def test_block_at_range_end_is_accepted(self):
        # The last aligned block inside [4000, 6000] spans 5992..5999.
        port = self._find(lambda p: p >= 5992, 8, (4000, 6000))
        assert port == 5992

    def test_scan_starts_at_search_start(self):
        # Everything below 4334 is busy: the search start floors onto the
        # aligned block at 4328 (spoiled) and the scan must continue to the
        # next aligned block, 4336.
        port = self._find(lambda p: p >= 4334, 8, (4000, 5000), search_start=4334)
        assert port == 4336

    def test_scan_wraps_to_window_bottom(self):
        # Only 4000..4007 are free; a scan starting at 4334 must wrap past
        # the busy tail of the window and find them.
        port = self._find(
            lambda p: 4000 <= p <= 4007, 8, (4000, 5000), search_start=4334
        )
        assert port == 4000

    def test_run_does_not_continue_across_wrap_seam(self):
        # Ports 4994..4999 (a 6-wide run) and 4000..4007 (an 8-wide run) are
        # free. Aligned blocks never straddle the range edge by construction,
        # and the 6-wide tail run is never a candidate; the scan starting at
        # 4990 floors to block 4488, cycles past the busy region, and wraps
        # to the block found after the seam: 4000.
        def free(p):
            return 4994 <= p <= 4999 or 4000 <= p <= 4007

        port = self._find(free, 8, (4000, 5000), search_start=4990)
        assert port == 4000

    def test_out_of_range_search_start_is_normalized(self):
        # A search start above the window wraps in and floors onto the grid:
        # 5334 -> block 4328 in [4000, 5000), so the first free block is 4336.
        port = self._find(lambda p: p >= 4335, 8, (4000, 5000), search_start=5334)
        assert port == 4336

    def test_excluded_ports_are_treated_as_busy(self):
        # Every probe says free, but the caller reserves 4000..4007 for
        # another variable: the first aligned block must start at 4008.
        port = self._find(
            lambda p: True, 8, (4000, 5000), exclude=range(4000, 4008)
        )
        assert port == 4008

    def test_returned_blocks_are_count_aligned(self):
        # Whatever the occupancy pattern, the returned base always lies on
        # the count-aligned grid anchored at the range start. That invariant
        # is what keeps concurrently probing ranks collision-free (see the
        # race test below).
        for free_fn in (
            lambda p: p >= 4010,
            lambda p: p >= 4013,
            lambda p: p >= 4336,
            lambda p: True,
        ):
            port = self._find(free_fn, 8, (4000, 6000))
            assert (port - 4000) % 8 == 0

    def test_racing_ranks_cannot_pick_colliding_ports(self):
        # Regression test for the sliding-scan race: with non-aligned bases,
        # a rank that found base 10 binds 10 + device_id, and another rank
        # seeing 11 busy could floor-slide to base 12 and bind 12 + 0 --
        # the same port a base-10 rank with device_id 2 binds. With the
        # aligned grid this is structurally impossible: ranks probing the
        # same snapshot (only port 9 bound, 0..7 busy) all land on the
        # block at 16, and their base + device_id ports stay distinct.
        free_fn = lambda p: p >= 8 and p != 9
        base = self._find(free_fn, 8, (0, 40))
        assert base == 16
        # rank1 bound 9 earlier from the block at 8; rank0 and rank2 probe
        # the stale snapshot and share the new block at 16.
        bound_ports = {9, base + 0, base + 2}
        assert len(bound_ports) == 3

    def test_raises_when_no_block_fits(self):
        # Alternating free/busy ports never form a run of 8.
        with pytest.raises(RuntimeError):
            self._find(lambda p: p % 2 == 0, 8, (4000, 5000))

    def test_rejects_invalid_count_and_range(self):
        with pytest.raises(ValueError):
            find_free_port_block(0, (4000, 5000))
        with pytest.raises(ValueError):
            find_free_port_block(8, (4000, 4004))

    def test_finds_a_real_ephemeral_port(self):
        # Sanity check against the real bind probe: any single free port in
        # the ephemeral range satisfies the request, from any start point.
        port = find_free_port_block(1, (32768, 65535), search_start=50000)
        assert 32768 <= port <= 65535


class TestEnsureClientPortBlock:
    @pytest.fixture(autouse=True)
    def _isolate_env_and_blocks(self, monkeypatch):
        for key in MX_VARS:
            monkeypatch.delenv(key, raising=False)
        # Each test starts with no in-process reservations.
        monkeypatch.setattr("modelexpress.ports._assigned_blocks", {})
        yield
        for key in MX_VARS:
            monkeypatch.delenv(key, raising=False)

    def _reset(self, monkeypatch):
        """Clear both variables and in-process reservations (per iteration)."""
        for key in MX_VARS:
            monkeypatch.delenv(key, raising=False)
        monkeypatch.setattr("modelexpress.ports._assigned_blocks", {})

    def _capture_scan_calls(self, monkeypatch, env_var, seed):
        """Run one assignment with a fixed time seed, stubbing out
        ``find_free_port_block``.

        Returns the (count, port_range, search_start, exclude) tuples the
        caller passed for the env var. The stub returns each
        ``search_start``, so the call is fully described.
        """
        captured = []

        def fake_find(count, port_range, search_start=None, exclude=()):
            captured.append((count, port_range, search_start, frozenset(exclude)))
            return search_start

        monkeypatch.setattr("modelexpress.ports.time.time_ns", lambda: seed)
        monkeypatch.setattr("modelexpress.ports.find_free_port_block", fake_find)
        ensure_client_port_block(env_var)
        return captured

    def test_start_is_rand_times_8_plus_base(self, monkeypatch):
        # With a pinned time seed the pick is reproducible: the start is
        # exactly Random(seed).randrange(500) * 8 + base.
        for seed in (12345, 987654321, 42):
            for env_var, base in (
                ("MX_METADATA_PORT", METADATA_PORT_BASE),
                ("MX_WORKER_GRPC_PORT", WORKER_GRPC_PORT_BASE),
            ):
                self._reset(monkeypatch)
                calls = self._capture_scan_calls(monkeypatch, env_var, seed)
                expected = (
                    random.Random(seed).randrange(PORT_BLOCK_CANDIDATES)
                    * CLIENT_PORTS_PER_INSTANCE
                )
                assert calls == [
                    (
                        8,
                        (base, base + RANGE_TOP - 1),
                        base + expected,
                        frozenset(),
                    )
                ]

    def test_second_assignment_excludes_first_block(self, monkeypatch):
        # Assigning both variables in one process: the gRPC search skips the
        # metadata block reserved earlier, which nothing binds yet.
        seed = 20261112
        captured = []

        def fake_find(count, port_range, search_start=None, exclude=()):
            captured.append((count, port_range, search_start, frozenset(exclude)))
            return search_start

        monkeypatch.setattr("modelexpress.ports.time.time_ns", lambda: seed)
        monkeypatch.setattr("modelexpress.ports.find_free_port_block", fake_find)
        ensure_client_port_block("MX_METADATA_PORT")
        ensure_client_port_block("MX_WORKER_GRPC_PORT")
        assert len(captured) == 2
        # The exclude of the second search is the first block, and each
        # search started at its own variable's base-derived offset.
        metadata_block = range(
            envs.MX_METADATA_PORT,
            envs.MX_METADATA_PORT + CLIENT_PORTS_PER_INSTANCE,
        )
        assert captured[1][3] == frozenset(metadata_block)

    def test_random_starts_are_aligned_and_inside_ranges(self, monkeypatch):
        # Whatever the seed: 8-aligned, inside the range, for both variables.
        for seed in (0, 1, 7, 20261111, 999999999999):
            for env_var, base in (
                ("MX_METADATA_PORT", METADATA_PORT_BASE),
                ("MX_WORKER_GRPC_PORT", WORKER_GRPC_PORT_BASE),
            ):
                self._reset(monkeypatch)
                calls = self._capture_scan_calls(monkeypatch, env_var, seed)
                assert len(calls) == 1
                count, port_range, search_start, _ = calls[0]
                assert count == 8
                assert port_range == (base, base + RANGE_TOP - 1)
                assert (search_start - base) % 8 == 0
                assert 0 <= search_start - base < RANGE_TOP

    def test_assigns_disjoint_blocks_inside_random_ranges(self):
        # Real bind probe: the blocks must land inside the reserved ranges
        # and must be disjoint port ranges.
        ensure_client_port_block("MX_METADATA_PORT")
        ensure_client_port_block("MX_WORKER_GRPC_PORT")
        metadata = envs.MX_METADATA_PORT
        grpc = envs.MX_WORKER_GRPC_PORT
        for port, base in (
            (metadata, METADATA_PORT_BASE),
            (grpc, WORKER_GRPC_PORT_BASE),
        ):
            assert base <= port <= base + RANGE_TOP - CLIENT_PORTS_PER_INSTANCE
        assert metadata != grpc
        assert (
            metadata + CLIENT_PORTS_PER_INSTANCE <= grpc
            or grpc + CLIENT_PORTS_PER_INSTANCE <= metadata
        )

    def test_assignment_is_idempotent_within_process(self):
        # A second ensure at a later bind site must keep the first pick:
        # every listener in this process derives from one base.
        ensure_client_port_block("MX_METADATA_PORT")
        first = envs.MX_METADATA_PORT
        with patch("modelexpress.ports.find_free_port_block") as find:
            ensure_client_port_block("MX_METADATA_PORT")
            find.assert_not_called()
        assert envs.MX_METADATA_PORT == first

    def test_no_free_block_raises_with_env_var_name(self, monkeypatch):
        # A range with no run of free ports must abort the load with an
        # error naming the env var the operator can set manually.
        def unavailable(*args, **kwargs):
            raise RuntimeError("No run of 8 consecutive free ports found")

        monkeypatch.setattr("modelexpress.ports.find_free_port_block", unavailable)
        with pytest.raises(RuntimeError) as excinfo:
            ensure_client_port_block("MX_METADATA_PORT")
        assert "Failed to auto-assign MX_METADATA_PORT" in str(excinfo.value)

        # With the metadata port pre-set, the worker gRPC range failure
        # surfaces with its own env var name.
        monkeypatch.setenv("MX_METADATA_PORT", "4321")
        with pytest.raises(RuntimeError) as excinfo:
            ensure_client_port_block("MX_WORKER_GRPC_PORT")
        assert "Failed to auto-assign MX_WORKER_GRPC_PORT" in str(excinfo.value)

    def test_user_provided_values_are_preserved(self, monkeypatch):
        # Explicit values always win: no search, no overwrite.
        monkeypatch.setenv("MX_METADATA_PORT", "4321")
        monkeypatch.setenv("MX_WORKER_GRPC_PORT", "6543")
        with patch("modelexpress.ports.find_free_port_block") as find:
            ensure_client_port_block("MX_METADATA_PORT")
            ensure_client_port_block("MX_WORKER_GRPC_PORT")
            find.assert_not_called()
        assert envs.MX_METADATA_PORT == 4321
        assert envs.MX_WORKER_GRPC_PORT == 6543
