# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Client base-port discovery for the per-worker listener ports.

The ModelExpress client derives its per-rank listener ports as
``MX_METADATA_PORT + device_id`` and ``MX_WORKER_GRPC_PORT + device_id``.
When either variable is unset, the first site that is about to bind ports
derived from it auto-picks a block of consecutive free ports and exports
the base through ``os.environ``, so every later read in this process (all
via :mod:`modelexpress.envs`) agrees without any launcher-side wiring.

Doing this at the bind site rather than in the engine launcher keeps the
probe-to-bind window to milliseconds: a block found free here is bound
immediately after, instead of being claimed at process start and left
unbound for the minutes the engine spends on CUDA init and model loading.
Within one instance each rank binds ``base + device_id`` from its own
base; because every base is ``count``-aligned on a grid anchored at the
range start, two ranks can only derive the same port by holding equal
device_ids, so ranks that probe concurrently and race between probe and
bind can never collide. The time-seeded random search start staggers
concurrently starting instances on a shared network namespace.

Explicit values always win: assignment only runs while the variable is
unset, so a deployment (e.g. the port-named Kubernetes Service topology)
that pins the base keeps working unchanged.
"""

from __future__ import annotations

import logging
import os
import random
import socket
import time
from typing import FrozenSet, Optional, Tuple

logger = logging.getLogger("modelexpress.ports")

# Ports per instance: one per rank/device the client serves listeners for.
CLIENT_PORTS_PER_INSTANCE = 8
# Search windows: 500 8-aligned candidate blocks per variable.
METADATA_PORT_BASE = 2000
WORKER_GRPC_PORT_BASE = 6000
PORT_BLOCK_CANDIDATES = 500

_PORT_BLOCK_TOP = PORT_BLOCK_CANDIDATES * CLIENT_PORTS_PER_INSTANCE

# Base port of each auto-assignable variable.
_PORT_VAR_BASES = {
    "MX_METADATA_PORT": METADATA_PORT_BASE,
    "MX_WORKER_GRPC_PORT": WORKER_GRPC_PORT_BASE,
}

# Blocks this process auto-assigned, so a later assignment excludes them
# (treated as occupied even though nothing binds them yet).
_assigned_blocks: dict[str, range] = {}


def _is_port_available(port: int) -> bool:
    """Return whether a port is available on all configured address families."""
    try:
        infos = socket.getaddrinfo(
            None,
            port,
            socket.AF_UNSPEC,
            socket.SOCK_STREAM,
            0,
            socket.AI_ADDRCONFIG | socket.AI_PASSIVE,
        )
        # One probe per address family: getaddrinfo may return duplicates.
        deduped: list[tuple] = []
        seen_families: set[socket.AddressFamily] = set()
        for info in infos:
            if info[0] not in seen_families:
                seen_families.add(info[0])
                deduped.append(info)
    except socket.gaierror:
        deduped = [(socket.AF_INET, socket.SOCK_STREAM, 0, "", ("0.0.0.0", port))]
    try:
        for family, socktype, proto, _, sockaddr in deduped:
            sock = socket.socket(family, socktype, proto)
            try:
                if family == socket.AF_INET6:
                    sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 1)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(sockaddr)
            finally:
                sock.close()
        return True
    except (OSError, OverflowError):
        return False


def find_free_port_block(
    count: int,
    port_range: Tuple[int, int],
    search_start: Optional[int] = None,
    exclude: FrozenSet[int] | set[int] = frozenset(),
) -> int:
    """Return the first port of ``count`` consecutive free TCP ports.

    Probes each port of ``port_range`` (inclusive) once, then checks
    candidate blocks in cyclic order starting at ``search_start`` (an
    out-of-range start wraps in). Candidate blocks are ``count``-aligned
    on a grid anchored at the range start, so no block ever crosses
    ``end`` and no port belongs to more than one candidate. Ports in
    ``exclude`` are treated as occupied even when the probe says they
    are free -- used for blocks reserved for another variable but not
    bound yet. Raises ``ValueError`` for a bad count/range and
    ``RuntimeError`` when no block fits.

    The alignment is what makes per-rank assignment race-safe: every
    consumer binds ``base + device_id`` with ``device_id < count``, and
    two aligned bases can only produce the same port when the
    ``device_id`` offsets are equal. A sliding (non-aligned) scan would
    let "base + 1" and "base + 3 - 2" collide across ranks that probe
    the same snapshot before any of them binds.
    """
    start, end = port_range
    span = end - start + 1
    if count <= 0 or span < count:
        raise ValueError(f"Invalid count={count} for port range [{start}, {end}]")
    free = [
        p not in exclude and _is_port_available(p) for p in range(start, end + 1)
    ]
    blocks = span // count
    first = 0 if search_start is None else ((search_start - start) % span) // count
    for i in range(blocks):
        s = ((first + i) % blocks) * count
        if all(free[s : s + count]):
            return start + s
    raise RuntimeError(
        f"No run of {count} consecutive free ports found in range "
        f"[{start}, {end}] starting at {start + first * count}. Free some "
        f"ports or set the base port explicitly via the environment variable."
    )


def ensure_client_port_block(env_var: str) -> None:
    """Auto-pick the base port for ``env_var`` (only when unset).

    Call at the site that is about to bind ``base + device_id`` listeners,
    then read the base through :mod:`modelexpress.envs` as usual. A
    time-seeded 8-aligned random pick (``rand * 8 + base``) staggers
    concurrently starting instances; the probe scan then skips occupied
    blocks from there -- including blocks this process already reserved
    for the other variable, which nothing binds until the client starts.

    Writes the result into ``os.environ`` so all later reads in this
    process see the same base, and aborts the load rather than colliding
    when no free block remains in the range.
    """
    try:
        base = _PORT_VAR_BASES[env_var]
    except KeyError:
        raise ValueError(f"Unknown client port variable {env_var!r}") from None

    if env_var in os.environ:
        return

    rng = random.Random(time.time_ns())
    offset = rng.randrange(PORT_BLOCK_CANDIDATES) * CLIENT_PORTS_PER_INSTANCE
    reserved_ports = {
        port
        for block in _assigned_blocks.values()
        for port in block
    }
    try:
        port = find_free_port_block(
            CLIENT_PORTS_PER_INSTANCE,
            (base, base + _PORT_BLOCK_TOP - 1),
            search_start=base + offset,
            exclude=reserved_ports,
        )
    except RuntimeError as exc:
        # No free run left in the range: abort rather than collide.
        raise RuntimeError(
            f"Failed to auto-assign {env_var}: {exc} "
            f"Export {env_var} to pin the base port."
        ) from exc
    os.environ[env_var] = str(port)
    _assigned_blocks[env_var] = range(port, port + CLIENT_PORTS_PER_INSTANCE)
    logger.info(
        "Auto-assigned %s=%d (found free port block [%d, %d) "
        "in search range [%d, %d))",
        env_var,
        port,
        port,
        port + CLIENT_PORTS_PER_INSTANCE,
        base,
        base + _PORT_BLOCK_TOP,
    )
