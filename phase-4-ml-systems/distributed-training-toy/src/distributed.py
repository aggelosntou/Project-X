"""
distributed.py — Communication primitives for data-parallel distributed training.

Implements from scratch (using multiprocessing queues):
  - broadcast(tensor, src=0)   — send from rank 0 to all workers
  - all_reduce(tensor, op)     — reduce (sum/mean) across all workers
  - barrier()                  — synchronize all workers
  - ring_all_reduce(tensor)    — bandwidth-optimal ring-allreduce

No torch.distributed.  Pure Python multiprocessing.
"""

import numpy as np
import multiprocessing as mp
import time
from typing import Optional


# ---------------------------------------------------------------------------
# Shared communication context
# ---------------------------------------------------------------------------

class CommContext:
    """
    Holds the queues and barriers that connect all worker processes.

    One CommContext is created by the launcher and passed to all workers.
    Each worker uses its rank to index into the queues.

    Layout:
      send_queues[i][j] — queue from rank i to rank j
      barrier_counter   — shared counter for barrier synchronisation
    """

    def __init__(self, world_size: int):
        self.world_size = world_size
        # NxN grid of queues: send_queues[src][dst]
        self.send_queues = [
            [mp.Queue(maxsize=4) for _ in range(world_size)]
            for _ in range(world_size)
        ]
        # Barrier: each worker increments; when count == world_size, all proceed
        self._barrier_counter  = mp.Value('i', 0)
        self._barrier_lock     = mp.Lock()
        self._barrier_events   = [mp.Event() for _ in range(world_size)]
        self._barrier_phase    = mp.Value('i', 0)   # to handle multiple barriers

    def send(self, src_rank: int, dst_rank: int, data: np.ndarray):
        """Non-blocking send from src to dst."""
        self.send_queues[src_rank][dst_rank].put(data.copy())

    def recv(self, src_rank: int, dst_rank: int) -> np.ndarray:
        """Blocking recv at dst_rank, message from src_rank."""
        return self.send_queues[src_rank][dst_rank].get()

    def barrier(self, rank: int):
        """Block until all workers reach this point."""
        with self._barrier_lock:
            self._barrier_counter.value += 1
            if self._barrier_counter.value == self.world_size:
                # Last worker: wake everyone
                for e in self._barrier_events:
                    e.set()
        self._barrier_events[rank].wait()
        # Reset for next use — only rank 0 resets to avoid race
        with self._barrier_lock:
            if rank == 0:
                for e in self._barrier_events:
                    e.clear()
                self._barrier_counter.value = 0


# ---------------------------------------------------------------------------
# Worker-side API
# ---------------------------------------------------------------------------

class DistributedWorker:
    """
    API available to each worker process.

    Usage:
        dist = DistributedWorker(rank, world_size, comm_ctx)
        dist.broadcast(tensor, src=0)
        dist.all_reduce(tensor, op='sum')
        dist.barrier()
    """

    def __init__(self, rank: int, world_size: int, ctx: CommContext):
        self.rank       = rank
        self.world_size = world_size
        self.ctx        = ctx

    # ------------------------------------------------------------------
    # broadcast
    # ------------------------------------------------------------------

    def broadcast(self, tensor: np.ndarray, src: int = 0) -> np.ndarray:
        """
        Broadcast tensor from rank `src` to all other ranks.
        After this call, all ranks have identical tensor equal to src's tensor.

        Algorithm: src sends to every other rank (naïve O(N) messages,
        O(data) per message).  For our toy implementation this is fine.
        """
        if self.rank == src:
            for dst in range(self.world_size):
                if dst != src:
                    self.ctx.send(src, dst, tensor)
            return tensor.copy()
        else:
            return self.ctx.recv(src, self.rank)

    # ------------------------------------------------------------------
    # all_reduce (naïve: gather at rank 0, reduce, broadcast back)
    # ------------------------------------------------------------------

    def all_reduce(self, tensor: np.ndarray, op: str = 'sum') -> np.ndarray:
        """
        All-reduce: each worker contributes a tensor, all receive the reduced result.

        This naïve implementation: all workers send to rank 0, rank 0 reduces,
        then broadcasts back.  Communication cost: 2*(N-1) messages.

        For the bandwidth-optimal version, see ring_all_reduce().
        """
        if self.rank != 0:
            # Send my tensor to rank 0
            self.ctx.send(self.rank, 0, tensor)
            # Wait to receive the result
            result = self.ctx.recv(0, self.rank)
            return result
        else:
            # Receive from all others and reduce
            accumulated = tensor.copy().astype(np.float64)
            for src in range(1, self.world_size):
                received = self.ctx.recv(src, 0)
                if op == 'sum' or op == 'mean':
                    accumulated += received.astype(np.float64)
                elif op == 'max':
                    accumulated = np.maximum(accumulated, received)
                elif op == 'min':
                    accumulated = np.minimum(accumulated, received)

            if op == 'mean':
                accumulated /= self.world_size

            result = accumulated.astype(tensor.dtype)
            # Broadcast result to all
            for dst in range(1, self.world_size):
                self.ctx.send(0, dst, result)
            return result

    # ------------------------------------------------------------------
    # ring_all_reduce — bandwidth-optimal
    # ------------------------------------------------------------------

    def ring_all_reduce(self, tensor: np.ndarray) -> np.ndarray:
        """
        Ring-allreduce: each worker passes chunks around the ring.

        Phases:
          1. Reduce-scatter: N-1 rounds, each worker sends 1/N of data to right
             neighbor, accumulates received data.  After this, each worker holds
             the fully-reduced result for its own chunk.
          2. All-gather: N-1 rounds, each worker sends its chunk to right neighbor.
             After this, all workers have the full reduced tensor.

        Bandwidth per worker per phase = (N-1)/N * data_size.
        Total bandwidth = 2*(N-1)/N * data_size  (approaches 2 for large N).
        This is optimal: you cannot do less communication than sending each byte
        to/from every worker once.

        Time complexity: O(N) rounds, but each round carries 1/N of data.
        Bandwidth efficiency: 100% for large N (each link saturated).
        """
        N = self.world_size
        rank = self.rank
        n_elements = tensor.size
        chunk_size = (n_elements + N - 1) // N   # ceiling division

        # Pad tensor to multiple of N for clean chunking
        padded_size = chunk_size * N
        flat = np.zeros(padded_size, dtype=np.float64)
        flat[:n_elements] = tensor.flatten()

        chunks = [flat[i*chunk_size:(i+1)*chunk_size].copy() for i in range(N)]
        right  = (rank + 1) % N
        left   = (rank - 1) % N

        # ---- Phase 1: Reduce-scatter ----
        # Each round: send chunk[(rank - round) % N] to right,
        # receive from left and accumulate into chunk[(rank - round - 1) % N]
        for step in range(N - 1):
            send_idx = (rank - step) % N
            recv_idx = (rank - step - 1) % N
            self.ctx.send(rank, right, chunks[send_idx])
            received = self.ctx.recv(left, rank)
            chunks[recv_idx] += received

        # ---- Phase 2: All-gather ----
        # Each round: send chunk[(rank - round + 1) % N] to right,
        # receive from left and store into chunk[(rank - round) % N]
        for step in range(N - 1):
            send_idx = (rank - step + 1) % N
            recv_idx = (rank - step) % N
            self.ctx.send(rank, right, chunks[send_idx])
            received = self.ctx.recv(left, rank)
            chunks[recv_idx] = received

        # Reconstruct tensor
        result_flat = np.concatenate(chunks)[:n_elements]
        return result_flat.reshape(tensor.shape).astype(tensor.dtype)

    # ------------------------------------------------------------------
    # barrier
    # ------------------------------------------------------------------

    def barrier(self):
        self.ctx.barrier(self.rank)
