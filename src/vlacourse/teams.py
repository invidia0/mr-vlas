"""Multi-robot task allocation (book 14.4; Lab 16): Bertsekas' auction algorithm, greedy, and brute force."""

import itertools

import numpy as np


def auction(cost, eps=None):
    """Min-cost assignment of n robots to m >= n tasks by the auction algorithm (Bertsekas, 1988).

    Each unassigned robot bids for its best task at current prices, raising that task's price by the gap
    between its best and second-best value plus eps; the previous holder is outbid. Terminates with an
    assignment within n * eps of optimal. Returns (assignment[robot] = task, number of bids).
    """
    benefit = -np.asarray(cost, float)
    n, m = benefit.shape
    assert m >= n, "at least as many tasks as robots"
    eps = eps if eps is not None else 1e-3 * (np.ptp(benefit) + 1e-9) / n
    price, owner = np.zeros(m), -np.ones(m, int)
    task = -np.ones(n, int)
    bids = 0
    while (task < 0).any():
        i = int(np.flatnonzero(task < 0)[0])
        v = benefit[i] - price
        j = int(np.argmax(v))
        second = np.partition(v, -2)[-2] if m > 1 else v[j]
        price[j] += v[j] - second + eps
        if owner[j] >= 0:
            task[owner[j]] = -1
        owner[j], task[i] = i, j
        bids += 1
    return task, bids


def greedy(cost):
    """Repeatedly assign the globally cheapest remaining (robot, task) pair."""
    cost = np.asarray(cost, float).copy()
    n, _ = cost.shape
    task = -np.ones(n, int)
    for _ in range(n):
        i, j = np.unravel_index(np.argmin(cost), cost.shape)
        task[i] = j
        cost[i, :], cost[:, j] = np.inf, np.inf
    return task


def brute_force(cost):
    """Exact min-cost assignment by enumeration (small problems only; used to test the others)."""
    cost = np.asarray(cost, float)
    n, m = cost.shape
    best = min(itertools.permutations(range(m), n), key=lambda p: sum(cost[i, j] for i, j in enumerate(p)))
    return np.array(best)


def total(cost, task, makespan=False):
    c = np.asarray(cost)[np.arange(len(task)), task]
    return c.max() if makespan else c.sum()


class LossyChannel:
    """Broadcast with distance-dependent delivery, i.i.d. loss and a fixed latency in steps (book 14.2)."""

    def __init__(self, n, p_loss=0.1, range_=np.inf, latency=1, seed=0):
        self.n, self.p_loss, self.range, self.latency = n, p_loss, range_, latency
        self.queue, self.rng, self.sent, self.delivered = [], np.random.default_rng(seed), 0, 0

    def send(self, t, sender, payload, positions):
        for j in range(self.n):
            if j == sender:
                continue
            self.sent += 1
            in_range = np.linalg.norm(np.asarray(positions[sender]) - np.asarray(positions[j])) <= self.range
            if in_range and self.rng.random() >= self.p_loss:
                self.queue.append((t + self.latency, j, sender, payload))

    def receive(self, t, j):
        got = [(s, p) for (td, r, s, p) in self.queue if r == j and td <= t]
        self.queue = [q for q in self.queue if not (q[1] == j and q[0] <= t)]
        self.delivered += len(got)
        return got


def decentralised_auction(cost, channel, positions, rounds=30, eps=None, rebroadcast=False):
    """Each robot runs the auction with its *local* view of prices and owners, updated only by the bids it
    receives over `channel`. With rebroadcast, assigned robots repeat their claim every round (as CBBA does),
    so lost bids are eventually delivered. Returns (each robot's believed task, number of messages sent). With
    message loss, two robots can end up believing they own the same task (a conflict)."""
    benefit = -np.asarray(cost, float)
    n, m = benefit.shape
    eps = eps if eps is not None else 1e-3 * (np.ptp(benefit) + 1e-9) / n
    price = np.zeros((n, m))                       # robot i's view of the prices
    owner = -np.ones((n, m), int)                  # robot i's view of who owns each task
    mine = -np.ones(n, int)
    for t in range(rounds):
        for i in range(n):                         # 1. read bids; a higher (price, bidder) pair wins
            for _, (j, p, bidder) in channel.receive(t, i):
                if (p, bidder) > (price[i, j], owner[i, j]):
                    price[i, j], owner[i, j] = p, bidder
                    if mine[i] == j and bidder != i:
                        mine[i] = -1               # outbid: bid again
        for i in range(n):                         # 2. unassigned robots bid with their local prices
            if mine[i] >= 0:
                if rebroadcast:
                    channel.send(t, i, (mine[i], price[i, mine[i]], i), positions)
                continue
            v = benefit[i] - price[i]
            j = int(np.argmax(v))
            second = np.partition(v, -2)[-2] if m > 1 else v[j]
            price[i, j] += v[j] - second + eps
            owner[i, j], mine[i] = i, j
            channel.send(t, i, (j, price[i, j], i), positions)
    return mine, channel.sent
