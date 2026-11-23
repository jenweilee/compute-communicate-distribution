"""
Multi-NPU task scheduler simulation with a Gantt chart.

Overview
--------
Tasks have an id, a duration and a list of dependencies (ids of tasks that
must finish before this one may start). The scheduler simulates N identical
NPUs using an event-driven list-scheduling algorithm:

  1. A task becomes "ready" once all of its dependencies have finished.
  2. Whenever NPUs are idle, ready tasks are dispatched in priority order.
  3. Priority = critical-path length (the longest chain of work from the task
     to the end of the graph, including the task itself). Tasks on a longer
     chain run first so the critical path is not delayed. Ties are broken by
     the smaller task id.

NPU requirements per task
-------------------------
    pin=None      No constraint; the task runs on any single idle NPU.
    pin=2         The task may only run on NPU 2.
    pin=[0, 1]    The task needs NPU 0 AND NPU 1 at the same time
                  (gang scheduling). It starts only when every listed NPU is
                  idle, and all of them are released together at the end.
    need=N        The task needs any N NPUs at the same time. The scheduler
                  chooses which ones. Only valid when pin is None.
    flex=True     Flexible ("moldable") task: when it starts, it takes as many
                  idle NPUs as are available right now (bounded by min_npus
                  and max_npus). In this mode `duration` means TOTAL WORK, and
                  the run time on k NPUs is ceil(duration / k), i.e. an ideal
                  linear speed-up is assumed. Cannot be combined with
                  pin or need > 1.

Usage
-----
    pip install matplotlib
    python task_scheduler.py
"""
from dataclasses import dataclass, field
import heapq
import sys


@dataclass
class Task:
    # ----- Inputs provided by the user -----
    id: int                      # Unique task id
    duration: int                # Run time (or total work if flex=True)
    deps: list = field(default_factory=list)  # Ids of prerequisite tasks
    pin: object = None           # None / int / list[int]: required NPU(s)
    need: int = None             # Number of NPUs required (any NPUs)
                                 #   default: len(pin) if pinned, else 1
    flex: bool = False           # Take as many idle NPUs as possible
    min_npus: int = 1            # flex only: minimum NPUs needed to start
    max_npus: int = None         # flex only: upper bound (None = unlimited)

    # ----- Outputs filled in by schedule() -----
    start: int = None            # Start time
    end: int = None              # Finish time
    npus: list = None            # NPUs actually used (list of NPU indices)

    def __post_init__(self):
        """Validate and normalise the NPU-related fields."""
        # --- Flexible task: validate its own parameters and finish early ---
        if self.flex:
            # flex decides its NPU count at run time, so a fixed NPU
            # requirement would be contradictory.
            if self.pin is not None or self.need not in (None, 1):
                raise ValueError(f"Task {self.id}: flex cannot be combined "
                                 f"with pin / need")
            if self.min_npus < 1:
                raise ValueError(f"Task {self.id}: min_npus must be >= 1")
            if self.max_npus is not None and self.max_npus < self.min_npus:
                raise ValueError(f"Task {self.id}: max_npus < min_npus")
            # The scheduler checks `need` against the NPU count, so use the
            # minimum the task requires in order to start.
            self.need = self.min_npus
            return

        # --- Pinned task: normalise pin into a sorted list of unique NPUs ---
        if self.pin is not None:
            pins = [self.pin] if isinstance(self.pin, int) else list(self.pin)
            if not pins:
                raise ValueError(f"Task {self.id}: pin is empty")
            if len(set(pins)) != len(pins):
                raise ValueError(f"Task {self.id}: duplicate NPU in pin {pins}")
            self.pin = sorted(pins)
            # If need was also given it must agree with the pin list.
            if self.need is not None and self.need != len(pins):
                raise ValueError(
                    f"Task {self.id}: need={self.need} conflicts with pin={pins}")
            self.need = len(pins)
        # --- Unpinned task: need defaults to a single NPU ---
        else:
            if self.need is None:
                self.need = 1
            if self.need < 1:
                raise ValueError(f"Task {self.id}: need must be >= 1")


def critical_path_lengths(tasks):
    """
    Return {task_id: length of the longest path from that task to the end}.

    The length includes the task's own duration. It is used as the scheduling
    priority: a larger value means more work depends on this task finishing.
    """
    # Build the reverse graph: task -> tasks that depend on it.
    children = {t.id: [] for t in tasks.values()}
    for t in tasks.values():
        for d in t.deps:
            children[d].append(t.id)

    memo = {}  # Cache so each task is computed only once

    def cp(tid):
        if tid not in memo:
            # Own duration + the longest chain among the dependents
            # (0 if the task has no dependents).
            memo[tid] = tasks[tid].duration + max(
                (cp(c) for c in children[tid]), default=0
            )
        return memo[tid]

    sys.setrecursionlimit(10000)  # Allow deep dependency chains
    return {tid: cp(tid) for tid in tasks}


def check_graph(tasks):
    """
    Validate the dependency graph.

    Raises ValueError if a task depends on an unknown id, or if the graph
    contains a cycle (a cycle would make some tasks impossible to run).
    """
    # Every dependency must refer to an existing task.
    for t in tasks.values():
        for d in t.deps:
            if d not in tasks:
                raise ValueError(f"Task {t.id} depends on unknown task {d}")

    # Cycle detection with Kahn's algorithm (topological sort): repeatedly
    # remove tasks with no unmet dependencies. If some tasks can never be
    # removed, they are part of (or blocked by) a cycle.
    indeg = {tid: len(t.deps) for tid, t in tasks.items()}
    children = {tid: [] for tid in tasks}
    for t in tasks.values():
        for d in t.deps:
            children[d].append(t.id)
    q = [tid for tid, n in indeg.items() if n == 0]
    seen = 0
    while q:
        u = q.pop()
        seen += 1
        for v in children[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    if seen != len(tasks):
        raise ValueError("Dependency graph has a cycle")


def schedule(task_list, num_npus=4):
    """
    Simulate the schedule and fill in start / end / npus for every task.

    The simulation jumps from one "task finished" event to the next instead of
    stepping through every time unit.
    """
    tasks = {t.id: t for t in task_list}
    check_graph(tasks)

    # Validate NPU requirements against the machine size.
    for t in task_list:
        if t.pin is not None and not all(0 <= c < num_npus for c in t.pin):
            raise ValueError(f"Task {t.id} pinned to invalid NPU {t.pin}")
        if t.need > num_npus:
            raise ValueError(
                f"Task {t.id} needs {t.need} NPUs but only {num_npus} exist")

    prio = critical_path_lengths(tasks)

    # remaining_deps[tid] = dependencies that have not finished yet.
    remaining_deps = {tid: set(t.deps) for tid, t in tasks.items()}
    # children[tid] = tasks that depend on tid (used to unlock them later).
    children = {tid: [] for tid in tasks}
    for t in tasks.values():
        for d in t.deps:
            children[d].append(t.id)

    # Ready queue: a min-heap of (-priority, id). Negating the priority makes
    # the task with the longest critical path pop first; ties go to lower id.
    ready = [(-prio[tid], tid) for tid, deps in remaining_deps.items() if not deps]
    heapq.heapify(ready)

    free_npus = list(range(num_npus))   # Currently idle NPUs
    running = []                        # Min-heap of (end_time, task_id, npus)
    time, done = 0, 0                   # Simulation clock, finished-task count

    while done < len(tasks):
        # ---------------------------------------------------------------
        # Step 1: dispatch ready tasks to idle NPUs, highest priority first.
        # A task whose NPU requirement cannot be met right now is "deferred"
        # (skipped for this round) so it does not block lower-priority tasks
        # that could run on the remaining NPUs.
        # ---------------------------------------------------------------
        deferred = []
        while ready and free_npus:
            item = heapq.heappop(ready)
            t = tasks[item[1]]
            runtime = t.duration        # Overridden below for flex tasks

            if t.pin is None:
                # Task may use any NPUs: it only needs enough idle ones.
                if len(free_npus) < t.need:
                    deferred.append(item)   # Not enough idle NPUs; wait
                    continue

                # Prefer NPUs that no other waiting pinned task is asking
                # for, so unconstrained tasks do not steal them.
                wanted = {c for _, i in ready + deferred
                          if tasks[i].pin for c in tasks[i].pin}
                choices = ([c for c in free_npus if c not in wanted] +
                           [c for c in free_npus if c in wanted])

                # How many NPUs to take: exactly `need`, except flex tasks
                # which take every idle NPU (up to max_npus).
                k = t.need
                if t.flex:
                    k = len(free_npus)
                    if t.max_npus is not None:
                        k = min(k, t.max_npus)
                npus = sorted(choices[:k])

                # Flex tasks run faster with more NPUs (ideal linear
                # speed-up): run time = ceil(total work / number of NPUs).
                if t.flex:
                    runtime = -(-t.duration // len(npus))
            elif all(c in free_npus for c in t.pin):
                # Pinned task: every required NPU must be idle at once.
                npus = list(t.pin)
            else:
                deferred.append(item)       # A required NPU is busy; wait
                continue

            # Commit the assignment: occupy the NPUs and record the times.
            for c in npus:
                free_npus.remove(c)
            t.start, t.end, t.npus = time, time + runtime, npus
            heapq.heappush(running, (t.end, item[1], npus))

        # Put the skipped tasks back so they are retried at the next event.
        for item in deferred:
            heapq.heappush(ready, item)

        # ---------------------------------------------------------------
        # Step 2: advance the clock to the next completion event.
        # (running is never empty here: if nothing runs, all NPUs are idle,
        # so every ready task can be dispatched in Step 1.)
        # ---------------------------------------------------------------
        time = running[0][0]

        # Finish every task that ends at this instant: free its NPUs and
        # unlock dependents whose prerequisites are now all complete.
        while running and running[0][0] == time:
            _, tid, npus = heapq.heappop(running)
            free_npus.extend(npus)
            free_npus.sort()
            done += 1
            for c in children[tid]:
                remaining_deps[c].discard(tid)
                if not remaining_deps[c]:
                    heapq.heappush(ready, (-prio[c], c))

    return task_list


def print_report(task_list, num_npus):
    """Print a per-task schedule table plus makespan and NPU utilization."""
    # Makespan = time at which the last task finishes.
    makespan = max(t.end for t in task_list)
    # Total busy NPU-time: tasks that hold several NPUs count once per NPU.
    busy = sum((t.end - t.start) * len(t.npus) for t in task_list)

    # Format a list of NPUs as "0,1,2" (or "-" for None).
    fmt = lambda xs: "-" if xs is None else ",".join(map(str, xs))

    print(f"{'Task':>4} {'NPU':>7} {'Pin':>7} {'Work':>4} {'Start':>5} "
          f"{'End':>4} {'Run':>3}  Deps")
    for t in sorted(task_list, key=lambda x: (x.start, x.npus)):
        pin = "flex" if t.flex else fmt(t.pin)
        # Work = nominal duration/work; Run = actual elapsed time.
        print(f"{t.id:>4} {fmt(t.npus):>7} {pin:>7} {t.duration:>4} "
              f"{t.start:>5} {t.end:>4} {t.end - t.start:>3}  {t.deps}")

    print(f"\nMakespan: {makespan}")
    # Utilization = busy NPU-time / total available NPU-time.
    print(f"Utilization: {busy / (makespan * num_npus):.1%}")


def plot_gantt(task_list, num_npus, filename="gantt.png", show=False):
    """
    Draw a Gantt chart: one row per NPU, one bar per task.

    Hatch patterns: pinned = "//", need > 1 = "..", flex = "xx".
    A task holding several NPUs is drawn once on each of its NPU rows.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")   # Headless backend: render to file only
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")    # Colour is chosen by task id
    fig, ax = plt.subplots(figsize=(11, 1.2 * num_npus + 1.5))

    for t in task_list:
        for npu in t.npus:
            # Bar width is the actual elapsed time (matters for flex tasks).
            ax.barh(npu, t.end - t.start, left=t.start, height=0.6,
                    color=cmap(t.id % 20), edgecolor="black",
                    hatch=("xx" if t.flex else "//" if t.pin is not None
                           else ".." if t.need > 1 else None))

            # Bar label: task id plus a tag describing how it was scheduled.
            label = f"T{t.id}"
            if t.flex:
                label += f"\n(flex x{len(t.npus)})"
            elif len(t.npus) > 1:
                label += "\n(gang)"
            elif t.pin is not None:
                label += "\n(pin)"
            ax.text((t.start + t.end) / 2, npu, label,
                    ha="center", va="center", fontsize=9, fontweight="bold")

    makespan = max(t.end for t in task_list)
    ax.set_yticks(range(num_npus))
    ax.set_yticklabels([f"NPU {i}" for i in range(num_npus)])
    ax.invert_yaxis()                   # NPU 0 at the top
    ax.set_xlim(0, makespan)
    # Keep the number of x ticks readable for long schedules.
    ax.set_xticks(range(0, makespan + 1, max(1, makespan // 20)))
    ax.set_xlabel("Time")
    ax.set_title(f"Gantt Chart - {num_npus} NPUs (makespan = {makespan})")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Gantt chart saved to {filename}")
    if show:
        plt.show()


if __name__ == "__main__":
    NUM_NPUS = 4

    # Task(id, duration, deps, pin=None / one NPU / list of NPUs,
    #      need=any N NPUs, flex=True [, min_npus, max_npus])
    # Gantt chart hatching: pin "//", need>1 "..", flex "xx".
    tasks = [
        Task(1, 3),                                  # Plain task, any NPU
        Task(2, 2),
        Task(3, 4),
        Task(4, 4, [1], flex=True, max_npus=2),      # Flex: up to 2 NPUs
        Task(5, 3, [1, 2], pin=[1, 2]),              # Needs NPU 1 and 2 together
        Task(6, 5, [3], pin=3),                      # Only runs on NPU 3
        Task(7, 2, [4, 5], need=2),                  # Any 2 NPUs together
        Task(8, 4, [5, 6], pin=[2, 3]),              # Needs NPU 2 and 3 together
        Task(9, 3, [2], pin=0),                      # Only runs on NPU 0
        Task(10, 6, [7, 8, 9], flex=True),           # Flex: all idle NPUs
        Task(11, 4, [6], need=2),                    # Any 2 NPUs together
        Task(12, 3, [10, 11], pin=[0, 1, 2, 3]),     # All 4 NPUs together
    ]

    schedule(tasks, NUM_NPUS)
    print_report(tasks, NUM_NPUS)
    plot_gantt(tasks, NUM_NPUS, "gantt.png")
