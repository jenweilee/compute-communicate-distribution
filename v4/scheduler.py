"""
Multi-CPU task scheduler simulation with a Gantt chart.

Overview
--------
Tasks have an id, a duration and a list of dependencies (ids of tasks that
must finish before this one may start). The scheduler simulates N identical
CPUs using an event-driven list-scheduling algorithm:

  1. A task becomes "ready" once all of its dependencies have finished.
  2. Whenever CPUs are idle, ready tasks are dispatched in priority order.
  3. Priority = critical-path length (the longest chain of work from the task
     to the end of the graph, including the task itself). Tasks on a longer
     chain run first so the critical path is not delayed. Ties are broken by
     the smaller task id.

CPU requirements per task
-------------------------
    pin=None      No constraint; the task runs on any single idle CPU.
    pin=2         The task may only run on CPU 2.
    pin=[0, 1]    The task needs CPU 0 AND CPU 1 at the same time
                  (gang scheduling). It starts only when every listed CPU is
                  idle, and all of them are released together at the end.
    need=N        The task needs any N CPUs at the same time. The scheduler
                  chooses which ones. Only valid when pin is None.
    flex=True     Flexible ("moldable") task: when it starts, it takes as many
                  idle CPUs as are available right now (bounded by min_cpus
                  and max_cpus). In this mode `duration` means TOTAL WORK, and
                  the run time on k CPUs is ceil(duration / k), i.e. an ideal
                  linear speed-up is assumed. Cannot be combined with
                  pin or need > 1.

Usage
-----
    pip install matplotlib
    python scheduler.py
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
    pin: object = None           # None / int / list[int]: required CPU(s)
    need: int = None             # Number of CPUs required (any CPUs)
                                 #   default: len(pin) if pinned, else 1
    flex: bool = False           # Take as many idle CPUs as possible
    min_cpus: int = 1            # flex only: minimum CPUs needed to start
    max_cpus: int = None         # flex only: upper bound (None = unlimited)

    # ----- Outputs filled in by schedule() -----
    start: int = None            # Start time
    end: int = None              # Finish time
    cpus: list = None            # CPUs actually used (list of CPU indices)

    def __post_init__(self):
        """Validate and normalise the CPU-related fields."""
        # --- Flexible task: validate its own parameters and finish early ---
        if self.flex:
            # flex decides its CPU count at run time, so a fixed CPU
            # requirement would be contradictory.
            if self.pin is not None or self.need not in (None, 1):
                raise ValueError(f"Task {self.id}: flex cannot be combined "
                                 f"with pin / need")
            if self.min_cpus < 1:
                raise ValueError(f"Task {self.id}: min_cpus must be >= 1")
            if self.max_cpus is not None and self.max_cpus < self.min_cpus:
                raise ValueError(f"Task {self.id}: max_cpus < min_cpus")
            # The scheduler checks `need` against the CPU count, so use the
            # minimum the task requires in order to start.
            self.need = self.min_cpus
            return

        # --- Pinned task: normalise pin into a sorted list of unique CPUs ---
        if self.pin is not None:
            pins = [self.pin] if isinstance(self.pin, int) else list(self.pin)
            if not pins:
                raise ValueError(f"Task {self.id}: pin is empty")
            if len(set(pins)) != len(pins):
                raise ValueError(f"Task {self.id}: duplicate CPU in pin {pins}")
            self.pin = sorted(pins)
            # If need was also given it must agree with the pin list.
            if self.need is not None and self.need != len(pins):
                raise ValueError(
                    f"Task {self.id}: need={self.need} conflicts with pin={pins}")
            self.need = len(pins)
        # --- Unpinned task: need defaults to a single CPU ---
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


def schedule(task_list, num_cpus=4):
    """
    Simulate the schedule and fill in start / end / cpus for every task.

    The simulation jumps from one "task finished" event to the next instead of
    stepping through every time unit.
    """
    tasks = {t.id: t for t in task_list}
    check_graph(tasks)

    # Validate CPU requirements against the machine size.
    for t in task_list:
        if t.pin is not None and not all(0 <= c < num_cpus for c in t.pin):
            raise ValueError(f"Task {t.id} pinned to invalid CPU {t.pin}")
        if t.need > num_cpus:
            raise ValueError(
                f"Task {t.id} needs {t.need} CPUs but only {num_cpus} exist")

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

    free_cpus = list(range(num_cpus))   # Currently idle CPUs
    running = []                        # Min-heap of (end_time, task_id, cpus)
    time, done = 0, 0                   # Simulation clock, finished-task count

    while done < len(tasks):
        # ---------------------------------------------------------------
        # Step 1: dispatch ready tasks to idle CPUs, highest priority first.
        # A task whose CPU requirement cannot be met right now is "deferred"
        # (skipped for this round) so it does not block lower-priority tasks
        # that could run on the remaining CPUs.
        # ---------------------------------------------------------------
        deferred = []
        while ready and free_cpus:
            item = heapq.heappop(ready)
            t = tasks[item[1]]
            runtime = t.duration        # Overridden below for flex tasks

            if t.pin is None:
                # Task may use any CPUs: it only needs enough idle ones.
                if len(free_cpus) < t.need:
                    deferred.append(item)   # Not enough idle CPUs; wait
                    continue

                # Prefer CPUs that no other waiting pinned task is asking
                # for, so unconstrained tasks do not steal them.
                wanted = {c for _, i in ready + deferred
                          if tasks[i].pin for c in tasks[i].pin}
                choices = ([c for c in free_cpus if c not in wanted] +
                           [c for c in free_cpus if c in wanted])

                # How many CPUs to take: exactly `need`, except flex tasks
                # which take every idle CPU (up to max_cpus).
                k = t.need
                if t.flex:
                    k = len(free_cpus)
                    if t.max_cpus is not None:
                        k = min(k, t.max_cpus)
                cpus = sorted(choices[:k])

                # Flex tasks run faster with more CPUs (ideal linear
                # speed-up): run time = ceil(total work / number of CPUs).
                if t.flex:
                    runtime = -(-t.duration // len(cpus))
            elif all(c in free_cpus for c in t.pin):
                # Pinned task: every required CPU must be idle at once.
                cpus = list(t.pin)
            else:
                deferred.append(item)       # A required CPU is busy; wait
                continue

            # Commit the assignment: occupy the CPUs and record the times.
            for c in cpus:
                free_cpus.remove(c)
            t.start, t.end, t.cpus = time, time + runtime, cpus
            heapq.heappush(running, (t.end, item[1], cpus))

        # Put the skipped tasks back so they are retried at the next event.
        for item in deferred:
            heapq.heappush(ready, item)

        # ---------------------------------------------------------------
        # Step 2: advance the clock to the next completion event.
        # (running is never empty here: if nothing runs, all CPUs are idle,
        # so every ready task can be dispatched in Step 1.)
        # ---------------------------------------------------------------
        time = running[0][0]

        # Finish every task that ends at this instant: free its CPUs and
        # unlock dependents whose prerequisites are now all complete.
        while running and running[0][0] == time:
            _, tid, cpus = heapq.heappop(running)
            free_cpus.extend(cpus)
            free_cpus.sort()
            done += 1
            for c in children[tid]:
                remaining_deps[c].discard(tid)
                if not remaining_deps[c]:
                    heapq.heappush(ready, (-prio[c], c))

    return task_list


def print_report(task_list, num_cpus):
    """Print a per-task schedule table plus makespan and CPU utilization."""
    # Makespan = time at which the last task finishes.
    makespan = max(t.end for t in task_list)
    # Total busy CPU-time: tasks that hold several CPUs count once per CPU.
    busy = sum((t.end - t.start) * len(t.cpus) for t in task_list)

    # Format a list of CPUs as "0,1,2" (or "-" for None).
    fmt = lambda xs: "-" if xs is None else ",".join(map(str, xs))

    print(f"{'Task':>4} {'CPU':>7} {'Pin':>7} {'Work':>4} {'Start':>5} "
          f"{'End':>4} {'Run':>3}  Deps")
    for t in sorted(task_list, key=lambda x: (x.start, x.cpus)):
        pin = "flex" if t.flex else fmt(t.pin)
        # Work = nominal duration/work; Run = actual elapsed time.
        print(f"{t.id:>4} {fmt(t.cpus):>7} {pin:>7} {t.duration:>4} "
              f"{t.start:>5} {t.end:>4} {t.end - t.start:>3}  {t.deps}")

    print(f"\nMakespan: {makespan}")
    # Utilization = busy CPU-time / total available CPU-time.
    print(f"Utilization: {busy / (makespan * num_cpus):.1%}")


def plot_gantt(task_list, num_cpus, filename="gantt.png", show=False):
    """
    Draw a Gantt chart: one row per CPU, one bar per task.

    Hatch patterns: pinned = "//", need > 1 = "..", flex = "xx".
    A task holding several CPUs is drawn once on each of its CPU rows.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")   # Headless backend: render to file only
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")    # Colour is chosen by task id
    fig, ax = plt.subplots(figsize=(11, 1.2 * num_cpus + 1.5))

    for t in task_list:
        for cpu in t.cpus:
            # Bar width is the actual elapsed time (matters for flex tasks).
            ax.barh(cpu, t.end - t.start, left=t.start, height=0.6,
                    color=cmap(t.id % 20), edgecolor="black",
                    hatch=("xx" if t.flex else "//" if t.pin is not None
                           else ".." if t.need > 1 else None))

            # Bar label: task id plus a tag describing how it was scheduled.
            label = f"T{t.id}"
            if t.flex:
                label += f"\n(flex x{len(t.cpus)})"
            elif len(t.cpus) > 1:
                label += "\n(gang)"
            elif t.pin is not None:
                label += "\n(pin)"
            ax.text((t.start + t.end) / 2, cpu, label,
                    ha="center", va="center", fontsize=9, fontweight="bold")

    makespan = max(t.end for t in task_list)
    ax.set_yticks(range(num_cpus))
    ax.set_yticklabels([f"CPU {i}" for i in range(num_cpus)])
    ax.invert_yaxis()                   # CPU 0 at the top
    ax.set_xlim(0, makespan)
    # Keep the number of x ticks readable for long schedules.
    ax.set_xticks(range(0, makespan + 1, max(1, makespan // 20)))
    ax.set_xlabel("Time")
    ax.set_title(f"Gantt Chart - {num_cpus} CPUs (makespan = {makespan})")
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"Gantt chart saved to {filename}")
    if show:
        plt.show()


if __name__ == "__main__":
    NUM_CPUS = 4

    # Task(id, duration, deps, pin=None / one CPU / list of CPUs,
    #      need=any N CPUs, flex=True [, min_cpus, max_cpus])
    # Gantt chart hatching: pin "//", need>1 "..", flex "xx".
    tasks = [
        Task(1, 3),                                  # Plain task, any CPU
        Task(2, 2),
        Task(3, 4),
        Task(4, 4, [1], flex=True, max_cpus=2),      # Flex: up to 2 CPUs
        Task(5, 3, [1, 2], pin=[1, 2]),              # Needs CPU 1 and 2 together
        Task(6, 5, [3], pin=3),                      # Only runs on CPU 3
        Task(7, 2, [4, 5], need=2),                  # Any 2 CPUs together
        Task(8, 4, [5, 6], pin=[2, 3]),              # Needs CPU 2 and 3 together
        Task(9, 3, [2], pin=0),                      # Only runs on CPU 0
        Task(10, 6, [7, 8, 9], flex=True),           # Flex: all idle CPUs
        Task(11, 4, [6], need=2),                    # Any 2 CPUs together
        Task(12, 3, [10, 11], pin=[0, 1, 2, 3]),     # All 4 CPUs together
    ]

    schedule(tasks, NUM_CPUS)
    print_report(tasks, NUM_CPUS)
    plot_gantt(tasks, NUM_CPUS, "gantt.png")
