"""
Random task generator that feeds scheduler.py.

It builds a random, always-acyclic dependency graph (DAG) and randomly mixes
the task kinds supported by scheduler.py:

    plain       runs on any single CPU
    pin         pinned to one specific CPU            (pin=3)
    gang        needs several specific CPUs together  (pin=[0, 1])
    need        needs any N CPUs together             (need=2)
    flex        takes as many idle CPUs as available (flex=True)

Examples
--------
    # 20 random tasks on 4 CPUs, reproducible with a seed
    python task_generator.py --tasks 20 --cpus 4 --seed 42

    # Larger run, more flexible tasks, show the chart in a window
    python task_generator.py --tasks 40 --cpus 8 --flex-prob 0.2 --show

    # Save the generated tasks, then replay exactly the same workload later
    python task_generator.py --tasks 15 --seed 7 --save-json tasks.json
    python task_generator.py --load-json tasks.json
"""
import argparse
import json
import random

from scheduler import Task, schedule, print_report, plot_gantt


def generate_tasks(num_tasks, num_cpus, seed=None,
                   min_dur=1, max_dur=8,
                   dep_prob=0.3, max_deps=3, window=10,
                   pin_prob=0.15, gang_prob=0.10,
                   need_prob=0.10, flex_prob=0.10):
    """
    Generate a list of random Task objects.

    num_tasks / num_cpus : size of the workload and of the simulated machine
    seed                 : random seed (same seed -> same workload)
    min_dur, max_dur     : task duration (or total work for flex) range
    dep_prob             : chance that each candidate predecessor becomes
                           a dependency
    max_deps             : maximum number of dependencies per task
    window               : a task may only depend on the `window` tasks
                           created just before it; this keeps the graph
                           wide enough to expose parallelism
    pin_prob, gang_prob, need_prob, flex_prob :
                           probability of each special task kind; whatever
                           is left over (1 - sum) becomes a plain task
    """
    if min_dur < 1 or max_dur < min_dur:
        raise ValueError("Need 1 <= min_dur <= max_dur")
    if not 0 <= pin_prob + gang_prob + need_prob + flex_prob <= 1:
        raise ValueError("The four task-kind probabilities must sum to <= 1")

    rng = random.Random(seed)       # Private RNG: does not touch global state
    tasks = []

    for tid in range(1, num_tasks + 1):
        duration = rng.randint(min_dur, max_dur)

        # --- Dependencies -------------------------------------------------
        # Only depend on EARLIER ids, which guarantees the graph is acyclic.
        candidates = range(max(1, tid - window), tid)
        deps = [c for c in candidates if rng.random() < dep_prob]
        if len(deps) > max_deps:
            deps = rng.sample(deps, max_deps)
        deps.sort()

        # --- Task kind ----------------------------------------------------
        # One uniform draw is mapped onto consecutive probability bands.
        r = rng.random()
        if r < pin_prob:
            kind = "pin"
        elif r < pin_prob + gang_prob:
            kind = "gang"
        elif r < pin_prob + gang_prob + need_prob:
            kind = "need"
        elif r < pin_prob + gang_prob + need_prob + flex_prob:
            kind = "flex"
        else:
            kind = "plain"

        # Multi-CPU kinds make no sense on a single-CPU machine.
        if kind in ("gang", "need") and num_cpus < 2:
            kind = "plain"

        # --- Kind-specific parameters -------------------------------------
        extra = {}
        if kind == "pin":
            extra["pin"] = rng.randrange(num_cpus)
        elif kind == "gang":
            size = rng.randint(2, num_cpus)
            extra["pin"] = sorted(rng.sample(range(num_cpus), size))
        elif kind == "need":
            extra["need"] = rng.randint(2, num_cpus)
        elif kind == "flex":
            extra["flex"] = True
            # Require at most half the machine to start; keep it small so
            # flex tasks are not blocked too often.
            extra["min_cpus"] = rng.randint(1, max(1, num_cpus // 2))
            if rng.random() < 0.5:          # Half of them also get a cap
                extra["max_cpus"] = rng.randint(extra["min_cpus"], num_cpus)

        tasks.append(Task(tid, duration, deps, **extra))

    return tasks


def describe(task):
    """Short human-readable description of a task's CPU requirement."""
    if task.flex:
        hi = task.max_cpus if task.max_cpus is not None else "all"
        return f"flex({task.min_cpus}..{hi})"
    if task.pin is not None:
        return f"pin={task.pin}" if len(task.pin) > 1 else f"pin={task.pin[0]}"
    if task.need > 1:
        return f"need={task.need}"
    return "-"


def print_tasks(tasks):
    """Print the generated workload (the scheduler's input)."""
    print(f"Generated {len(tasks)} tasks")
    print(f"{'Task':>4} {'Dur':>3}  {'Requirement':<14} Deps")
    for t in tasks:
        print(f"{t.id:>4} {t.duration:>3}  {describe(t):<14} {t.deps}")
    print()


def tasks_to_json(tasks, path):
    """Save only the INPUT fields so the same workload can be replayed."""
    rows = []
    for t in tasks:
        row = {"id": t.id, "duration": t.duration, "deps": t.deps}
        if t.flex:
            row.update(flex=True, min_cpus=t.min_cpus, max_cpus=t.max_cpus)
        elif t.pin is not None:
            row["pin"] = t.pin
        elif t.need > 1:
            row["need"] = t.need
        rows.append(row)
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Tasks saved to {path}")


def tasks_from_json(path):
    """Load tasks written by tasks_to_json (or written by hand)."""
    with open(path) as f:
        return [Task(**row) for row in json.load(f)]


def main():
    p = argparse.ArgumentParser(description="Generate random tasks and "
                                            "schedule them with scheduler.py")
    p.add_argument("--tasks", type=int, default=20, help="number of tasks")
    p.add_argument("--cpus", type=int, default=4, help="number of CPUs")
    p.add_argument("--seed", type=int, default=None, help="random seed")
    p.add_argument("--min-dur", type=int, default=1)
    p.add_argument("--max-dur", type=int, default=8)
    p.add_argument("--dep-prob", type=float, default=0.3)
    p.add_argument("--max-deps", type=int, default=3)
    p.add_argument("--window", type=int, default=10)
    p.add_argument("--pin-prob", type=float, default=0.15)
    p.add_argument("--gang-prob", type=float, default=0.10)
    p.add_argument("--need-prob", type=float, default=0.10)
    p.add_argument("--flex-prob", type=float, default=0.10)
    p.add_argument("--output", default="gantt_random.png",
                   help="Gantt chart image file")
    p.add_argument("--show", action="store_true",
                   help="also open the chart in a window")
    p.add_argument("--save-json", metavar="FILE",
                   help="save the generated tasks to a JSON file")
    p.add_argument("--load-json", metavar="FILE",
                   help="load tasks from JSON instead of generating them")
    a = p.parse_args()

    # 1. Get the workload: either replay a saved one or generate a new one.
    if a.load_json:
        tasks = tasks_from_json(a.load_json)
    else:
        tasks = generate_tasks(
            a.tasks, a.cpus, seed=a.seed,
            min_dur=a.min_dur, max_dur=a.max_dur,
            dep_prob=a.dep_prob, max_deps=a.max_deps, window=a.window,
            pin_prob=a.pin_prob, gang_prob=a.gang_prob,
            need_prob=a.need_prob, flex_prob=a.flex_prob)
    print_tasks(tasks)
    if a.save_json:
        tasks_to_json(tasks, a.save_json)

    # 2. Hand the tasks to the scheduler, then report and draw the chart.
    schedule(tasks, a.cpus)
    print_report(tasks, a.cpus)
    plot_gantt(tasks, a.cpus, a.output, show=a.show)


if __name__ == "__main__":
    main()
