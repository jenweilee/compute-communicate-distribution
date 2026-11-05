"""
4-CPU task scheduler simulation (list scheduling) + Gantt chart
- 每個 task 有 id、duration、dependencies
- 事件驅動模擬:CPU 閒置時,從 ready queue 挑優先權最高的 task
- 優先權 = critical path 長度(越長越先做),同分則 id 小者先
- Task 可用 pin=N 指定只能在 CPU N 上執行(不指定則任一 CPU 皆可)
"""
from dataclasses import dataclass, field
import heapq
import sys


@dataclass
class Task:
    id: int
    duration: int
    deps: list = field(default_factory=list)
    pin: int = None          # 指定 CPU(None = 不限)
    # 以下由排程器填入
    start: int = None
    end: int = None
    cpu: int = None


def critical_path_lengths(tasks):
    """每個 task 到終點的最長路徑長度(含自己),用來當優先權。"""
    children = {t.id: [] for t in tasks.values()}
    for t in tasks.values():
        for d in t.deps:
            children[d].append(t.id)

    memo = {}

    def cp(tid):
        if tid not in memo:
            memo[tid] = tasks[tid].duration + max(
                (cp(c) for c in children[tid]), default=0
            )
        return memo[tid]

    sys.setrecursionlimit(10000)
    return {tid: cp(tid) for tid in tasks}


def check_graph(tasks):
    """檢查 dependency 是否存在,且沒有循環。"""
    for t in tasks.values():
        for d in t.deps:
            if d not in tasks:
                raise ValueError(f"Task {t.id} depends on unknown task {d}")
    # Kahn's algorithm 偵測 cycle
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
    tasks = {t.id: t for t in task_list}
    check_graph(tasks)
    for t in task_list:
        if t.pin is not None and not (0 <= t.pin < num_cpus):
            raise ValueError(f"Task {t.id} pinned to invalid CPU {t.pin}")
    prio = critical_path_lengths(tasks)

    remaining_deps = {tid: set(t.deps) for tid, t in tasks.items()}
    children = {tid: [] for tid in tasks}
    for t in tasks.values():
        for d in t.deps:
            children[d].append(t.id)

    # ready queue: (-priority, id)
    ready = [(-prio[tid], tid) for tid, deps in remaining_deps.items() if not deps]
    heapq.heapify(ready)

    free_cpus = list(range(num_cpus))   # 閒置的 CPU
    running = []                         # (end_time, task_id, cpu)
    time, done = 0, 0

    while done < len(tasks):
        # 依優先權逐一分配;指定 CPU 正忙的 task 先跳過,等該 CPU 空出來
        deferred = []
        while ready and free_cpus:
            item = heapq.heappop(ready)
            t = tasks[item[1]]
            if t.pin is None:
                # 不限 CPU 的 task,盡量避開別的 ready task 指定要用的 CPU
                wanted = {tasks[i].pin for _, i in ready + deferred
                          if tasks[i].pin is not None}
                choices = [c for c in free_cpus if c not in wanted] or free_cpus
                cpu = choices[0]
            elif t.pin in free_cpus:
                cpu = t.pin
            else:
                deferred.append(item)
                continue
            free_cpus.remove(cpu)
            t.start, t.end, t.cpu = time, time + t.duration, cpu
            heapq.heappush(running, (t.end, item[1], cpu))
        for item in deferred:
            heapq.heappush(ready, item)

        # 時間推進到下一個 task 完成
        time = running[0][0]
        while running and running[0][0] == time:
            _, tid, cpu = heapq.heappop(running)
            free_cpus.append(cpu)
            free_cpus.sort()
            done += 1
            for c in children[tid]:
                remaining_deps[c].discard(tid)
                if not remaining_deps[c]:
                    heapq.heappush(ready, (-prio[c], c))

    return task_list


def print_report(task_list, num_cpus):
    makespan = max(t.end for t in task_list)
    busy = sum(t.duration for t in task_list)
    print(f"{'Task':>4} {'CPU':>3} {'Pin':>4} {'Start':>5} {'End':>4}  Deps")
    for t in sorted(task_list, key=lambda x: (x.start, x.cpu)):
        pin = "-" if t.pin is None else t.pin
        print(f"{t.id:>4} {t.cpu:>3} {pin:>4} {t.start:>5} {t.end:>4}  {t.deps}")
    print(f"\nMakespan: {makespan}")
    print(f"Utilization: {busy / (makespan * num_cpus):.1%}")


def plot_gantt(task_list, num_cpus, filename="gantt.png", show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(11, 1.2 * num_cpus + 1.5))

    for t in task_list:
        ax.barh(t.cpu, t.duration, left=t.start, height=0.6,
                color=cmap(t.id % 20), edgecolor="black",
                hatch="//" if t.pin is not None else None)
        ax.text(t.start + t.duration / 2, t.cpu,
                f"T{t.id}" + ("\n(pin)" if t.pin is not None else ""),
                ha="center", va="center", fontsize=9, fontweight="bold")

    makespan = max(t.end for t in task_list)
    ax.set_yticks(range(num_cpus))
    ax.set_yticklabels([f"CPU {i}" for i in range(num_cpus)])
    ax.invert_yaxis()
    ax.set_xlim(0, makespan)
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

    # 範例:Task(id, duration, deps, pin=指定CPU),pin 的 task 在圖上以斜線標示
    tasks = [
        Task(1, 3),
        Task(2, 2),
        Task(3, 4),
        Task(4, 2, [1]),
        Task(5, 3, [1, 2]),
        Task(6, 5, [3], pin=3),
        Task(7, 2, [4, 5]),
        Task(8, 4, [5, 6], pin=3),
        Task(9, 3, [2], pin=0),
        Task(10, 2, [7, 8, 9]),
        Task(11, 4, [6]),
        Task(12, 3, [10, 11]),
    ]

    schedule(tasks, NUM_CPUS)
    print_report(tasks, NUM_CPUS)
    plot_gantt(tasks, NUM_CPUS, "gantt.png")
