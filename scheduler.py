"""
4-CPU task scheduler simulation (list scheduling) + Gantt chart
- 每個 task 有 id、duration、dependencies
- 事件驅動模擬:CPU 閒置時,從 ready queue 挑優先權最高的 task
- 優先權 = critical path 長度(越長越先做),同分則 id 小者先
- pin 可指定 CPU:
    pin=None      不限,任一 CPU 皆可(預設佔用 1 顆)
    need=N        不指定哪幾顆,但需要「同時」佔用任意 N 顆 CPU(pin=None 時有效)
    pin=2         只能在 CPU 2 上執行
    pin=[0, 1]    必須「同時」佔用 CPU 0 和 1 一起執行(gang scheduling),
                  兩顆都閒置時才會開始,結束時一起釋放
"""
from dataclasses import dataclass, field
import heapq
import sys


@dataclass
class Task:
    id: int
    duration: int
    deps: list = field(default_factory=list)
    pin: object = None       # None / int / list[int]:指定 CPU
    need: int = None         # 需要幾顆 CPU(未指定時:有 pin 則為 len(pin),否則 1)
    # 以下由排程器填入
    start: int = None
    end: int = None
    cpus: list = None        # 實際使用的 CPU(list)

    def __post_init__(self):
        if self.pin is not None:
            pins = [self.pin] if isinstance(self.pin, int) else list(self.pin)
            if not pins:
                raise ValueError(f"Task {self.id}: pin is empty")
            if len(set(pins)) != len(pins):
                raise ValueError(f"Task {self.id}: duplicate CPU in pin {pins}")
            self.pin = sorted(pins)
            if self.need is not None and self.need != len(pins):
                raise ValueError(
                    f"Task {self.id}: need={self.need} conflicts with pin={pins}")
            self.need = len(pins)
        else:
            if self.need is None:
                self.need = 1
            if self.need < 1:
                raise ValueError(f"Task {self.id}: need must be >= 1")


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
        if t.pin is not None and not all(0 <= c < num_cpus for c in t.pin):
            raise ValueError(f"Task {t.id} pinned to invalid CPU {t.pin}")
        if t.need > num_cpus:
            raise ValueError(
                f"Task {t.id} needs {t.need} CPUs but only {num_cpus} exist")
    prio = critical_path_lengths(tasks)

    remaining_deps = {tid: set(t.deps) for tid, t in tasks.items()}
    children = {tid: [] for tid in tasks}
    for t in tasks.values():
        for d in t.deps:
            children[d].append(t.id)

    ready = [(-prio[tid], tid) for tid, deps in remaining_deps.items() if not deps]
    heapq.heapify(ready)

    free_cpus = list(range(num_cpus))
    running = []                         # (end_time, task_id, [cpus])
    time, done = 0, 0

    while done < len(tasks):
        # 依優先權逐一分配;所需 CPU 沒有全部空出來的 task 先跳過
        deferred = []
        while ready and free_cpus:
            item = heapq.heappop(ready)
            t = tasks[item[1]]
            if t.pin is None:
                if len(free_cpus) < t.need:       # 閒置 CPU 不夠,先等
                    deferred.append(item)
                    continue
                # 不限 CPU 的 task:優先挑「別的 ready task 沒指定要用」的 CPU
                wanted = {c for _, i in ready + deferred
                          if tasks[i].pin for c in tasks[i].pin}
                choices = ([c for c in free_cpus if c not in wanted] +
                           [c for c in free_cpus if c in wanted])
                cpus = sorted(choices[:t.need])
            elif all(c in free_cpus for c in t.pin):
                cpus = list(t.pin)
            else:
                deferred.append(item)
                continue
            for c in cpus:
                free_cpus.remove(c)
            t.start, t.end, t.cpus = time, time + t.duration, cpus
            heapq.heappush(running, (t.end, item[1], cpus))
        for item in deferred:
            heapq.heappush(ready, item)

        # 時間推進到下一個 task 完成
        time = running[0][0]
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
    makespan = max(t.end for t in task_list)
    busy = sum(t.duration * len(t.cpus) for t in task_list)   # CPU-time
    fmt = lambda xs: "-" if xs is None else ",".join(map(str, xs))
    print(f"{'Task':>4} {'CPU':>6} {'Pin':>6} {'Start':>5} {'End':>4}  Deps")
    for t in sorted(task_list, key=lambda x: (x.start, x.cpus)):
        print(f"{t.id:>4} {fmt(t.cpus):>6} {fmt(t.pin):>6} "
              f"{t.start:>5} {t.end:>4}  {t.deps}")
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
        for cpu in t.cpus:           # 多 CPU 的 task 在每一列各畫一格
            ax.barh(cpu, t.duration, left=t.start, height=0.6,
                    color=cmap(t.id % 20), edgecolor="black",
                    hatch=("//" if t.pin is not None
                           else ".." if t.need > 1 else None))
            label = f"T{t.id}"
            if len(t.cpus) > 1:
                label += "\n(gang)"
            elif t.pin is not None:
                label += "\n(pin)"
            ax.text(t.start + t.duration / 2, cpu, label,
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

    # Task(id, duration, deps, pin=None / 單一 CPU / CPU 清單, need=任意 N 顆)
    # 有 pin 的 task 以斜線填滿、need>1 的以點點填滿
    tasks = [
        Task(1, 3),
        Task(2, 2),
        Task(3, 4),
        Task(4, 2, [1]),
        Task(5, 3, [1, 2], pin=[1, 2]),      # 同時佔用 CPU 1、2
        Task(6, 5, [3], pin=3),              # 只能用 CPU 3
        Task(7, 2, [4, 5], need=2),          # 任意 2 顆 CPU 一起跑
        Task(8, 4, [5, 6], pin=[2, 3]),      # 同時佔用 CPU 2、3
        Task(9, 3, [2], pin=0),
        Task(10, 2, [7, 8, 9]),
        Task(11, 4, [6], need=2),
        Task(12, 3, [10, 11], pin=[0, 1, 2, 3]),  # 全部 4 顆一起
    ]

    schedule(tasks, NUM_CPUS)
    print_report(tasks, NUM_CPUS)
    plot_gantt(tasks, NUM_CPUS, "gantt.png")
