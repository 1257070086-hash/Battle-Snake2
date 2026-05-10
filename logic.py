"""
BattleSnake AI — logic.py v10.0
================================
架构：Make/Unmake + Paranoid Minimax + 迭代加深
核心评估：Voronoi 领地差 + 真实可达空间 + BFS 食物距离

修复历史 bug：
- undo 死蛇：用 pre_move_snapshot 而非死亡瞬间快照
- just_ate：(turn > 0) and (health == 100)，排除开局误判
- Voronoi 中立格：NEUTRAL 锁死，永不扩展
- 对手尾可踩：owner.just_ate 判断
- 等长对头：两败俱伤
- 对手 length 字段不缩水
"""

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────
# 常量
# ─────────────────────────────────────────────────────────────

DIRS = {"up": (0,1), "down": (0,-1), "left": (-1,0), "right": (1,0)}
DIR_VECS = list(DIRS.values())
NEUTRAL = "__NEUTRAL__"     # Voronoi 中立格标记


# ─────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────

@dataclass
class Snake:
    id:       str
    body:     deque      # (x,y) tuples, body[0] = head
    health:   int
    length:   int
    alive:    bool = True
    just_ate: bool = False  # 本回合刚吃食（尾巴不移走）


@dataclass
class GameState:
    w:          int
    h:          int
    turn:       int
    snakes:     list          # list[Snake]
    food:       set           # set[(x,y)]
    me_id:      str
    occupied:   dict = field(default_factory=dict)   # (x,y) -> snake_id
    undo_stack: list = field(default_factory=list)

    def me(self) -> Optional[Snake]:
        for s in self.snakes:
            if s.id == self.me_id and s.alive:
                return s
        return None

    def alive_snakes(self):
        return [s for s in self.snakes if s.alive]

    def get_snake(self, sid) -> Optional[Snake]:
        for s in self.snakes:
            if s.id == sid:
                return s
        return None


def state_from_request(data) -> GameState:
    board   = data["board"]
    turn    = data["turn"]
    snakes  = []
    occupied = {}
    for s in board["snakes"]:
        body = deque((p["x"], p["y"]) for p in s["body"])
        # turn=0 时所有蛇 health=100，不算刚吃食
        just_ate = (turn > 0) and (s["health"] == 100)
        snakes.append(Snake(
            id=s["id"], body=body,
            health=s["health"], length=s["length"],
            just_ate=just_ate,
        ))
        for seg in body:
            occupied[seg] = s["id"]
    return GameState(
        w=board["width"], h=board["height"], turn=turn,
        snakes=snakes,
        food=set((f["x"], f["y"]) for f in board["food"]),
        me_id=data["you"]["id"],
        occupied=occupied,
    )


def in_bounds(p, w, h):
    return 0 <= p[0] < w and 0 <= p[1] < h


# ─────────────────────────────────────────────────────────────
# Make / Unmake（增量维护 occupied）
# ─────────────────────────────────────────────────────────────

def apply_moves(st: GameState, moves: dict):
    """
    moves: {snake_id: dir_name}
    ① Step 0：移动前快照（供 undo 回滚死蛇）
    ② Step 1：推进头/尾，增量维护 occupied
    ③ Step 2：死亡判定
    ④ Step 3：死亡处理
    """
    diff = {
        "turn":             st.turn,
        "pre_move_snap":    {},   # sid -> 移动前完整快照（含死蛇）
        "per_snake":        {},   # sid -> {added_head, removed_tail, ate_food, prev_owner}
        "killed":           [],   # [sid]
    }

    # ── Step 0: 移动前快照 ──────────────────────────────────────
    for s in st.alive_snakes():
        if s.id not in moves:
            continue
        diff["pre_move_snap"][s.id] = {
            "body":     list(s.body),
            "health":   s.health,
            "length":   s.length,
            "just_ate": s.just_ate,
        }

    # ── Step 1: 推进 ────────────────────────────────────────────
    new_heads = {}
    for s in st.alive_snakes():
        if s.id not in moves:
            continue
        d   = DIRS[moves[s.id]]
        old_head = s.body[0]
        new_head = (old_head[0] + d[0], old_head[1] + d[1])
        new_heads[s.id] = new_head

        snap = {
            "added_head":   new_head,
            "removed_tail": None,
            "ate_food":     False,
            "prev_owner":   st.occupied.get(new_head),  # 被覆盖前的 owner
        }

        s.body.appendleft(new_head)
        s.health -= 1

        # occupied：新头写入（出界的头不写，死亡判定后清理）
        if in_bounds(new_head, st.w, st.h):
            st.occupied[new_head] = s.id

        if new_head in st.food:
            s.health   = 100
            s.length  += 1
            s.just_ate = True
            st.food.discard(new_head)
            snap["ate_food"] = True
        else:
            s.just_ate = False
            tail = s.body.pop()
            snap["removed_tail"] = tail
            # 尾巴移走：仅当没有其他段在同格时删除
            if st.occupied.get(tail) == s.id and tail not in s.body:
                del st.occupied[tail]

        diff["per_snake"][s.id] = snap

    # ── Step 2: 死亡判定 ────────────────────────────────────────
    # 收集所有活蛇 body[1:] 为障碍集合
    body_segs = set()
    for s in st.alive_snakes():
        for i, seg in enumerate(s.body):
            if i > 0:
                body_segs.add(seg)

    killed_ids = set()
    head_groups: dict = {}
    for sid, h in new_heads.items():
        head_groups.setdefault(h, []).append(sid)

    for sid, head in new_heads.items():
        s = st.get_snake(sid)
        # 出界
        if not in_bounds(head, st.w, st.h):
            killed_ids.add(sid); continue
        # 饿死
        if s.health <= 0:
            killed_ids.add(sid); continue
        # 撞身体（body_segs 已排除所有新头）
        if head in body_segs:
            killed_ids.add(sid); continue

    # 对头碰撞（同格到达的多条蛇）
    for h, ids in head_groups.items():
        if len(ids) < 2:
            continue
        lengths = [(sid, st.get_snake(sid).length) for sid in ids]
        max_len = max(L for _, L in lengths)
        max_cnt = sum(1 for _, L in lengths if L == max_len)
        for sid, L in lengths:
            if L < max_len or max_cnt > 1:
                killed_ids.add(sid)

    # ── Step 3: 死亡处理 ────────────────────────────────────────
    for sid in killed_ids:
        s = st.get_snake(sid)
        diff["killed"].append(sid)
        for seg in s.body:
            if st.occupied.get(seg) == sid:
                del st.occupied[seg]
        s.alive = False

    st.turn += 1
    st.undo_stack.append(diff)


def undo(st: GameState):
    """
    严格逆序回滚：
    1. 复活死蛇（标记 alive=True）
    2. 用 pre_move_snap 恢复所有移动过蛇的原始状态（含死蛇）
    3. 恢复被吃的食物
    4. 全量重建 occupied（简单正确）
    """
    diff = st.undo_stack.pop()
    st.turn = diff["turn"]

    # 1. 复活死蛇
    for sid in diff["killed"]:
        s = st.get_snake(sid)
        s.alive = True

    # 2. 用移动前快照恢复所有蛇（含死蛇，统一处理，不跳过）
    for sid, snap in diff["pre_move_snap"].items():
        s = st.get_snake(sid)
        s.body     = deque(snap["body"])
        s.health   = snap["health"]
        s.length   = snap["length"]
        s.just_ate = snap["just_ate"]

    # 3. 恢复被吃的食物
    for sid, snap in diff["per_snake"].items():
        if snap["ate_food"]:
            st.food.add(snap["added_head"])

    # 4. 全量重建 occupied
    st.occupied = {}
    for s in st.alive_snakes():
        for seg in s.body:
            st.occupied[seg] = s.id


# ─────────────────────────────────────────────────────────────
# 合法走法
# ─────────────────────────────────────────────────────────────

def legal_moves_for(st: GameState, snake: Snake):
    """
    返回不会立即出界/撞身的方向列表。
    对手的尾巴若下回合会移走（not just_ate），视为可踩。
    """
    head = snake.body[0]
    out  = []
    for name, d in DIRS.items():
        nxt = (head[0] + d[0], head[1] + d[1])
        if not in_bounds(nxt, st.w, st.h):
            continue
        owner_id = st.occupied.get(nxt)
        if owner_id is None:
            out.append(name)
            continue
        owner = st.get_snake(owner_id)
        if owner is None:
            out.append(name)
            continue
        # 占用方的尾下回合会移走 → 可踩
        if (not owner.just_ate) and nxt == owner.body[-1] and len(owner.body) > 1:
            out.append(name)
    return out


# ─────────────────────────────────────────────────────────────
# Voronoi（中立格修复）
# ─────────────────────────────────────────────────────────────

def voronoi_areas(st: GameState) -> dict:
    """
    多源 BFS：所有活蛇头同时扩散。
    同步到达且等长 → NEUTRAL 锁死，永不扩展、永不被覆盖。
    返回 {snake_id: 领地格数}
    """
    dist = {}
    q    = deque()
    for s in st.alive_snakes():
        head = s.body[0]
        dist[head] = (0, s.id, s.length)
        q.append((head, 0, s.id, s.length))

    areas = {s.id: 0 for s in st.alive_snakes()}

    while q:
        (x, y), step, sid, length = q.popleft()
        # 中立格不再扩展
        if sid is NEUTRAL:
            continue
        # lazy deletion：确认该记录仍有效
        if dist.get((x, y)) != (step, sid, length):
            continue
        if step > 0:
            areas[sid] = areas.get(sid, 0) + 1

        for dx, dy in DIR_VECS:
            nxt = (x + dx, y + dy)
            if not in_bounds(nxt, st.w, st.h):
                continue
            if nxt in st.occupied:
                continue
            new_step = step + 1
            old = dist.get(nxt)
            if old is None:
                dist[nxt] = (new_step, sid, length)
                q.append((nxt, new_step, sid, length))
            elif old[0] == new_step and old[1] is not NEUTRAL and old[1] != sid:
                # 同步到达
                if length > old[2]:
                    # 我方更长，抢占
                    dist[nxt] = (new_step, sid, length)
                    q.append((nxt, new_step, sid, length))
                elif length == old[2]:
                    # 等长 → 永久中立
                    dist[nxt] = (new_step, NEUTRAL, 0)
                # 对方更长：保持不变

    return areas


# ─────────────────────────────────────────────────────────────
# 一次 BFS：可达空间 + 最近食物真实距离
# ─────────────────────────────────────────────────────────────

def reach_and_food_bfs(start, st: GameState):
    """
    从 start 出发单次 BFS，同时输出：
    - reach: 可达格数（不含 start）
    - nearest_food_dist: 最近食物的真实步数（None = 不可达）
    """
    if not in_bounds(start, st.w, st.h) or start in st.occupied:
        return 0, None

    seen  = {start}
    q     = deque([(start, 0)])
    reach = 0
    nearest_food_dist = None

    while q:
        (x, y), d = q.popleft()
        reach += 1
        if d > 0 and (x, y) in st.food:
            if nearest_food_dist is None:
                nearest_food_dist = d

        for dx, dy in DIR_VECS:
            nxt = (x + dx, y + dy)
            if nxt in seen or not in_bounds(nxt, st.w, st.h):
                continue
            if nxt in st.occupied:
                continue
            seen.add(nxt)
            q.append((nxt, d + 1))

    return reach - 1, nearest_food_dist   # reach 不含起点


# ─────────────────────────────────────────────────────────────
# 估值函数
# ─────────────────────────────────────────────────────────────

def evaluate(st: GameState) -> float:
    me = st.me()
    if me is None or me.health <= 0:
        return -10000.0

    head   = me.body[0]
    total  = st.w * st.h

    # ① Voronoi 领地差（对抗信号）
    areas      = voronoi_areas(st)
    my_area    = areas.get(st.me_id, 0)
    enemy_area = sum(v for k, v in areas.items() if k != st.me_id)
    voronoi_score = (my_area - enemy_area) / total   # [-1, 1]

    # ② 真实可达空间 + 最近食物距离（一次 BFS）
    reach, food_bfs_dist = reach_and_food_bfs(head, st)

    # 陷阱信号：可达 < 自身长度时触发
    if reach < me.length:
        reach_score = -1.0 + (reach / max(me.length, 1)) * 0.5
    else:
        reach_score = min(reach / total, 1.0)

    # ③ 食物分
    max_dist = st.w + st.h
    if food_bfs_dist is not None:
        food_score = 1.0 - food_bfs_dist / max_dist
    elif st.food:
        # 不可达时用 Manhattan 兜底，分数减半
        md = min(abs(head[0]-fx) + abs(head[1]-fy) for fx, fy in st.food)
        food_score = (1.0 - md / max_dist) * 0.3
    else:
        food_score = 0.0

    if me.health < 30:
        food_w = 1.5
    elif me.health < 60:
        food_w = 0.6
    else:
        food_w = 0.2

    # ④ 中心偏置（轻微，防贴边）
    cx, cy = (st.w - 1) / 2.0, (st.h - 1) / 2.0
    center_score = 1.0 - (abs(head[0] - cx) + abs(head[1] - cy)) / (cx + cy)

    # ⑤ 长度激励
    length_cap   = total / 4
    length_score = min(me.length / length_cap, 1.0)

    # ⑥ 对手数量（越多越危险）
    enemies_alive = sum(1 for s in st.alive_snakes() if s.id != st.me_id)

    return (voronoi_score  * 2.5
            + reach_score  * 2.0
            + food_score   * food_w
            + center_score * 0.1
            + length_score * 0.4
            - enemies_alive * 0.15)


# ─────────────────────────────────────────────────────────────
# Paranoid Minimax + 迭代加深
# ─────────────────────────────────────────────────────────────

MAX_TRACKED_ENEMIES = 2


def relevant_enemies(st: GameState):
    me = st.me()
    if me is None:
        return []
    head = me.body[0]
    es = [s for s in st.alive_snakes() if s.id != st.me_id]
    es.sort(key=lambda s: abs(s.body[0][0] - head[0]) + abs(s.body[0][1] - head[1]))
    return es[:MAX_TRACKED_ENEMIES]


def enumerate_joint(st: GameState, snakes: list):
    """枚举多条对手蛇的联合走法（笛卡尔积）。"""
    if not snakes:
        return [{}]
    results = [{}]
    for s in snakes:
        moves = legal_moves_for(st, s) or ["up"]
        new_results = []
        for combo in results:
            for m in moves:
                nc = dict(combo)
                nc[s.id] = m
                new_results.append(nc)
        results = new_results
    return results


class TimeUp(Exception):
    pass


def search(st: GameState, depth: int, alpha: float, beta: float,
           deadline: float, to_move: str) -> float:
    """
    标准 Minimax，双方各走一步各消耗 1 depth。
    to_move: "me" | "enemies"
    """
    if time.monotonic() > deadline:
        raise TimeUp

    me = st.me()
    if me is None:
        return -10000.0 - depth   # 越早死越糟

    if depth == 0:
        return evaluate(st)

    if to_move == "me":
        moves = legal_moves_for(st, me)
        if not moves:
            return -5000.0 - depth

        # Move ordering：邻居自由格数多的先搜（O(4) 近似启发）
        head = me.body[0]
        scored = []
        for m in moves:
            d = DIRS[m]
            nxt = (head[0] + d[0], head[1] + d[1])
            free = sum(
                1 for dd in DIR_VECS
                if in_bounds((nxt[0]+dd[0], nxt[1]+dd[1]), st.w, st.h)
                and (nxt[0]+dd[0], nxt[1]+dd[1]) not in st.occupied
            )
            scored.append((free, m))
        scored.sort(reverse=True)

        best = -float("inf")
        for _, m in scored:
            apply_moves(st, {st.me_id: m})
            try:
                v = search(st, depth - 1, alpha, beta, deadline, "enemies")
            finally:
                undo(st)
            if v > best:
                best = v
            if best > alpha:
                alpha = best
            if alpha >= beta:
                break
        return best

    else:  # "enemies"
        enemies = relevant_enemies(st)
        if not enemies:
            return search(st, depth - 1, alpha, beta, deadline, "me")

        joints = enumerate_joint(st, enemies)
        worst  = float("inf")
        for combo in joints:
            apply_moves(st, combo)
            try:
                v = search(st, depth - 1, alpha, beta, deadline, "me")
            finally:
                undo(st)
            if v < worst:
                worst = v
            if worst < beta:
                beta = worst
            if alpha >= beta:
                break
        return worst


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def choose_move(data) -> str:
    st = state_from_request(data)
    me = st.me()
    if me is None:
        return "up"

    deadline = time.monotonic() + 0.40   # 500ms 限制，留 100ms 余量

    legal = legal_moves_for(st, me)
    if not legal:
        # 全死局面：选空间最大的方向硬撑
        head = me.body[0]
        best, best_sp = "up", -1
        for n, d in DIRS.items():
            nxt = (head[0]+d[0], head[1]+d[1])
            if not in_bounds(nxt, st.w, st.h):
                continue
            sp, _ = reach_and_food_bfs(nxt, st)
            if sp > best_sp:
                best, best_sp = n, sp
        return best

    best_move = legal[0]
    # 迭代加深：每次增加 2 ply（一个完整回合）
    for depth in range(2, 14, 2):
        try:
            cur_best = legal[0]
            cur_val  = -float("inf")
            for m in legal:
                apply_moves(st, {st.me_id: m})
                try:
                    v = search(st, depth - 1, -float("inf"), float("inf"),
                               deadline, "enemies")
                finally:
                    undo(st)
                if v > cur_val:
                    cur_val  = v
                    cur_best = m
            best_move = cur_best
        except TimeUp:
            break

    return best_move



