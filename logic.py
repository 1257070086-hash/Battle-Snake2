import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ==================== 常量 ====================

DIRS = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}
DIR_VECS = list(DIRS.values())
NEUTRAL = "__NEUTRAL__"          # Voronoi 中立格哨兵
MAX_TRACKED_ENEMIES = 2          # 完整模拟最近的几条对手蛇
BEAM_SIZE = 16                   # 多人局对手联合走法截断数
DEADLINE_BUDGET = 0.40           # 决策时间预算（秒），默认 500ms 留 100ms 余量


# ==================== 数据结构 ====================

@dataclass
class Snake:
    id: str
    body: deque                  # (x, y) tuples，body[0] 是头
    health: int
    length: int
    alive: bool = True
    just_ate: bool = False       # 本回合是否刚吃食


@dataclass
class GameState:
    w: int
    h: int
    turn: int
    snakes: list                 # list[Snake]
    food: set                    # set[(x, y)]
    me_id: str
    occupied: dict = field(default_factory=dict)   # (x, y) -> snake_id
    undo_stack: list = field(default_factory=list)

    def me(self) -> Optional[Snake]:
        for s in self.snakes:
            if s.id == self.me_id and s.alive:
                return s
        return None

    def alive_snakes(self):
        return [s for s in self.snakes if s.alive]

    def get_snake(self, sid):
        for s in self.snakes:
            if s.id == sid:
                return s
        return None


def state_from_request(data) -> GameState:
    board = data["board"]
    snakes = []
    occupied = {}
    for s in board["snakes"]:
        body = deque((p["x"], p["y"]) for p in s["body"])
        just_ate = (data["turn"] > 0) and (s["health"] == 100)
        snakes.append(Snake(s["id"], body, s["health"], s["length"],
                            just_ate=just_ate))
        for seg in body:
            occupied[seg] = s["id"]
    return GameState(
        w=board["width"], h=board["height"], turn=data["turn"],
        snakes=snakes,
        food=set((f["x"], f["y"]) for f in board["food"]),
        me_id=data["you"]["id"],
        occupied=occupied,
    )


def in_bounds(p, w, h):
    return 0 <= p[0] < w and 0 <= p[1] < h


# ==================== Make / Unmake ====================

def apply_moves(st: GameState, moves: dict):
    """
    moves: {snake_id: dir_name}，覆盖每条 alive 蛇的方向。
    pre_move_snapshot 在 Step 0 保存，作为 undo 的统一回滚基准。
    """
    diff = {
        "turn": st.turn,
        "pre_move_snapshot": {},   # sid -> 移动前完整快照
        "per_snake": {},           # sid -> {ate_food, added_head}（仅用于食物恢复）
        "killed": [],              # list[sid]
    }

    # Step 0: 保存所有即将移动蛇的"移动前快照"
    for s in st.alive_snakes():
        if s.id not in moves:
            continue
        diff["pre_move_snapshot"][s.id] = {
            "body": list(s.body),
            "health": s.health,
            "length": s.length,
            "just_ate": s.just_ate,
        }

    # Step 1: 推进每条蛇
    new_heads = {}
    for s in st.alive_snakes():
        if s.id not in moves:
            continue
        d = DIRS[moves[s.id]]
        old_head = s.body[0]
        new_head = (old_head[0] + d[0], old_head[1] + d[1])
        new_heads[s.id] = new_head

        s.body.appendleft(new_head)
        s.health -= 1

        per_snap = {"added_head": new_head, "ate_food": False}

        if new_head in st.food:
            s.health = 100
            s.length += 1
            s.just_ate = True
            st.food.discard(new_head)
            per_snap["ate_food"] = True
            # 不缩尾
        else:
            s.just_ate = False
            s.body.pop()  # 缩尾

        diff["per_snake"][s.id] = per_snap

    # Step 2: 收集所有蛇的 body[1:] 用于撞身判定
    body_segs = set()
    for s in st.alive_snakes():
        for i, seg in enumerate(s.body):
            if i == 0:
                continue
            body_segs.add(seg)

    # Step 3: 死亡判定
    killed_ids = set()
    head_count = {}
    for sid, h in new_heads.items():
        head_count.setdefault(h, []).append(sid)

    for sid, head in new_heads.items():
        s = st.get_snake(sid)
        if not in_bounds(head, st.w, st.h):
            killed_ids.add(sid); continue
        if s.health <= 0:
            killed_ids.add(sid); continue
        if head in body_segs:
            killed_ids.add(sid); continue

    # 对头碰撞
    for h, ids in head_count.items():
        if len(ids) < 2:
            continue
        lengths = [(sid, st.get_snake(sid).length) for sid in ids]
        max_len = max(L for _, L in lengths)
        max_count = sum(1 for _, L in lengths if L == max_len)
        for sid, L in lengths:
            if L < max_len or max_count > 1:
                killed_ids.add(sid)

    # Step 4: 死亡处理
    for sid in killed_ids:
        diff["killed"].append(sid)
        s = st.get_snake(sid)
        s.alive = False

    st.turn += 1

    # Step 5: 全量重建 occupied（牺牲性能换正确性）
    st.occupied = {}
    for s in st.alive_snakes():
        for seg in s.body:
            st.occupied[seg] = s.id

    st.undo_stack.append(diff)


def undo(st: GameState):
    diff = st.undo_stack.pop()
    st.turn = diff["turn"]

    # 1. 死蛇复活
    for sid in diff["killed"]:
        s = st.get_snake(sid)
        s.alive = True

    # 2. 用 pre_move_snapshot 整体回滚（含死蛇）
    for sid, snap in diff["pre_move_snapshot"].items():
        s = st.get_snake(sid)
        s.body = deque(snap["body"])
        s.health = snap["health"]
        s.length = snap["length"]
        s.just_ate = snap["just_ate"]

    # 3. 恢复被吃的食物
    for sid, per in diff["per_snake"].items():
        if per["ate_food"]:
            st.food.add(per["added_head"])

    # 4. 重建 occupied
    st.occupied = {}
    for s in st.alive_snakes():
        for seg in s.body:
            st.occupied[seg] = s.id


# ==================== 合法走法 ====================

def legal_moves_for(st: GameState, snake: Snake):
    """不会立即出界 / 撞身的方向。会移走的尾巴格视为可踩。"""
    head = snake.body[0]
    out = []
    for name, d in DIRS.items():
        nxt = (head[0] + d[0], head[1] + d[1])
        if not in_bounds(nxt, st.w, st.h):
            continue
        owner_id = st.occupied.get(nxt)
        if owner_id is None:
            out.append(name); continue
        owner = st.get_snake(owner_id)
        if owner is None or not owner.alive:
            out.append(name); continue
        # 占用方的尾巴下回合会移走（除非它刚吃食）
        if (not owner.just_ate) and len(owner.body) > 1 and nxt == owner.body[-1]:
            out.append(name)
    return out


# ==================== 评估辅助 ====================

def flood_fill_from(start, st: GameState, limit=None):
    """从 start 出发的纯 flood-fill，用于陷阱检测。"""
    if not in_bounds(start, st.w, st.h):
        return 0
    if start in st.occupied:
        return 0
    seen = {start}
    q = deque([start])
    while q:
        if limit and len(seen) >= limit:
            break
        cur = q.popleft()
        for dx, dy in DIR_VECS:
            nxt = (cur[0] + dx, cur[1] + dy)
            if nxt in seen or not in_bounds(nxt, st.w, st.h):
                continue
            if nxt in st.occupied:
                continue
            seen.add(nxt); q.append(nxt)
    return len(seen)


def voronoi_areas(st: GameState):
    """multi-source BFS：每条活蛇头同时扩散，得到领地归属。
       中立格用 NEUTRAL 哨兵永久锁死。"""
    dist = {}
    q = deque()
    for s in st.alive_snakes():
        head = s.body[0]
        dist[head] = (0, s.id, s.length)
        q.append((head, 0, s.id, s.length))

    areas = {s.id: 0 for s in st.alive_snakes()}

    while q:
        (x, y), step, sid, length = q.popleft()
        if sid is NEUTRAL:
            continue
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
            elif old[0] == new_step and old[1] != sid and old[1] is not NEUTRAL:
                if length > old[2]:
                    dist[nxt] = (new_step, sid, length)
                    q.append((nxt, new_step, sid, length))
                elif length == old[2]:
                    dist[nxt] = (new_step, NEUTRAL, 0)   # 永久中立
    return areas


def food_distance_score(st: GameState, head):
    """BFS 真实路径距离，找最近食物。不可达时退化为 Manhattan 兜底（×0.3）。"""
    if not st.food:
        return 0.0
    max_dist = st.w + st.h

    seen = {head}
    q = deque([(head, 0)])
    while q:
        (x, y), d = q.popleft()
        if (x, y) in st.food and (x, y) != head:
            return 1.0 - d / max_dist
        if d >= max_dist:
            continue
        for dx, dy in DIR_VECS:
            nxt = (x + dx, y + dy)
            if nxt in seen or not in_bounds(nxt, st.w, st.h):
                continue
            if nxt in st.occupied:
                continue
            seen.add(nxt); q.append((nxt, d + 1))

    # 不可达兜底
    md = min(abs(head[0] - fx) + abs(head[1] - fy) for fx, fy in st.food)
    return (1.0 - md / max_dist) * 0.3


# ==================== 评估函数 ====================

def evaluate(st: GameState):
    me = st.me()
    if me is None or me.health <= 0:
        return -10000.0
    head = me.body[0]

    # 1. Voronoi 领地
    areas = voronoi_areas(st)
    my_area = areas.get(st.me_id, 0)
    enemy_area = sum(v for k, v in areas.items() if k != st.me_id)
    voronoi_score = (my_area - enemy_area) / (st.w * st.h)

    # 2. 真实可达空间（陷阱信号，与 Voronoi 解耦）
    reach = flood_fill_from(head, st, limit=st.w * st.h)
    if reach < me.length:
        reach_score = -1.0 + (reach / max(me.length, 1)) * 0.5
    else:
        reach_score = min(reach / (me.length * 2), 1.0)

    # 3. 食物（BFS 真实距离）
    food_score = food_distance_score(st, head)
    if me.health < 30:
        food_w = 1.5
    elif me.health < 60:
        food_w = 0.6
    else:
        food_w = 0.2

    # 4. 中心倾向（弱 tie-breaker）
    cx, cy = (st.w - 1) / 2, (st.h - 1) / 2
    max_cd = cx + cy
    center_score = 1.0 - (abs(head[0] - cx) + abs(head[1] - cy)) / max_cd

    # 5. 长度（自适应上界）
    length_cap = st.w * st.h / 4
    length_score = min(me.length / length_cap, 1.0)

    # 6. 对手数量惩罚
    enemies_alive = sum(1 for s in st.alive_snakes() if s.id != st.me_id)

    return (voronoi_score * 2.5 +
            reach_score   * 2.0 +
            food_score    * food_w +
            center_score  * 0.1 +
            length_score  * 0.4 -
            enemies_alive * 0.15)


# ==================== Paranoid Negamax + 迭代加深 ====================

class TimeUp(Exception):
    pass


def relevant_enemies(st: GameState):
    me = st.me()
    if me is None:
        return []
    head = me.body[0]
    es = [s for s in st.alive_snakes() if s.id != st.me_id]
    es.sort(key=lambda s: abs(s.body[0][0] - head[0]) + abs(s.body[0][1] - head[1]))
    return es[:MAX_TRACKED_ENEMIES]


def enumerate_joint_with_beam(st: GameState, snakes: list, beam_size=BEAM_SIZE):
    """对手联合走法枚举 + beam 截断。
       每条蛇按"朝向我方头"的启发式排序，最多取前 2 个走法。"""
    if not snakes:
        return [{}]

    me = st.me()
    me_head = me.body[0] if me else None

    per_snake_moves = []
    for s in snakes:
        moves = legal_moves_for(st, s) or ["up"]
        scored = []
        sh = s.body[0]
        for m in moves:
            d = DIRS[m]
            nxt = (sh[0] + d[0], sh[1] + d[1])
            score = 0
            if me_head is not None:
                # 越靠近我方头越优先（攻击性走法先展开，alpha-beta 命中更早）
                score -= abs(nxt[0] - me_head[0]) + abs(nxt[1] - me_head[1])
            scored.append((score, m))
        scored.sort(reverse=True)
        per_snake_moves.append([m for _, m in scored[:2]])

    # 笛卡尔积
    res = [{}]
    for i, s in enumerate(snakes):
        nr = []
        for combo in res:
            for m in per_snake_moves[i]:
                nc = dict(combo); nc[s.id] = m
                nr.append(nc)
        res = nr

    return res[:beam_size]


def search(st: GameState, depth: int, alpha: float, beta: float,
           deadline: float, to_move: str):
    """标准 minimax，每层 depth - 1。"""
    if time.monotonic() > deadline:
        raise TimeUp

    me = st.me()
    if me is None:
        return -10000.0 - depth
    if depth == 0:
        return evaluate(st)

    if to_move == "me":
        moves = legal_moves_for(st, me)
        if not moves:
            return -5000.0 - depth

        # Move ordering：邻居空格越多越优先
        head = me.body[0]
        scored = []
        for m in moves:
            d = DIRS[m]
            nxt = (head[0] + d[0], head[1] + d[1])
            free = sum(1 for dd in DIR_VECS
                       if in_bounds((nxt[0] + dd[0], nxt[1] + dd[1]), st.w, st.h)
                       and (nxt[0] + dd[0], nxt[1] + dd[1]) not in st.occupied)
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
    else:
        enemies = relevant_enemies(st)
        if not enemies:
            return search(st, depth - 1, alpha, beta, deadline, "me")

        joints = enumerate_joint_with_beam(st, enemies)
        worst = float("inf")
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


def choose_move(data):
    st = state_from_request(data)
    me = st.me()
    if me is None:
        return "up"

    deadline = time.monotonic() + DEADLINE_BUDGET
    legal = legal_moves_for(st, me)

    # 全死局面：挑空间最大的方向硬撑
    if not legal:
        head = me.body[0]
        best, best_sp = "up", -1
        for n, d in DIRS.items():
            nxt = (head[0] + d[0], head[1] + d[1])
            if not in_bounds(nxt, st.w, st.h):
                continue
            sp = flood_fill_from(nxt, st)
            if sp > best_sp:
                best, best_sp = n, sp
        return best

    best_move = legal[0]
    # 迭代加深：每次 +2 ply（一个完整回合）
    for depth in range(2, 12, 2):
        try:
            cur_best, cur_val = legal[0], -float("inf")
            for m in legal:
                apply_moves(st, {st.me_id: m})
                try:
                    v = search(st, depth - 1, -float("inf"), float("inf"),
                               deadline, "enemies")
                finally:
                    undo(st)
                if v > cur_val:
                    cur_val, cur_best = v, m
            best_move = cur_best
        except TimeUp:
            break

    return best_move



