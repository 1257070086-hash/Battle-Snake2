"""
BattleSnake AI — logic.py v9.0
================================
核心策略：Flood Fill Minimax

唯一目标：让 flood_fill(我的头) - flood_fill(对手的头) 最大化。

evaluate = my_ff - enemy_ff_avg
  - my_ff 下降（被困） → 不走
  - enemy_ff 下降（被压） → 去做
  - 血快见底不吃食物 → 几步后死 → my_ff 归零 → 自然去吃

没有 Voronoi、没有 food_score、没有 center_bias、没有权重互搏。
"""

import time
from collections import deque

DIRECTIONS = {
    "up":    ( 0,  1),
    "down":  ( 0, -1),
    "left":  (-1,  0),
    "right": ( 1,  0),
}
DIR_LIST  = list(DIRECTIONS.values())
OPPOSITES = {"up": "down", "down": "up", "left": "right", "right": "left"}

MAX_DEPTH  = 8
TIME_LIMIT = 0.380   # Railway 500ms 限制，留 120ms 余量

_last_move: dict = {}   # snake_id → 上一步方向（跨回合防 U 型回头）


# ─────────────────────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────────────────────

def choose_move(data: dict) -> str:
    start = time.time()

    board  = data["board"]
    me     = data["you"]
    width  = board["width"]
    height = board["height"]
    snakes = board["snakes"]
    food   = [(f["x"], f["y"]) for f in board["food"]]

    state = build_state(me, snakes, food, width, height)
    state["last_dir"] = _last_move.get(me["id"])

    safe = get_safe_moves(state["my_head"], state, me["id"],
                          state["my_length"], width, height)
    if not safe:
        return "up"
    if len(safe) == 1:
        mv = list(safe.keys())[0]
        _last_move[me["id"]] = mv
        return mv

    # flood fill 排序：大的先搜，Alpha-Beta 剪枝更高效
    safe = sort_by_ff(safe, state["occupied"], width, height)

    # ── 血量紧急：直接贪心走向最近安全食物，不经过 Minimax ──────
    # health < 60：比对手短时主动找食物发育；< 35：快死了必须吃
    health = me["health"]
    snakes_alive = board["snakes"]
    max_enemy_len = max((s["length"] for s in snakes_alive if s["id"] != me["id"]), default=0)
    need_grow = (my_len := state["my_length"]) <= max_enemy_len  # 比对手短
    urgent = health < 35 or (health < 60 and need_grow)
    if urgent and state["food_set"]:
        nearest_food = min(
            state["food_set"],
            key=lambda f: manhattan(state["my_head"], f)
        )
        best_food_mv = min(
            safe.items(),
            key=lambda kv: manhattan(kv[1], nearest_food)
        )
        _last_move[me["id"]] = best_food_mv[0]
        return best_food_mv[0]

    best = list(safe.keys())[0]
    for depth in range(2, MAX_DEPTH + 1, 2):     # 2 → 4 → 6 → 8
        if time.time() - start > TIME_LIMIT * 0.75:
            break
        mv, _ = alphabeta_root(state, safe, me["id"], width, height,
                                depth, start)
        if mv:
            best = mv

    _last_move[me["id"]] = best
    return best


# ─────────────────────────────────────────────────────────────
# 走法排序
# ─────────────────────────────────────────────────────────────

def sort_by_ff(moves: dict, occupied: set, width: int, height: int) -> dict:
    def score(kv):
        direction, (nx, ny) = kv
        ff = flood_fill(nx, ny, width, height, occupied, max_cells=150)
        # 同分时：离边界越远越好（最小 Chebyshev 距离到边）
        edge_dist = min(nx, width-1-nx, ny, height-1-ny)
        return (ff, edge_dist)   # ff 优先，同分时 edge_dist 打破平局

    scored = sorted(moves.items(), key=score, reverse=True)
    return dict(scored)


# ─────────────────────────────────────────────────────────────
# Alpha-Beta Minimax
# ─────────────────────────────────────────────────────────────

def alphabeta_root(state, safe, me_id, width, height, depth, start):
    best_mv    = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta  = float("inf")
    last_dir = state.get("last_dir")

    for direction, my_next in safe.items():
        if time.time() - start > TIME_LIMIT:
            break
        ns = step_state(state, me_id, my_next, width, height)
        sc = alphabeta(ns, depth - 1, alpha, beta, False,
                       me_id, width, height, start)

        # 轻微回头惩罚（防 U 型往复，不影响真正好的回头）
        if last_dir and OPPOSITES.get(last_dir) == direction:
            sc -= 3

        if sc > best_score:
            best_score = sc
            best_mv    = direction
        alpha = max(alpha, best_score)

    return best_mv, best_score


def alphabeta(state, depth, alpha, beta, is_maximizing,
              me_id, width, height, start):
    if depth == 0 or time.time() - start > TIME_LIMIT:
        return evaluate(state, me_id, width, height)

    if is_maximizing:
        safe = get_safe_moves(state["my_head"], state, me_id,
                              state["my_length"], width, height)
        if not safe:
            return -10000   # 我死局

        val = float("-inf")
        for _, my_next in safe.items():
            ns = step_state(state, me_id, my_next, width, height)
            # 吃食物即时奖励：Minimax 内部感知到"吃了变长"的价值
            # 短蛇发育奖励大，长蛇吃食物奖励小
            ate_bonus = 0
            if ns.get("my_just_ate"):
                ate_bonus = max(0, 20 - ns["my_length"] * 1)
            child = alphabeta(ns, depth - 1, alpha, beta,
                              False, me_id, width, height, start)
            val = max(val, child + ate_bonus)
            alpha = max(alpha, val)
            if val >= beta:
                break
        return val

    else:
        # 最近的活着的敌蛇
        enemies = [s for s in state["live_snakes"]
                   if s["id"] != me_id and s["health"] > 0]
        if not enemies:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me_id, width, height, start)

        nearest = min(enemies, key=lambda s: manhattan(
            state["my_head"], (s["head"]["x"], s["head"]["y"])
        ))
        e_head = (nearest["head"]["x"], nearest["head"]["y"])
        e_safe = get_safe_moves(e_head, state, nearest["id"],
                                nearest["length"], width, height)
        if not e_safe:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me_id, width, height, start)

        val = float("inf")
        for _, e_next in e_safe.items():
            ns  = step_state(state, nearest["id"], e_next, width, height)
            val = min(val, alphabeta(ns, depth - 1, alpha, beta,
                                      True, me_id, width, height, start))
            beta = min(beta, val)
            if val <= alpha:
                break
        return val


# ─────────────────────────────────────────────────────────────
# 估值函数 — 唯一核心：my_ff - enemy_ff
# ─────────────────────────────────────────────────────────────

def evaluate(state, me_id, width, height):
    my_pos = state["my_head"]
    my_len = state["my_length"]
    occ    = state["occupied"]

    # 我的可达空间
    my_ff = flood_fill(my_pos[0], my_pos[1], width, height, occ)

    # 我被困死 → 最坏
    if my_ff < my_len:
        return -10000

    enemies = [s for s in state["live_snakes"]
               if s["id"] != me_id and s["health"] > 0]

    if not enemies:
        return my_ff   # 无对手，空间越大越好

    # 对手的可达空间（取平均，多蛇时不偏向某一条）
    en_ff_total = sum(
        flood_fill(s["head"]["x"], s["head"]["y"], width, height, occ)
        for s in enemies
    )
    en_ff_avg = en_ff_total / len(enemies)

    # 对手全部被困死 → 最好
    if en_ff_avg < 1:
        return 10000

    # 长度优势：比对手长 = 头对头安全 + 切割能力更强
    # 每长一格约等于 2 格空间优势（经验值，可调）
    max_enemy_len = max(s["length"] for s in enemies)
    len_bonus = (my_len - max_enemy_len) * 2

    return my_ff - en_ff_avg + len_bonus


# ─────────────────────────────────────────────────────────────
# 安全走法过滤（两层）
# ─────────────────────────────────────────────────────────────

def get_safe_moves(head, state, my_id, my_len, width, height):
    occ    = state["occupied"]
    snakes = state["live_snakes"]

    # 第一层：边界 + 蛇身（硬过滤，不可妥协）
    layer1 = {}
    for direction, (dx, dy) in DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            continue
        if (nx, ny) in occ:
            continue
        layer1[direction] = (nx, ny)

    if not layer1:
        return layer1

    # 第二层：头对头危险（软过滤：全部危险时退回 layer1）
    layer2 = {}
    for direction, pos in layer1.items():
        risky = any(
            s["id"] != my_id
            and manhattan(pos, (s["head"]["x"], s["head"]["y"])) == 1
            and s["length"] >= my_len
            for s in snakes
        )
        if not risky:
            layer2[direction] = pos

    return layer2 if layer2 else layer1


# ─────────────────────────────────────────────────────────────
# 状态构建与推进
# ─────────────────────────────────────────────────────────────

def build_state(me, snakes, food, width, height):
    food_set = set(food)
    occupied = set()

    for snake in snakes:
        body = snake["body"]
        # 头尾重叠 = 上回合刚吃食物，尾部不收缩
        just_ate = (len(body) >= 2
                    and body[0]["x"] == body[1]["x"]
                    and body[0]["y"] == body[1]["y"])
        for i, seg in enumerate(body):
            if i == len(body) - 1 and not just_ate:
                continue   # 尾部下回合移走，不计入障碍
            occupied.add((seg["x"], seg["y"]))

    return {
        "occupied":    occupied,
        "food_set":    food_set,
        "my_head":     (me["head"]["x"], me["head"]["y"]),
        "my_length":   me["length"],
        "my_health":   me["health"],
        "live_snakes": list(snakes),
        "me_id":       me["id"],
    }


def step_state(state, snake_id, new_head, width, height):
    """
    推进一步：
    - 目标蛇：头前进，判断吃食物，更新 body/health/length
    - 其余蛇：尾部释放（保守模拟——它们也在移动）
    - 死蛇移除
    """
    food_set    = state["food_set"]
    new_food    = set(food_set)
    new_occ     = set()
    new_snakes  = []

    new_my_head   = state["my_head"]
    new_my_length = state["my_length"]
    new_my_health = state["my_health"]
    new_my_ate    = False

    for s in state["live_snakes"]:
        body  = list(s["body"])
        new_s = dict(s)

        if s["id"] == snake_id:
            ate        = new_head in food_set
            new_body   = [{"x": new_head[0], "y": new_head[1]}] + body
            if not ate:
                new_body = new_body[:-1]
            if ate:
                new_food.discard(new_head)
            new_s["body"]   = new_body
            new_s["head"]   = {"x": new_head[0], "y": new_head[1]}
            new_s["length"] = len(new_body)
            new_s["health"] = 100 if ate else max(0, s["health"] - 1)

            if s["id"] == state.get("me_id"):
                new_my_head   = new_head
                new_my_length = new_s["length"]
                new_my_health = new_s["health"]
                new_my_ate    = ate
        else:
            # 对手蛇：尾部出队（尾部肯定移走）
            en_just_ate = (len(body) >= 2
                           and body[0]["x"] == body[1]["x"]
                           and body[0]["y"] == body[1]["y"])
            new_body = body[:-1] if (not en_just_ate and len(body) > 1) else body
            new_s["body"]   = new_body
            new_s["length"] = len(new_body)
            new_s["health"] = max(0, s["health"] - 1)

        if new_s["health"] <= 0:
            continue   # 死蛇移除

        # 重建 occupied（尾部不算）
        sb       = new_s["body"]
        ate_flag = (len(sb) >= 2
                    and sb[0]["x"] == sb[1]["x"]
                    and sb[0]["y"] == sb[1]["y"])
        for i, seg in enumerate(sb):
            if i == len(sb) - 1 and not ate_flag:
                continue
            new_occ.add((seg["x"], seg["y"]))

        new_snakes.append(new_s)

    return {
        "occupied":    new_occ,
        "food_set":    new_food,
        "my_head":     new_my_head,
        "my_length":   new_my_length,
        "my_health":   new_my_health,
        "my_just_ate": new_my_ate,
        "live_snakes": new_snakes,
        "me_id":       state.get("me_id"),
    }


# ─────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────

def flood_fill(sx, sy, width, height, occupied, max_cells=300):
    visited = {(sx, sy)}
    queue   = deque([(sx, sy)])
    while queue and len(visited) < max_cells:
        cx, cy = queue.popleft()
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if (nx, ny) in visited:
                continue
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) in occupied:
                continue
            visited.add((nx, ny))
            queue.append((nx, ny))
    return len(visited)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
