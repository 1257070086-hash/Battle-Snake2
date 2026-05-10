"""
BattleSnake AI — logic.py v7.0
架构：Minimax + Alpha-Beta + Voronoi 差值核心

设计原则：
  搜索更深，而不是 evaluate 更复杂。
  evaluate 只保留 2 个正交指标，靠深度预见转圈/绕死/头对头。

evaluate 指标（精简）：
  1. voronoi_diff   = (我的格子 - 对手格子) / 总格子  [主导，阶段权重 1~4]
  2. food_score     = 最近安全食物，阶段权重 + 按需激活
  3. survival       = 死局硬截断 -1000

Minimax 改进：
  - 对手节点：枚举最近敌蛇的所有安全走法，step_state 更新其头部
  - MAX_DEPTH = 8，迭代加深 2→4→6→8，用完 380ms
"""

import time
from collections import deque

DIRECTIONS = {
    "up":    (0,  1),
    "down":  (0, -1),
    "left":  (-1, 0),
    "right": (1,  0),
}
DIR_LIST  = list(DIRECTIONS.values())
OPPOSITES = {"up": "down", "down": "up", "left": "right", "right": "left"}

MAX_DEPTH  = 8
TIME_LIMIT = 0.380   # 秒，Railway 响应限制 500ms，留 120ms 余量


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

# 全局：跨回合记住上一步方向，防 U 型回头
_last_move: dict = {}


def choose_move(data: dict) -> str:
    start_time = time.time()

    board  = data["board"]
    me     = data["you"]
    width  = board["width"]
    height = board["height"]
    snakes = board["snakes"]
    food   = [(f["x"], f["y"]) for f in board["food"]]

    state = build_state(me, snakes, food, width, height)
    state["last_dir"] = _last_move.get(me["id"])

    # 获取安全走法
    safe = get_safe_moves(state["my_head"], state, me["id"], state["my_length"],
                          width, height)
    if not safe:
        return "up"
    if len(safe) == 1:
        move = list(safe.keys())[0]
        _last_move[me["id"]] = move
        return move

    # 走法排序（flood fill 大的先搜，Alpha-Beta 剪枝更高效）
    safe = sort_by_space(safe, state, width, height)

    # 迭代加深 Minimax
    best_move = list(safe.keys())[0]
    for depth in range(2, MAX_DEPTH + 1, 2):   # 2, 4, 6, 8
        if time.time() - start_time > TIME_LIMIT * 0.8:
            break
        move, _ = alphabeta_root(state, safe, me["id"], width, height,
                                 depth, start_time)
        if move:
            best_move = move

    _last_move[me["id"]] = best_move
    return best_move


# ─────────────────────────────────────────────
# 走法排序
# ─────────────────────────────────────────────

def sort_by_space(moves: dict, state, width, height) -> dict:
    """flood fill 排序，近距离食物轻微加成（让 Alpha-Beta 优先搜食物方向）"""
    occupied  = state["occupied"]
    food_list = list(state["food_set"])
    scored = []

    for direction, pos in moves.items():
        ff = flood_fill(pos[0], pos[1], width, height, occupied, max_cells=60)
        food_bonus = 0
        for f in food_list:
            d = manhattan(pos, f)
            if d == 0:   food_bonus = max(food_bonus, 20)
            elif d == 1: food_bonus = max(food_bonus, 5)
        scored.append((direction, pos, ff + food_bonus))

    scored.sort(key=lambda x: x[2], reverse=True)
    return {d: p for d, p, _ in scored}


# ─────────────────────────────────────────────
# Alpha-Beta Minimax
# ─────────────────────────────────────────────

def alphabeta_root(state, safe, me_id, width, height, depth, start_time):
    best_move  = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta  = float("inf")

    last_dir = state.get("last_dir")

    for direction, my_next in safe.items():
        if time.time() - start_time > TIME_LIMIT:
            break
        new_state = step_state(state, me_id, my_next, width, height)
        score = alphabeta(new_state, depth - 1, alpha, beta, False,
                          me_id, width, height, start_time)

        # 回头惩罚：防 U 型来回
        if last_dir and OPPOSITES.get(last_dir) == direction:
            score -= 0.6

        if score > best_score:
            best_score = score
            best_move  = direction
        alpha = max(alpha, best_score)

    return best_move, best_score


def alphabeta(state, depth, alpha, beta, is_maximizing,
              me_id, width, height, start_time):
    if depth == 0 or time.time() - start_time > TIME_LIMIT:
        return evaluate(state, me_id, width, height)

    if is_maximizing:
        safe = get_safe_moves(state["my_head"], state, me_id, state["my_length"],
                              width, height)
        if not safe:
            return -1000

        value = float("-inf")
        for _, my_next in safe.items():
            new_state = step_state(state, me_id, my_next, width, height)
            child = alphabeta(new_state, depth - 1, alpha, beta,
                              False, me_id, width, height, start_time)
            value = max(value, child)
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value

    else:
        # 对手节点：找最近敌蛇，枚举其所有安全走法，取最坏情况
        enemies = [s for s in state["live_snakes"]
                   if s["id"] != me_id and s["health"] > 0]
        if not enemies:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me_id, width, height, start_time)

        nearest = min(enemies, key=lambda s: manhattan(
            state["my_head"], (s["head"]["x"], s["head"]["y"])
        ))
        e_head = (nearest["head"]["x"], nearest["head"]["y"])
        e_safe = get_safe_moves(e_head, state, nearest["id"], nearest["length"],
                                width, height)
        if not e_safe:
            # 对手无路可走，对我有利 → 继续我方节点
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me_id, width, height, start_time)

        value = float("inf")
        for _, e_next in e_safe.items():
            # ✅ 修复：对手走一步，更新其头部位置到 new_state
            new_state = step_state(state, nearest["id"], e_next, width, height)
            child = alphabeta(new_state, depth - 1, alpha, beta,
                              True, me_id, width, height, start_time)
            value = min(value, child)
            beta = min(beta, value)
            if value <= alpha:
                break
        return value


# ─────────────────────────────────────────────
# 估值函数（精简：仅 voronoi_diff + food_score）
# ─────────────────────────────────────────────

def evaluate(state, me_id, width, height):
    my_pos   = state["my_head"]
    my_len   = state["my_length"]
    health   = state["my_health"]
    occupied = state["occupied"]
    snakes   = state["live_snakes"]

    # ① 死局硬截断
    # 用完整 flood fill（不限制 max_cells）更准确判断能否存活
    my_ff = flood_fill(my_pos[0], my_pos[1], width, height, occupied)
    if my_ff < my_len:
        return -1000

    # ② Voronoi 差值（主导指标）
    voronoi_mine, voronoi_total = voronoi_control(state, me_id, width, height)
    voronoi_enemy = voronoi_total - voronoi_mine
    voronoi_diff = (voronoi_mine - voronoi_enemy) / max(voronoi_total, 1)

    # ③ 食物分（阶段权重）
    enemies = [s for s in snakes if s["id"] != me_id]
    max_enemy_len = max((s["length"] for s in enemies), default=0)

    # 阶段权重：短蛇食物优先，长蛇地盘优先
    if my_len < 6:
        w_voronoi, w_food_base = 1.0, 4.0
    elif my_len < 12:
        w_voronoi, w_food_base = 2.5, 2.5
    else:
        w_voronoi, w_food_base = 4.0, 1.0

    # 需求判断：血低或比对手短 → 放大食物权重
    need_food  = (health < 40) or (my_len <= max_enemy_len + 1)
    food_weight = w_food_base * (1.5 if need_food else 0.4)

    best_food_score = 0.0
    for f in state["food_set"]:
        dist = manhattan(my_pos, f)
        base = 1.0 / (1 + dist)
        # 食物被大蛇包围 → 降权（头对头陷阱）
        for s in enemies:
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan(eh, f) <= 1 and s["length"] >= my_len:
                base *= 0.1
                break
        best_food_score = max(best_food_score, base)

    food_score = best_food_score * food_weight

    return voronoi_diff * w_voronoi + food_score


# ─────────────────────────────────────────────
# 安全走法过滤（三层）
# ─────────────────────────────────────────────

def get_safe_moves(head, state, my_id, my_len, width, height):
    occupied = state["occupied"]
    snakes   = state["live_snakes"]

    # 第一层：边界 + 蛇身（硬过滤）
    layer1 = {}
    for direction, (dx, dy) in DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            continue
        if (nx, ny) in occupied:
            continue
        layer1[direction] = (nx, ny)

    if not layer1:
        return layer1

    # 第二层：头对头危险（软过滤）
    layer2 = {}
    for direction, pos in layer1.items():
        risky = False
        for s in snakes:
            if s["id"] == my_id:
                continue
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan(pos, eh) == 1 and s["length"] >= my_len:
                risky = True
                break
        if not risky:
            layer2[direction] = pos

    candidates = layer2 if layer2 else layer1

    # 第三层：空间下限（软过滤）
    spacious = {}
    for direction, pos in candidates.items():
        ff = flood_fill(pos[0], pos[1], width, height, occupied,
                        max_cells=my_len * 2 + 5)
        if ff >= my_len:
            spacious[direction] = pos

    return spacious if spacious else candidates


# ─────────────────────────────────────────────
# 状态构建
# ─────────────────────────────────────────────

def build_state(me, snakes, food, width, height):
    food_set = set(food)
    occupied = set()

    for snake in snakes:
        body = snake["body"]
        just_ate = (len(body) >= 2 and
                    body[0]["x"] == body[1]["x"] and
                    body[0]["y"] == body[1]["y"])
        for i, seg in enumerate(body):
            if i == len(body) - 1 and not just_ate:
                continue  # 尾部下回合移走，不算障碍
            occupied.add((seg["x"], seg["y"]))

    return {
        "occupied":    occupied,
        "food_set":    food_set,
        "my_head":     (me["head"]["x"], me["head"]["y"]),
        "my_length":   me["length"],
        "my_health":   me["health"],
        "my_just_ate": False,
        "live_snakes": list(snakes),
        "me_id":       me["id"],
    }


def step_state(state, snake_id, new_head, width, height):
    """
    推进一步：
    - 目标蛇：头前进、尾出队、吃食物
    - 其余蛇：尾部释放（保守假设它们也在移动）
    """
    food_set     = state["food_set"]
    new_food     = set(food_set)
    new_snakes   = []
    new_occupied = set()

    new_my_head   = state["my_head"]
    new_my_length = state["my_length"]
    new_my_health = state["my_health"]
    new_my_ate    = False

    for s in state["live_snakes"]:
        body = list(s["body"])
        new_s = dict(s)

        if s["id"] == snake_id:
            ate_food   = new_head in food_set
            new_body   = [{"x": new_head[0], "y": new_head[1]}] + body
            if not ate_food:
                new_body = new_body[:-1]
            if ate_food:
                new_food.discard(new_head)
            new_len    = len(new_body)
            new_health = 100 if ate_food else max(0, s["health"] - 1)

            new_s["body"]   = new_body
            new_s["head"]   = {"x": new_head[0], "y": new_head[1]}
            new_s["length"] = new_len
            new_s["health"] = new_health

            if s["id"] == state.get("me_id"):
                new_my_head   = new_head
                new_my_length = new_len
                new_my_health = new_health
                new_my_ate    = ate_food
        else:
            # 对手蛇：尾部出队（不知道它走哪，但尾部肯定移走）
            just_ate_enemy = (len(body) >= 2 and
                              body[0]["x"] == body[1]["x"] and
                              body[0]["y"] == body[1]["y"])
            new_body = body[:-1] if (not just_ate_enemy and len(body) > 1) else body
            new_s["body"]   = new_body
            new_s["length"] = len(new_body)
            new_s["health"] = max(0, s["health"] - 1)

        if new_s["health"] <= 0:
            continue

        # 重建 occupied（尾部不算：下回合会移走）
        s_body   = new_s["body"]
        just_ate = (len(s_body) >= 2 and
                    s_body[0]["x"] == s_body[1]["x"] and
                    s_body[0]["y"] == s_body[1]["y"])
        for i, seg in enumerate(s_body):
            if i == len(s_body) - 1 and not just_ate:
                continue
            new_occupied.add((seg["x"], seg["y"]))

        new_snakes.append(new_s)

    return {
        "occupied":    new_occupied,
        "food_set":    new_food,
        "my_head":     new_my_head,
        "my_length":   new_my_length,
        "my_health":   new_my_health,
        "my_just_ate": new_my_ate,
        "live_snakes": new_snakes,
        "me_id":       state.get("me_id"),
    }


# ─────────────────────────────────────────────
# Voronoi 控制面积
# ─────────────────────────────────────────────

def voronoi_control(state, my_id, width, height):
    """BFS 多源扩展，返回 (我的格子数, 总已分配格子数)"""
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    owner    = {}
    dist_map = {}
    queue    = deque()

    for s in snakes:
        h = (s["head"]["x"], s["head"]["y"])
        if h not in dist_map:
            owner[h]    = s["id"]
            dist_map[h] = 0
            queue.append((h, s["id"]))

    while queue:
        (cx, cy), sid = queue.popleft()
        d = dist_map[(cx, cy)]
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) in occupied or (nx, ny) in dist_map:
                continue
            dist_map[(nx, ny)] = d + 1
            owner[(nx, ny)]    = sid
            queue.append(((nx, ny), sid))

    mine  = sum(1 for sid in owner.values() if sid == my_id)
    total = len(owner)
    return mine, total


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def flood_fill(start_x, start_y, width, height, occupied, max_cells=300):
    visited = {(start_x, start_y)}
    queue   = deque([(start_x, start_y)])
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
