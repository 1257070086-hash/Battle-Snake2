"""
BattleSnake AI — logic.py v6.0
架构：Minimax + Alpha-Beta + Voronoi 差值核心

核心设计：
  空间 = 走的可能性，不是格子数量。
  好的走法 = 走完这步后，我的 Voronoi 格子更多，对手的 Voronoi 格子更少。

evaluate 只有 3 个正交指标：
  1. voronoi_diff   = (我的格子 - 对手格子) / 总格子  [主导，权重 4.0]
  2. food_score     = 最近安全食物吸引力，按需激活   [动态 0.3 ~ 3.0]
  3. survival       = 死局硬截断 -1000

删除的补丁：endgame_strategy / trap_bonus / force_food /
           attack_bonus / position_score / len_adv
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

MAX_DEPTH  = 6
TIME_LIMIT = 0.400   # 秒，超过 350ms 停止扩展


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def choose_move(data: dict) -> str:
    start_time = time.time()

    board  = data["board"]
    me     = data["you"]
    width  = board["width"]
    height = board["height"]
    snakes = board["snakes"]
    food   = [(f["x"], f["y"]) for f in board["food"]]

    state = build_state(me, snakes, food, width, height)

    # 获取安全走法
    safe = get_safe_moves(state["my_head"], state, me["id"], state["my_length"],
                          width, height)
    if not safe:
        return "up"
    if len(safe) == 1:
        return list(safe.keys())[0]

    # 根节点走法排序：flood fill 大的方向先搜，Alpha-Beta 剪枝更高效
    safe = sort_by_space(safe, state, width, height)

    # Minimax 迭代加深
    best_move = list(safe.keys())[0]
    for depth in [4, MAX_DEPTH]:
        if time.time() - start_time > TIME_LIMIT * 0.85:
            break
        move, _ = alphabeta_root(state, safe, me["id"], width, height,
                                 depth, start_time)
        if move:
            best_move = move

    return best_move


# ─────────────────────────────────────────────
# 走法排序（仅用 flood fill，供 Alpha-Beta 剪枝优化）
# ─────────────────────────────────────────────

def sort_by_space(moves: dict, state, width, height) -> dict:
    occupied = state["occupied"]
    scored = []
    for direction, pos in moves.items():
        ff = flood_fill(pos[0], pos[1], width, height, occupied, max_cells=60)
        scored.append((direction, pos, ff))
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

    for direction, my_next in safe.items():
        if time.time() - start_time > TIME_LIMIT:
            break
        new_state = step_state(state, me_id, my_next, width, height)
        score = alphabeta(new_state, depth - 1, alpha, beta, False,
                          me_id, width, height, start_time)
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
            return -1000  # 我方死局

        value = float("-inf")
        for _, my_next in safe.items():
            new_state = step_state(state, me_id, my_next, width, height)
            value = max(value, alphabeta(new_state, depth - 1, alpha, beta,
                                         False, me_id, width, height, start_time))
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value

    else:
        # 对手节点：选距离最近的一条敌蛇，假设它走对我最不利的方向
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
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me_id, width, height, start_time)

        value = float("inf")
        for _, e_next in e_safe.items():
            new_state = step_state(state, nearest["id"], e_next, width, height)
            value = min(value, alphabeta(new_state, depth - 1, alpha, beta,
                                          True, me_id, width, height, start_time))
            beta = min(beta, value)
            if value <= alpha:
                break
        return value


# ─────────────────────────────────────────────
# 估值函数（核心：3 个正交指标）
# ─────────────────────────────────────────────

def evaluate(state, me_id, width, height):
    my_pos   = state["my_head"]
    my_len   = state["my_length"]
    health   = state["my_health"]
    occupied = state["occupied"]
    snakes   = state["live_snakes"]

    # ① 死局硬截断（最高优先级）
    my_ff = flood_fill(my_pos[0], my_pos[1], width, height, occupied, max_cells=my_len * 2)
    if my_ff < my_len:
        return -1000

    # ② Voronoi 差值 = (我的格子 - 对手格子) / 总格子
    #    这才是真正的"空间可能性"：我能先到多少格，对手能先到多少格
    voronoi_mine, voronoi_total = voronoi_control(state, me_id, width, height)
    voronoi_enemy = voronoi_total - voronoi_mine
    voronoi_diff = (voronoi_mine - voronoi_enemy) / max(voronoi_total, 1)

    # ③ 食物分（按需激活）
    enemies = [s for s in snakes if s["id"] != me_id]
    max_enemy_len = max((s["length"] for s in enemies), default=0)

    # 按需条件：血量低 或 比最长对手短
    need_food = (health < 40) or (my_len <= max_enemy_len + 1)
    food_weight = 3.0 if need_food else 0.3

    food_score = 0.0
    food_list = list(state["food_set"])
    if food_list and food_weight > 0.3:  # 不需要食物时不计算，节省性能
        best = 0.0
        for f in food_list:
            dist = manhattan(my_pos, f)
            score = 1.0 / (1 + dist)

            # 降权：食物被大蛇头包围（头对头陷阱）
            for s in enemies:
                eh = (s["head"]["x"], s["head"]["y"])
                if manhattan(eh, f) <= 1 and s["length"] >= my_len:
                    score *= 0.1  # 危险食物，强力降权
                    break

            best = max(best, score)
        food_score = best
    elif food_list:
        # 不需要食物时给极小权重，避免完全忽视血量
        dist = min(manhattan(my_pos, f) for f in food_list)
        food_score = 1.0 / (1 + dist) * 0.1

    return voronoi_diff * 4.0 + food_score * food_weight


# ─────────────────────────────────────────────
# 安全走法过滤（三层过滤）
# ─────────────────────────────────────────────

def get_safe_moves(head, state, my_id, my_len, width, height):
    occupied = state["occupied"]
    snakes   = state["live_snakes"]

    # 第一层：边界 + 蛇身碰撞（最严格，不可绕过）
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

    # 第二层：头对头危险（软过滤：所有方向都危险时退回 layer1）
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

    candidates = layer2 if layer2 else layer1  # 软降级

    # 第三层：空间下限过滤（死路剪枝，软过滤）
    spacious = {}
    for direction, pos in candidates.items():
        ff = flood_fill(pos[0], pos[1], width, height, occupied,
                        max_cells=my_len * 2 + 5)
        if ff >= my_len:
            spacious[direction] = pos

    return spacious if spacious else candidates  # 软降级


# ─────────────────────────────────────────────
# 状态构建
# ─────────────────────────────────────────────

def build_state(me, snakes, food, width, height):
    food_set = set(food)
    occupied = set()

    for snake in snakes:
        body = snake["body"]
        # 判断上回合是否吃了食物（头和第二节重叠 = 刚吃食物，尾部不收缩）
        just_ate = (len(body) >= 2 and
                    body[0]["x"] == body[1]["x"] and
                    body[0]["y"] == body[1]["y"])
        for i, seg in enumerate(body):
            pos = (seg["x"], seg["y"])
            # 尾部：没吃食物时下回合移走，不计入障碍
            if i == len(body) - 1 and not just_ate:
                continue
            occupied.add(pos)

    return {
        "occupied":    occupied,
        "food_set":    food_set,
        "my_head":     (me["head"]["x"], me["head"]["y"]),
        "my_length":   me["length"],
        "my_health":   me["health"],
        "live_snakes": snakes,
        "me_id":       me["id"],
    }


def step_state(state, snake_id, new_head, width, height):
    """
    推进一步棋盘状态：
    - 目标蛇：执行移动（头前进、尾出队、吃食物判断）
    - 其余蛇：尾部同样释放（保守处理：认为它们也在移动）
    - 血量归零的蛇从 live_snakes 移除
    """
    food_set     = state["food_set"]
    new_food     = set(food_set)
    new_snakes   = []
    new_occupied = set()

    new_my_head   = state["my_head"]
    new_my_length = state["my_length"]
    new_my_health = state["my_health"]

    for s in state["live_snakes"]:
        body = list(s["body"])

        if s["id"] == snake_id:
            # 目标蛇：正式移动
            ate_food   = new_head in food_set
            new_body   = [{"x": new_head[0], "y": new_head[1]}] + body
            if not ate_food:
                new_body = new_body[:-1]  # 尾部收缩
            if ate_food:
                new_food.discard(new_head)
            new_len    = len(new_body)
            new_health = 100 if ate_food else max(0, s["health"] - 1)

            new_s = dict(s)
            new_s["body"]   = new_body
            new_s["head"]   = {"x": new_head[0], "y": new_head[1]}
            new_s["length"] = new_len
            new_s["health"] = new_health

            if s["id"] == state.get("me_id"):
                new_my_head   = new_head
                new_my_length = new_len
                new_my_health = new_health
        else:
            # 其余蛇：不知道它走哪，但尾部会释放
            # 保守处理：身体不变，但尾部释放（下一回合 occupied 更新）
            new_health = max(0, s["health"] - 1)
            new_s = dict(s)
            new_s["health"] = new_health

        # 死蛇移除
        if new_s["health"] <= 0:
            continue

        # 重建 occupied
        s_body = new_s["body"]
        # 判断当前蛇是否刚吃（头和第二节重叠）
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
        "live_snakes": new_snakes,
        "me_id":       state.get("me_id"),
    }


# ─────────────────────────────────────────────
# Voronoi 控制面积计算
# ─────────────────────────────────────────────

def voronoi_control(state, my_id, width, height):
    """
    BFS 多源扩展：每个空格分配给最先到达的蛇。
    返回 (我的格子数, 总已分配格子数)
    """
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    owner    = {}
    dist     = {}
    queue    = deque()

    for s in snakes:
        h = (s["head"]["x"], s["head"]["y"])
        if h not in dist:
            owner[h] = s["id"]
            dist[h]  = 0
            queue.append((h, s["id"]))

    while queue:
        (cx, cy), sid = queue.popleft()
        d = dist[(cx, cy)]
        for dx, dy in DIR_LIST:
            nx, ny = cx + dx, cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) in occupied or (nx, ny) in dist:
                continue
            dist[(nx, ny)]  = d + 1
            owner[(nx, ny)] = sid
            queue.append(((nx, ny), sid))

    mine  = sum(1 for sid in owner.values() if sid == my_id)
    total = len(owner)
    return mine, total


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def flood_fill(start_x, start_y, width, height, occupied, max_cells=200):
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
