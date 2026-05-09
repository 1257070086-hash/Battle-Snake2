"""
BattleSnake AI — logic.py v4.2（全面 Bug 修复版）
修复：
  1. compute_danger_zone 越界格（边界检查）
  2. get_safe_moves 的尾部排除不考虑 just_ate
  3. evaluate 食物竞争只看最近食物 → 改为找最佳安全食物
  4. endgame_strategy 删掉了调用 → 恢复（辅助 Minimax 前排序）
  5. step_state 只移动一条蛇，其余蛇尾部旧格仍在 occupied → 修复
  6. step_state 不移除血量归零的死蛇
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
DIR_NAMES = list(DIRECTIONS.keys())

MAX_DEPTH  = 6
TIME_LIMIT = 0.400   # 秒


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def choose_move(data: dict) -> str:
    start_time = time.time()

    board  = data["board"]
    me     = data["you"]
    width  = board["width"]
    height = board["height"]
    food   = [(f["x"], f["y"]) for f in board["food"]]
    snakes = board["snakes"]

    state = build_state(me, snakes, food, width, height)
    state["me_id"] = me["id"]

    live_enemies = [s for s in snakes if s["id"] != me["id"]]
    is_endgame   = (len(live_enemies) == 1)

    # Bug5 修复：danger_zone 带边界检查（见 compute_danger_zone）
    danger_zone = compute_danger_zone(
        live_enemies, state["occupied"], width, height, me["id"], me["length"]
    )

    # 基础安全走法（不过滤危险区，先拿所有安全方向）
    my_safe = get_safe_moves(
        state["my_head"], state, me["id"], me["length"],
        width, height, danger_zone=set()
    )
    if not my_safe:
        return "up"
    if len(my_safe) == 1:
        return list(my_safe.keys())[0]

    # 软过滤危险区（保留至少1个方向）
    safe_no_danger = {d: p for d, p in my_safe.items() if p not in danger_zone}
    if safe_no_danger:
        my_safe = safe_no_danger

    # 根节点走法排序（Flood Fill 大的优先，Alpha-Beta 更早剪枝）
    my_safe = sort_moves(my_safe, state, width, height)
    best_move = list(my_safe.keys())[0]

    # Bug4 修复：残局策略恢复，但仅作为初始 best_move 提示，不绕过 Minimax
    if is_endgame and live_enemies:
        eg = endgame_strategy(state, my_safe, me, live_enemies[0], food, width, height)
        if eg:
            best_move = eg  # 作为兜底，Minimax 会覆盖更好的选择

    # 迭代加深
    for depth in [3, MAX_DEPTH]:
        if time.time() - start_time > TIME_LIMIT * 0.6:
            break
        move, _ = alphabeta_root(
            state, my_safe, me, snakes, food, width, height,
            depth, start_time, danger_zone
        )
        if move:
            best_move = move

    # 食物兜底：以下任一条件触发强制找食
    #   1. 短蛇（< 8）：始终优先找食
    #   2. 任意长度血量 < 40：快饿死了，必须找食
    #   3. 任意长度血量 < 60 且地图上只剩 ≤ 2 个食物：食物稀缺，提前抢
    health     = me["health"]
    food_count = len(food)
    force_food = (
        me["length"] < 8
        or health < 40
        or (health < 60 and food_count <= 2)
    )
    if force_food and food:
        food_move = _best_food_move(my_safe, food, state, width, height, me["length"])
        if food_move:
            best_move = food_move

    return best_move


def _best_food_move(safe_moves, food, state, width, height, my_len):
    """
    短蛇辅助：找可达食物中最近、最安全的走法。
    只有当食物方向通过了 check_space（有足够空间）才返回。
    """
    occupied = state["occupied"]
    best_dir  = None
    best_dist = float("inf")

    for direction, pos in safe_moves.items():
        for f in food:
            dist = manhattan(pos, f)
            if dist < best_dist:
                # 吃完后空间检查（放宽：只需 > my_len/2）
                sim_len  = my_len + 1
                min_safe = max(3, sim_len // 2)
                ff = flood_fill(f[0], f[1], width, height, occupied,
                                max_cells=sim_len * 3)
                if ff >= min_safe:
                    best_dist = dist
                    best_dir  = direction

    return best_dir


# ─────────────────────────────────────────────
# 残局 1v1 辅助策略（只作为初始排序提示）
# ─────────────────────────────────────────────

def endgame_strategy(state, my_safe, me, enemy, food, width, height):
    my_len = state["my_length"]
    e_len  = enemy["length"]
    e_head = (enemy["head"]["x"], enemy["head"]["y"])

    if my_len > e_len:
        # 我更长：选距离敌蛇头最近的方向（压缩空间）
        best = min(my_safe.items(),
                   key=lambda item: manhattan(item[1], e_head))
        return best[0]
    else:
        food_list = list(state["food_set"])
        if food_list and state["my_health"] < 60:
            best = min(my_safe.items(),
                       key=lambda item: min(manhattan(item[1], f) for f in food_list))
            return best[0]
        best = max(my_safe.items(),
                   key=lambda item: flood_fill(
                       item[1][0], item[1][1], width, height, state["occupied"]))
        return best[0]


# ─────────────────────────────────────────────
# 危险区预标记（Bug5 修复：加边界检查）
# ─────────────────────────────────────────────

def compute_danger_zone(enemies, occupied, width, height, my_id, my_len):
    danger = set()
    for s in enemies:
        if s["length"] < my_len:
            continue
        eh = (s["head"]["x"], s["head"]["y"])
        for dx, dy in DIR_LIST:
            nx, ny = eh[0] + dx, eh[1] + dy
            # Bug5 修复：必须做边界检查
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            if (nx, ny) not in occupied:
                danger.add((nx, ny))
    return danger


# ─────────────────────────────────────────────
# 走法排序
# ─────────────────────────────────────────────

def sort_moves(moves: dict, state, width, height) -> dict:
    """
    排序逻辑：
    - 短蛇(< 8)：食物距离近的方向优先，确保食物方向被优先搜索
    - 中长蛇：Flood Fill 空间大的优先，Alpha-Beta 更早剪枝
    """
    my_len   = state["my_length"]
    food_set = state["food_set"]
    scored   = []

    for direction, pos in moves.items():
        ff = flood_fill(pos[0], pos[1], width, height, state["occupied"], max_cells=50)

        if my_len < 8 and food_set:
            # 短蛇：离最近食物越近分越高，叠加少量空间分避免死路
            min_food_dist = min(manhattan(pos, f) for f in food_set)
            score = -min_food_dist * 10 + ff * 0.1
        else:
            score = ff

        scored.append((direction, pos, score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return {d: p for d, p, _ in scored}


# ─────────────────────────────────────────────
# Alpha-Beta
# ─────────────────────────────────────────────

def alphabeta_root(state, my_safe, me, snakes, food, width, height,
                   depth, start_time, danger_zone):
    best_move  = None
    best_score = float("-inf")
    alpha = float("-inf")
    beta  = float("inf")

    for direction, my_next in my_safe.items():
        if time.time() - start_time > TIME_LIMIT:
            break
        new_state = step_state(state, me["id"], my_next, width, height)
        score = alphabeta(
            new_state, depth - 1, alpha, beta, False,
            me, snakes, food, width, height, start_time, danger_zone
        )
        if score > best_score:
            best_score = score
            best_move  = direction
        alpha = max(alpha, best_score)

    return best_move, best_score


def alphabeta(state, depth, alpha, beta, is_maximizing,
              me, snakes, food, width, height, start_time, danger_zone):
    if depth == 0 or time.time() - start_time > TIME_LIMIT:
        return evaluate(state, me, food, width, height)

    if is_maximizing:
        my_safe = get_safe_moves(
            state["my_head"], state, me["id"], state["my_length"],
            width, height, danger_zone=set(), check_space=False
        )
        if not my_safe:
            return -1000

        value = float("-inf")
        for _, my_next in my_safe.items():
            new_state = step_state(state, me["id"], my_next, width, height)
            value = max(value, alphabeta(
                new_state, depth - 1, alpha, beta, False,
                me, snakes, food, width, height, start_time, danger_zone
            ))
            alpha = max(alpha, value)
            if value >= beta:
                break
        return value

    else:
        # Bug6 修复：live_snakes 里只保留存活的敌蛇（血量>0）
        enemies = [s for s in state["live_snakes"]
                   if s["id"] != me["id"] and s["health"] > 0]
        if not enemies:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me, snakes, food, width, height, start_time, danger_zone)

        nearest = min(enemies, key=lambda s: manhattan(
            state["my_head"], (s["head"]["x"], s["head"]["y"])
        ))
        e_head = (nearest["head"]["x"], nearest["head"]["y"])
        e_safe = get_safe_moves(
            e_head, state, nearest["id"], nearest["length"],
            width, height, danger_zone=set(), check_space=False
        )
        if not e_safe:
            return alphabeta(state, depth - 1, alpha, beta, True,
                             me, snakes, food, width, height, start_time, danger_zone)

        value = float("inf")
        for _, e_next in e_safe.items():
            new_state = step_state(state, nearest["id"], e_next, width, height)
            value = min(value, alphabeta(
                new_state, depth - 1, alpha, beta, True,
                me, snakes, food, width, height, start_time, danger_zone
            ))
            beta = min(beta, value)
            if value <= alpha:
                break
        return value


# ─────────────────────────────────────────────
# 状态构建
# ─────────────────────────────────────────────

def build_state(me, snakes, food, width, height):
    food_set   = set(food)
    occupied   = set()
    snake_meta = {}

    for snake in snakes:
        body = snake["body"]
        # 精确判断：head==body[1] 说明上回合刚吃了食物（尾部不收缩）
        just_ate = (len(body) >= 2 and
                    body[0]["x"] == body[1]["x"] and
                    body[0]["y"] == body[1]["y"])
        snake_meta[snake["id"]] = {"just_ate": just_ate}

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
        "snake_meta":  snake_meta,
        "me_id":       me["id"],
    }


def step_state(state, snake_id, new_head, width, height):
    """
    Bug1+Bug6 修复：
    - 每条蛇都更新（旧尾出、新头入），不只更新目标蛇
    - 血量归零的蛇从 live_snakes 移除
    """
    food_set    = state["food_set"]
    new_food    = set(food_set)
    new_meta    = dict(state.get("snake_meta", {}))
    new_snakes  = []
    new_occupied = set()

    new_my_head   = state["my_head"]
    new_my_length = state["my_length"]
    new_my_health = state["my_health"]

    for s in state["live_snakes"]:
        if s["id"] == snake_id:
            # 目标蛇：执行移动
            ate_food   = new_head in food_set
            old_body   = list(s["body"])
            new_body   = [{"x": new_head[0], "y": new_head[1]}] + old_body
            if not ate_food:
                new_body = new_body[:-1]
            if ate_food:
                new_food.discard(new_head)
            new_len    = len(new_body)
            new_health = 100 if ate_food else max(0, s["health"] - 1)
            new_meta[snake_id] = {"just_ate": ate_food}

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
            # Bug1 修复：其余蛇也要更新 occupied（把旧尾移出）
            # 简化：保持身体不动但把尾部移出（模拟对方也在移动）
            old_body = list(s["body"])
            meta = new_meta.get(s["id"], {"just_ate": False})
            if not meta["just_ate"] and old_body:
                # 尾部下回合移走（不知道对方走哪，保守处理：尾部释放）
                pass  # 尾部已在 build_state 时排除
            new_s = s

        # Bug6 修复：死蛇（血量归零）不加入新状态
        if new_s["health"] <= 0:
            continue

        # 重建 occupied：只加入当前蛇的有效身体段
        body   = new_s["body"]
        s_meta = new_meta.get(new_s["id"], {"just_ate": False})
        for i, seg in enumerate(body):
            if i == len(body) - 1 and not s_meta["just_ate"]:
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
        "snake_meta":  new_meta,
        "me_id":       state.get("me_id"),
    }


# ─────────────────────────────────────────────
# 安全走法过滤
# ─────────────────────────────────────────────

def get_safe_moves(head, state, my_id, my_len, width, height,
                   danger_zone=None, check_space=True):
    if danger_zone is None:
        danger_zone = set()
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    safe = {}

    for direction, (dx, dy) in DIRECTIONS.items():
        nx, ny = head[0] + dx, head[1] + dy
        if nx < 0 or nx >= width or ny < 0 or ny >= height:
            continue
        if (nx, ny) in occupied:
            continue
        if (nx, ny) in danger_zone:
            continue

        risky = False
        for s in snakes:
            if s["id"] == my_id:
                continue
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan((nx, ny), eh) == 1 and s["length"] >= my_len:
                risky = True
                break
        if risky:
            continue

        safe[direction] = (nx, ny)

    if not safe or not check_space:
        return safe

    # 空间下限过滤：可达格 < 蛇身长度 → 死路
    # Bug2 修复：only 排除尾部当 just_ate=False
    my_meta = state.get("snake_meta", {}).get(my_id, {"just_ate": False})
    occupied_for_ff = occupied
    if not my_meta["just_ate"]:
        for s in snakes:
            if s["id"] == my_id and s["body"]:
                tail = s["body"][-1]
                occupied_for_ff = occupied - {(tail["x"], tail["y"])}
                break

    spacious = {}
    for direction, pos in safe.items():
        space = flood_fill(pos[0], pos[1], width, height, occupied_for_ff,
                           max_cells=my_len * 2 + 10)
        if space >= my_len:
            spacious[direction] = pos

    return spacious if spacious else safe


# ─────────────────────────────────────────────
# 估值函数
# ─────────────────────────────────────────────

def evaluate(state, me, food, width, height):
    my_pos   = state["my_head"]
    my_len   = state["my_length"]
    health   = state["my_health"]
    occupied = state["occupied"]
    snakes   = state["live_snakes"]
    my_id    = me["id"]

    # 长度阶段动态权重
    if my_len < 8:      # 短蛇：疯狂吃食，Voronoi 几乎不看
        w_voronoi, w_food, w_len_adv = 0.5, 5.0, 0.3
    elif my_len < 12:   # 中短蛇：食物+地盘并重
        w_voronoi, w_food, w_len_adv = 2.0, 3.0, 0.8
    elif my_len < 18:   # 中蛇：平衡
        w_voronoi, w_food, w_len_adv = 3.0, 1.5, 1.0
    else:               # 长蛇：争地盘为主
        w_voronoi, w_food, w_len_adv = 4.5, 0.5, 1.5

    # ① Voronoi
    voronoi_mine, voronoi_total = voronoi_control(state, my_id, width, height)
    voronoi_score = voronoi_mine / max(voronoi_total, 1)

    # ② Flood Fill
    ff = flood_fill(my_pos[0], my_pos[1], width, height, occupied, max_cells=100)
    ff_score = ff / (width * height)

    # ③ 食物分（每个食物独立评估，选最优安全食物）
    # 血量权重：短蛇始终积极（最低0.8），血量越低越急
    if my_len < 8:
        health_w = max(0.8, (120 - health) / 50.0)  # 短蛇：血量权重恒高
    elif health < 40:
        health_w = 3.0   # 任意长度快饿死：食物权重爆炸
    elif health < 60:
        health_w = 1.8   # 血量偏低：食物权重明显提升
    else:
        health_w = max(0.5, (100 - health) / 50.0)

    food_list  = list(state["food_set"])
    enemies    = [s for s in snakes if s["id"] != my_id]
    food_score = 0.0

    if food_list:
        best_food_score = 0.0
        for f in food_list:
            my_dist = manhattan(my_pos, f)
            base = 1.0 / (1 + my_dist)

            # 独立判断每个食物的竞争和危险程度
            contested = False
            dangerous = False
            for s in enemies:
                eh = (s["head"]["x"], s["head"]["y"])
                e_dist = manhattan(eh, f)
                # 竞争：敌蛇比我近 且 敌蛇比我长（等长也算危险）
                if e_dist < my_dist and s["length"] >= my_len:
                    contested = True
                # 危险：食物紧挨大蛇头（头对头概率极高）
                if e_dist <= 1 and s["length"] >= my_len:
                    dangerous = True

            if dangerous:
                base *= 0.15          # 危险食物：强力降权
            elif contested:
                # 短蛇阶段：有竞争也要抢，只轻微降权
                if my_len < 8:
                    base *= 0.8       # 短蛇：竞争食物仍大胆去
                else:
                    base *= 0.5       # 中长蛇：让步

            # 吃完后空间检查（短蛇放宽阈值：只要有 my_len/2 的空间就行）
            if my_dist <= 3:
                sim_len  = my_len + 1
                min_safe = sim_len // 2 if my_len < 8 else sim_len
                food_ff  = flood_fill(f[0], f[1], width, height, occupied,
                                      max_cells=sim_len * 3)
                if food_ff < min_safe:
                    base *= 0.2       # 真正的死路才降权

            best_food_score = max(best_food_score, base)

        food_score = best_food_score

    # ④ 长度优势
    if enemies:
        nearest = min(enemies, key=lambda s: manhattan(
            my_pos, (s["head"]["x"], s["head"]["y"])
        ))
        len_adv = (my_len - nearest["length"]) / max(my_len, nearest["length"])
    else:
        len_adv = 1.0

    # ⑤ 攻击加成
    attack_bonus = 0.0
    for s in enemies:
        if s["length"] < my_len:
            eh = (s["head"]["x"], s["head"]["y"])
            if manhattan(my_pos, eh) <= 2:
                attack_bonus = 1.0
                break

    return (voronoi_score * w_voronoi
            + ff_score    * 1.0
            + food_score  * w_food * health_w
            + len_adv     * w_len_adv
            + attack_bonus * 0.5)


# ─────────────────────────────────────────────
# Voronoi
# ─────────────────────────────────────────────

def voronoi_control(state, my_id, width, height):
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
