// sokoban-solver-01: a push-optimal Sokoban solver for make.fun's generated levels.
//
// A generated level is a one-square-wide hallway across the board, where the player starts, with one to four
// rooms off it. A box pushed through a room's door into the hallway can never be pushed back (the hallway is
// one square wide, so nobody can stand on the far side of such a box), so every box stays in its room and the
// level's push optimum is the sum of its rooms' optima. This solver therefore:
//
//   1. cuts the player's row out of the floor and takes each remaining part that holds boxes as a room;
//   2. solves each room push-optimally with an A* search over pushes — a state is the room's boxes plus the
//      player's reachable region (named by its lowest square) — guided by a min-cost box-to-goal matching
//      over exact single-box push distances (pulled back from each goal over open floor), with dead-square
//      pruning (a box on a square no push can bring to any goal ends the line) and frozen 2x2 pruning;
//   3. joins the rooms' move strings with the player's walks between them.
//
// When a level is not a set of rooms, the same A* runs on the whole board instead. Every choice is fixed —
// direction order, tie-breaks, work caps counted in expanded states, never the clock — so two runs of this
// program on the same set write byte-identical answers.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The four directions: [dx, dy, the walk letter, the push letter]. Fixed order: ties always break the same way. */
const DIRS: [number, number, string, string][] = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** Work caps, counted in expanded states (never the clock, so runs are deterministic). */
const ROOM_CAP = 2_000_000;
const BOARD_CAP = 3_000_000;
const GREEDY_CAP = 2_000_000;

interface Level {
  w: number;
  h: number;
  wall: Uint8Array;
  goals: number[];
  boxes: number[];
  player: number;
}

/** A level in XSB text: width and height, walls (short rows and the outer frame padded as walls), goals, boxes, player. */
function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals: number[] = [];
  const boxes: number[] = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

/**
 * Exact single-box push distances to every goal, pulled back over open floor: dist[g * size + cell] is the
 * fewest pushes that bring one box from cell to goal g when no other box exists (the player is assumed to
 * reach any square a push needs; only walls block). Infinity where no push sequence reaches the goal.
 * A square with infinite distance to every goal is a dead square: a box there can never be solved.
 */
function pullDistances(w: number, wall: Uint8Array, goals: number[]): Int32Array {
  const size = wall.length;
  const dist = new Int32Array(goals.length * size).fill(-1);
  const queue = new Int32Array(size);
  goals.forEach((g, gi) => {
    const base = gi * size;
    let qh = 0;
    let qt = 0;
    queue[qt++] = g;
    dist[base + g] = 0;
    while (qh < qt) {
      const x = queue[qh++];
      const dx = dist[base + x] + 1;
      for (const [ddx, ddy] of DIRS) {
        const d = ddy * w + ddx;
        const y = x - d; // the box stood at y and was pushed along d onto x; the player stood at y - d
        if (y < 0 || y >= size) continue;
        if (!wall[y] && !wall[y - d] && dist[base + y] < 0) {
          dist[base + y] = dx;
          queue[qt++] = y;
        }
      }
    }
  });
  return dist;
}

/**
 * The A* estimate: the cheapest assignment of distinct goals to boxes, summed over the pull distances
 * (a lower bound on the pushes still needed, since one push moves one box one square). For many boxes the
 * assignment is relaxed to each box's nearest goal, still a lower bound. Infinity when some box reaches
 * no goal or no assignment avoids a dead square.
 */
function estimate(boxes: number[], dist: Int32Array, size: number, goalCount: number): number {
  const n = boxes.length;
  if (n === 0) return 0;
  if (n > 6 || goalCount > 8) {
    let sum = 0;
    for (const b of boxes) {
      let best = Infinity;
      for (let gi = 0; gi < goalCount; gi++) {
        const d = dist[gi * size + b];
        if (d >= 0 && d < best) best = d;
      }
      if (best === Infinity) return Infinity;
      sum += best;
    }
    return sum;
  }
  // Min-cost matching by permutation over goals (n <= 6 boxes, goalCount <= 8): assign boxes in order.
  const used = new Uint8Array(goalCount);
  let best = Infinity;
  const go = (bi: number, acc: number): void => {
    if (acc >= best) return;
    if (bi === n) {
      best = acc;
      return;
    }
    const b = boxes[bi];
    for (let gi = 0; gi < goalCount; gi++) {
      if (used[gi]) continue;
      const d = dist[gi * size + b];
      if (d < 0) continue;
      used[gi] = 1;
      go(bi + 1, acc + d);
      used[gi] = 0;
    }
  };
  go(0, 0);
  return best;
}

/** Whether the box just pushed to t sits in a frozen 2x2 block of walls and boxes with a box off its goal. */
function frozenBlock(w: number, wall: Uint8Array, occupied: Uint8Array, goal: Uint8Array, t: number): boolean {
  for (const a of [t, t - 1, t - w, t - w - 1]) {
    const b = a + 1;
    const c = a + w;
    const d = c + 1;
    if (!(wall[a] || occupied[a]) || !(wall[b] || occupied[b]) || !(wall[c] || occupied[c]) || !(wall[d] || occupied[d])) continue;
    if ((occupied[a] && !goal[a]) || (occupied[b] && !goal[b]) || (occupied[c] && !goal[c]) || (occupied[d] && !goal[d])) return true;
  }
  return false;
}

/** The shortest walk from `from` to `to` around walls and boxes, in lower LURD letters (first direction first on ties). */
function walkTo(w: number, wall: Uint8Array, occupied: Uint8Array, from: number, to: number): string {
  if (from === to) return '';
  const size = wall.length;
  const prev = new Int32Array(size).fill(-1);
  const how = new Array<string>(size);
  const queue = new Int32Array(size);
  let qh = 0;
  let qt = 0;
  queue[qt++] = from;
  prev[from] = from;
  while (qh < qt) {
    const p = queue[qh++];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (q < 0 || q >= size || wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue[qt++] = q;
    }
  }
  if (prev[to] === -1) return '';
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

interface Solved {
  pushes: number;
  moves: string;
}

/**
 * Solve one push problem: boxes to goals on a board whose walls are `wallMove` (real walls plus any boxes
 * standing outside the problem, which act as walls), with push distances measured on `wallReal` (real walls
 * only, so the estimate never overstates). mode 'astar' is push-optimal; mode 'greedy' takes the first
 * solution the estimate leads to, as a fallback when the optimal search would take too long.
 */
function solvePushes(
  w: number,
  wallMove: Uint8Array,
  wallReal: Uint8Array,
  goals: number[],
  startBoxes: number[],
  startPlayer: number,
  cap: number,
  mode: 'astar' | 'greedy',
): Solved | null {
  const size = wallMove.length;
  const goalSet = new Uint8Array(size);
  for (const g of goals) goalSet[g] = 1;
  const dist = pullDistances(w, wallReal, goals);
  const live = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    for (let gi = 0; gi < goals.length; gi++) {
      if (dist[gi * size + i] >= 0) {
        live[i] = 1;
        break;
      }
    }
  }
  if (startBoxes.some((b) => !live[b])) return null;
  const solved = (boxes: number[]): boolean => boxes.every((b) => goalSet[b] === 1);
  if (solved(startBoxes)) return { pushes: 0, moves: '' };
  const h0 = estimate(startBoxes, dist, size, goals.length);
  if (h0 === Infinity) return null;

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  /** Flood the player's region from `from` (boxes in the way); returns its lowest square, marks it in `arr`. */
  const flood = (arr: Int32Array, from: number): number => {
    stamp++;
    let low = from;
    let n = 0;
    stack[n++] = from;
    arr[from] = stamp;
    while (n > 0) {
      const p = stack[--n];
      if (p < low) low = p;
      const qs = [p - w, p + w, p - 1, p + 1];
      for (const q of qs) {
        if (q >= 0 && q < size && !wallMove[q] && !occupied[q] && arr[q] !== stamp) {
          arr[q] = stamp;
          stack[n++] = q;
        }
      }
    }
    return low;
  };
  const reach = (from: number): number => flood(seen, from); // the region of the state being expanded
  const reachNext = (from: number): number => flood(mark, from); // a successor's region, in the other array
  const keyOf = (boxes: number[], low: number): string => `${low}|${boxes.join(',')}`;

  // The states, in the order found: boxes, the player's square after the push that made the state, parentage.
  const sBoxes: number[][] = [startBoxes.slice()];
  const sPlayer: number[] = [startPlayer];
  const sParent: number[] = [-1];
  const sFrom: number[] = [-1]; // the pushed box's square before the push
  const sTo: number[] = [-1]; // ... and after it
  const bestG = new Map<string, number>();
  for (const b of startBoxes) occupied[b] = 1;
  bestG.set(keyOf(startBoxes, reach(startPlayer)), 0);
  for (const b of startBoxes) occupied[b] = 0;

  // A binary heap on (priority, estimate, insertion order): priority is g + h for A*, h alone for greedy.
  const heap: [number, number, number, number][] = [];
  let seq = 0;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0];
    const last = heap.pop() as [number, number, number, number];
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1], heap[c])) c++;
        if (!less(heap[c], heap[i])) break;
        [heap[i], heap[c]] = [heap[c], heap[i]];
        i = c;
      }
    }
    return top;
  };
  heapPush([mode === 'astar' ? h0 : h0, h0, seq++, 0]);

  let found = -1;
  let expanded = 0;
  while (heap.length > 0 && found < 0) {
    const [, , , s] = heapPop();
    const boxes = sBoxes[s];
    // Recompute this state's g from its depth in the parent chain only when needed: store g alongside.
    // (Depth equals pushes made, since every edge is one push.)
    let g = 0;
    for (let t = s; sParent[t] >= 0; t = sParent[t]) g++;
    const curKey = (() => {
      for (const b of boxes) occupied[b] = 1;
      const low = reach(sPlayer[s]);
      for (const b of boxes) occupied[b] = 0;
      return keyOf(boxes, low);
    })();
    if ((bestG.get(curKey) ?? Infinity) < g) continue; // a cheaper route to this state is already known
    if (solved(boxes)) {
      found = s;
      break;
    }
    if (++expanded > cap) return null;
    for (const b of boxes) occupied[b] = 1;
    reach(sPlayer[s]);
    const here = stamp;
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (const [dx, dy] of DIRS) {
        const d = dy * w + dx;
        const t = b + d;
        if (t < 0 || t >= size) continue;
        if (seen[b - d] !== here || wallMove[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const frozen = frozenBlock(w, wallMove, occupied, goalSet, t);
        occupied[t] = 0;
        occupied[b] = 1;
        if (frozen) continue;
        const next = boxes.slice();
        next[bi] = t;
        next.sort((a, b2) => a - b2);
        const h = estimate(next, dist, size, goals.length);
        if (h === Infinity) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        const low = reachNext(b); // the player ends the push on the box's old square
        occupied[t] = 0;
        occupied[b] = 1;
        const k = keyOf(next, low);
        if ((bestG.get(k) ?? Infinity) <= g + 1) continue;
        bestG.set(k, g + 1);
        const idx = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(s);
        sFrom.push(b);
        sTo.push(t);
        heapPush([mode === 'astar' ? g + 1 + h : h, h, seq++, idx]);
        if (mode === 'greedy' && solved(next)) {
          found = idx;
          break;
        }
      }
      if (found >= 0) break;
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the moves: walk to each push's stand square, then push, state by state from the start.
  const path: number[] = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = startPlayer;
  let boxes = startBoxes;
  for (const s of path) {
    const from = sFrom[s];
    const to = sTo[s];
    const d = to - from;
    const dir = DIRS.find(([dx, dy]) => dy * w + dx === d) as [number, number, string, string];
    for (const b of boxes) occupied[b] = 1;
    moves += walkTo(w, wallMove, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    moves += dir[3];
    player = from;
    boxes = sBoxes[s];
  }
  return { pushes: path.length, moves };
}

/** The square the player stands on after `moves` (LURD, case aside) from `from` on a board `w` wide. */
function endOf(w: number, from: number, moves: string): number {
  let p = from;
  for (const ch of moves.toLowerCase()) p += ch === 'u' ? -w : ch === 'd' ? w : ch === 'l' ? -1 : 1;
  return p;
}

interface Room {
  squares: number[];
  boxes: number[];
  goals: number[];
  done?: boolean;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway: a box there reaches no goal)
 * out of the floor, and each part left that holds a box is a room. Null when some part holds more or fewer
 * boxes than goals, or a box starts in the hallway: then the level is no set of rooms.
 */
function roomsOf(level: Level): Room[] | null {
  const { w, wall, goals, boxes, player } = level;
  const dist = pullDistances(w, wall, goals);
  const size = wall.length;
  const live = new Uint8Array(size);
  for (let i = 0; i < size; i++) {
    for (let gi = 0; gi < goals.length; gi++) {
      if (dist[gi * size + i] >= 0) {
        live[i] = 1;
        break;
      }
    }
  }
  const row = Math.floor(player / w);
  const cut = new Uint8Array(size);
  for (let i = row * w; i < (row + 1) * w; i++) if (!wall[i] && !live[i]) cut[i] = 1;
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(size).fill(-1);
  const rooms: Room[] = [];
  for (let s = 0; s < size; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (q >= 0 && q < size && !wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0) || rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Solve one level: room by room when it is a set of rooms, else the whole board at once. */
function solveLevel(text: string): string {
  const level = parseLevel(text);
  const { w, wall, goals, boxes, player } = level;
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushes(w, wall, wall, goals, boxes, player, BOARD_CAP, 'astar')
      ?? solvePushes(w, wall, wall, goals, boxes, player, GREEDY_CAP, 'greedy');
    return whole === null ? '' : whole.moves;
  }
  let at = player;
  let moves = '';
  for (const room of rooms) {
    const wallMove = wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      for (const b of other.done ? other.goals : other.boxes) wallMove[b] = 1;
    }
    const sub = solvePushes(w, wallMove, wall, room.goals, room.boxes.slice().sort((a, b) => a - b), at, ROOM_CAP, 'astar')
      ?? solvePushes(w, wallMove, wall, room.goals, room.boxes.slice().sort((a, b) => a - b), at, GREEDY_CAP, 'greedy');
    if (sub === null) return '';
    moves += sub.moves;
    at = endOf(w, at, sub.moves);
    room.done = true;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
