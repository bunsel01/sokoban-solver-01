# Sokoban solver — make.fun sokoban-solver-01

A push-optimal Sokoban solver for make.fun's generated levels, written for the
sokoban-solver-01 bounty ("Build the best Sokoban solver"). One program,
`main.ts` at the repository root (Node 22), answers every instance of a set in
one run: it reads `/input/<id>/level.txt` (or `$MAKEFUN_INPUT_DIR`) and writes
`/output/<id>/solution.txt` (or `$MAKEFUN_OUTPUT_DIR`) with the player's moves
in LURD.

## How it works

A generated level is a one-square-wide hallway across the board, where the
player starts, with one to four rooms off it. A box pushed through a room's
door into the hallway can never be pushed back — the hallway is one square
wide, so nobody can stand on the far side of such a box — so every box stays
in its room and the level's push optimum is the sum of its rooms' optima.

The solver follows that structure:

1. **Rooms.** Cut the dead squares of the player's row (the hallway) out of
   the floor; each remaining part that holds boxes is a room.
2. **Per room, push-optimal A\*.** A state is the room's boxes plus the
   player's reachable region, named by its lowest square. The estimate is a
   min-cost box-to-goal matching over exact single-box push distances (pulled
   back from each goal over open floor), which also marks the dead squares:
   a box on a square no push can bring to any goal ends the line. Frozen 2x2
   blocks of walls and boxes with a box off its goal are pruned too.
3. **Join.** The rooms' move strings are concatenated; the player simply
   walks from where one room left it to the next room's pushes.

When a level is not a set of rooms, the same A* runs on the whole board, with
a greedy best-first pass as a fallback if the optimal search would take too
long. Every choice is fixed — direction order, tie-breaks, and work caps
counted in expanded states, never the clock — so two runs on the same set
write byte-identical answers.

## Local run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<dir with i001..i005/level.txt> MAKEFUN_OUTPUT_DIR=out node main.ts
```
