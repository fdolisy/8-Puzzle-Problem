# 8-Puzzle-Problem
## Instructions to run
1. Ensure that your machine is running a version of python3
   * you can check in your machine by running `<your python command> --version`, where your python command is how you run python eg. `<py, python, python3>`
   * If not, install python3 [here](https://www.python.org/downloads/)
2. Ensure numpy is installed, if not install numpy
   * `pip install numpy`
   * Further instructions can be found [here](https://numpy.org/install/)
3. Run the code with the following line
   * `<your python command> Assignment1.py <algorithm_name> <input_file_path>`
   * where `<algorithm_name>` is one of the following `[dfs,ids,astar1,astar2]`
   * where `<input_file_path>` is the source file  containing the space separated input state
     * 6 7 1 8 2 * 5 4 3 
  ## Sample input with coressponding output
  * input file containing - 
```6 7 1 8 2 * 5 4 3```
  * output
```
6 7 1
8 2 *
5 4 3

6 7 1
8 * 2
5 4 3

6 7 1
* 8 2
5 4 3

* 7 1
6 8 2
5 4 3

7 * 1
6 8 2
5 4 3

7 8 1
6 * 2
5 4 3


Number of moves =  5
Number of states enqueued =  16
```
## Heuristic analysis
The Manhattan distance and the Misplaced Tiles heuristic are two common heuristic methods in the 8-Puzzle problem.

The Manhattan distance calculates the total number of blocks that a tile needs to be moved in the vertical and horizontal direction to reach its goal position. The Manhattan distance heuristic is considered admissible because it never overestimates the actual cost of reaching the goal state.

The Misplaced Tiles heuristic, on the other hand, calculates the number of tiles that are in the incorrect position in the current state compared to the goal state. This heuristic is also admissible because it provides an estimate of the minimum number of moves required to reach the goal state.

In terms of efficiency, the Manhattan distance heuristic is often considered to be more effective in practice than the Misplaced Tiles heuristic. This is because it takes into account both the horizontal and vertical distances between the tiles, which results in a more accurate estimate of the actual cost of reaching the goal state.

However both remain as good heusitics because they are both admissible.
