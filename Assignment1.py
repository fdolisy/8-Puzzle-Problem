"""
 >>> Author: Fanny Dolisy
 >>> Title: The 8 puzzle problem
"""
import numpy as np
import os
import re
import sys

num_states_enqueued = 1
# Test!!!
class Board_Node():
    """
    Class that represents the puzzle board node in the state tree
    """
    def __init__(self, state, parent, depth, h_score=0, g_score=0, f_score=0) :
        """
        Constructor
        Args: 
            state: the current 2d array state of the puzzle
            parent: the parent node
            depth: the depth of this node in the tree
            h_score: the score from the heuristic functions (h(n))
            g_score: the actual cost (g(n))
            f_score: h(n) + g(n), used by A*
        """
        self.state          = state             
        self.parent         = parent            
        self.depth          = depth            
        
        self.h_score        = h_score           # h(n)
        self.g_score        = g_score           # g(n)
        self.f_score        = f_score           # f(n)
        
        # --- children node
        self.upper_neighbor = None              # state when swapping blank with tile above it
        self.left_neighbor  = None              # state when swapping blank with tile left of it
        self.lower_neighbor = None              # state when swapping blank with tile below it
        self.right_neighbor = None              # state when swapping blank with tile right of it

    
    def print_results(self) -> None:
        """
        Prints the path from the input to the goal
        """
        global num_states_enqueued
        state_stack = [self.state]
        
        # number of moves is equal to the depth
        num_moves = self.depth

        # add to a state stack
        while self.parent:
            self = self.parent
            state_stack.append(self.state)

        # print the path of states
        while state_stack:
            array = state_stack.pop()
            str_array = re.sub('0', '*', str(array))
            print(re.sub('( \[|\[|\])', '', str_array))
            print()
        print("\nNumber of moves = ", num_moves)
        print("Number of states enqueued = ", num_states_enqueued)
        exit()

    
    def attempt_swap(self, zero_row: int, zero_col: int, row_adjust: int, col_adjust: int) -> tuple[bool, np.ndarray]:
        """
        Attempts to see if swapping the blank in a desired direction is possible, if it is, the swap occurs
        Args: 
            zero_row: what row the zero cant be on for a swap to occur
            zero_col: what col the zero cant be on for a swap to occur
            row_adjust: which way to shift row
            col_adjust: which was to shift col
        Returns
            bool indicating if swap has occured
            the newly swapped state
        """
        global num_states_enqueued
        # find the current pos of the blank
        zero_index=[i[0] for i in np.where(self.state==0)] 
        # check if swap is possible
        if zero_index[zero_row] == zero_col: 
            return False, None
        else:
            # swap
            swap_value = self.state[zero_index[0]+row_adjust, zero_index[1]+col_adjust]
            new_state = self.state.copy()
            new_state[zero_index[0], zero_index[1]] = swap_value
            new_state[zero_index[0] + row_adjust, zero_index[1] + col_adjust] = 0
            num_states_enqueued += 1
            return True, new_state


def dfs(root_node: Board_Node, goal_state: np.ndarray, depth_limit: int) -> None:
    """
    Implements dfs with a limit of 10,
    if the goal is not found anywhere on or before depth 10, a failure message is printed

    Args:
        root_node: the inputted board
        goal_state: the board we are searching for
        depth_limit: the maximum depth we can search on
    """
    frontier_nodes= [root_node]    
    # use a set to ensure no dupes  
    visited = set([]) 
    
    while frontier_nodes:
        current_node = frontier_nodes.pop(0)  
        visited.add(current_node) 

        # if we found the goal
        if np.array_equal(current_node.state.reshape(1,9)[0], goal_state.reshape(1,9)[0]):
            current_node.print_results()

        neighbor_depth = current_node.depth + 1

        # see if we can move blank left
        left_movement_valid, new_state = current_node.attempt_swap(1,0,0,-1)
        if left_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in visited) 
                    and neighbor_depth <= depth_limit):
                current_node.left_neighbor = Board_Node(state=new_state,
                                            parent=current_node,
                                            depth=neighbor_depth)

                frontier_nodes.insert(0,current_node.left_neighbor)

        # see if we can move blank up
        up_movement_valid, new_state = current_node.attempt_swap(0,0,-1,0)
        if up_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in visited) 
                    and neighbor_depth <= depth_limit):
                current_node.upper_neighbor = Board_Node(state=new_state,
                                            parent=current_node,
                                            depth=neighbor_depth)

                frontier_nodes.insert(0,current_node.upper_neighbor)

        # see if we can move blank right
        right_movement_valid, new_state = current_node.attempt_swap(1,2,0,1)
        if right_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in visited) 
                    and neighbor_depth <= depth_limit):
                current_node.right_neighbor = Board_Node(state=new_state,
                                                parent=current_node,
                                                depth=neighbor_depth)

                frontier_nodes.insert(0,current_node.right_neighbor)

        # see if we can move blank down
        down_movement_valid, new_state = current_node.attempt_swap(0,2,1,0)
        if down_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in visited) 
                    and neighbor_depth <= depth_limit):
                current_node.lower_neighbor = Board_Node(state=new_state,
                                            parent=current_node,
                                            depth=neighbor_depth)

                frontier_nodes.insert(0,current_node.lower_neighbor)
            
    print('goal state was not found before or at depth 10.')
    exit()


def ids(root_node: Board_Node, goal_state: np.ndarray) -> None:
    """
    Using DFS, simply iterate through [0,10] as the depth limit to search using ids
    Args:
        root_node: the inputted board
        goal_state: the board we are searching for
    """
    for i in range(11):
        dfs(root_node, goal_state, i)

def astar(root_node: Board_Node, goal_state: np.ndarray, h_function: str) -> None:
    """
    Similar to dfs, except with a priority queue that is organized according to the lowest f(n)
    Args:
        root_node: the inputted board
        goal_state: the board we are searching for
        h_function: which heuristic function the user wants to use
    """
    visited = set([]) 
    depth_limit = 10
    priority_queue = [root_node]
    while priority_queue:
        # sort the priority queue
        priority_queue = sorted(priority_queue, key=lambda x:x.f_score)
        current_node = priority_queue.pop(0)
        
        # goal is found
        if np.array_equal(current_node.state.reshape(1,9)[0], goal_state.reshape(1,9)[0]):
                current_node.print_results()
        
        visited.add(current_node)
        
        neighbor_depth = current_node.depth + 1
        # gscore in this case is equivalent to the depth
        g_score = current_node.g_score + 1

        # check if blank can move up
        up_movement_valid, new_state = current_node.attempt_swap(0,0,-1,0)
        if up_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in visited) 
                    and neighbor_depth <= depth_limit):
                h_score = get_h_cost(new_state, goal_state, h_function)
                f_score = g_score + h_score
                current_node.upper_neighbor = Board_Node(state=new_state,
                                    parent=current_node,
                                    depth=neighbor_depth,
                                    h_score=h_score,
                                    g_score=g_score,
                                    f_score=f_score)
                priority_queue.append(current_node.upper_neighbor)
        
        # check if blank can move down
        down_movement_valid, new_state = current_node.attempt_swap(0,2,1,0)
        if down_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in visited) 
                    and neighbor_depth <= depth_limit):
                h_score = get_h_cost(new_state, goal_state, h_function)
                f_score = g_score + h_score
                current_node.lower_neighbor = Board_Node(state=new_state,
                                    parent=current_node,
                                    depth=neighbor_depth,
                                    h_score=h_score,
                                    g_score=g_score,
                                    f_score=f_score)
                priority_queue.append(current_node.lower_neighbor)
        
        # check if blank can move right
        right_movement_valid, new_state = current_node.attempt_swap(1,2,0,1)
        if right_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in visited) 
                    and neighbor_depth <= depth_limit):
                h_score = get_h_cost(new_state, goal_state, h_function)
                f_score = g_score + h_score
                current_node.right_neighbor = Board_Node(state=new_state,
                                    parent=current_node,
                                    depth=neighbor_depth,
                                    h_score=h_score,
                                    g_score=g_score,
                                    f_score=f_score)
                priority_queue.append(current_node.right_neighbor)
        
        # check if blank can move left
        left_movement_valid, new_state = current_node.attempt_swap(1,0,0,-1)
        if left_movement_valid:
            if (not any(np.array_equal(x.state.reshape(1,9)[0],new_state.reshape(1,9)[0]) for x in priority_queue) 
                    and neighbor_depth <= depth_limit):
                h_score = get_h_cost(new_state, goal_state, h_function)
                f_score = g_score + h_score
                current_node.left_neighbor = Board_Node(state=new_state,
                                    parent=current_node,
                                    depth=neighbor_depth,
                                    h_score=h_score,
                                    g_score=g_score,
                                    f_score=f_score)
                priority_queue.append(current_node.left_neighbor)

    print('goal state was not found before or at depth 10.')
    exit()

def get_h_cost(new_state: np.ndarray, goal_state:np.ndarray, h_function: str) -> int:
    """
    Maps the desired algorithm to the appropriate function call
    Args
        new_state: the state that needs to get an hcost
        goal_state: the goal we are looking for
        h_function: the heuristic method we want to use
    Return
        the heuristic cost
    """
    if h_function == 'num_misplaced':
            return h_misplaced_cost(new_state, goal_state)
    elif h_function == 'manhattan':
         return h_manhattan_cost(new_state)

def h_manhattan_cost(new_state: np.ndarray) -> int:
    """
    Calculates the manhattan distance of the inputted state
    Args:
        new_state: the state that needs to be compared to the goal
    Return
        the manhattan cost of the new state
    """
    goal_positions = {7:(0,0), 8:(0,1), 1:(0,2), 6:(1,0), 0:(1,1), 2:(1,2), 5:(2,0), 4:(2,1),3:(2,2)} 
    sum_manhattan = 0
    for i in range(3):
        for j in range(3):
            if new_state[i,j] != 0:
                sum_manhattan += sum(abs(a-b) for a,b in zip((i,j), goal_positions[new_state[i,j]]))
    return sum_manhattan


def h_misplaced_cost(new_state: np.ndarray, goal_state: np.ndarray) -> int:
    """
    Calculates the misplaced cost of the inputted state
    Args:
        new_state: the state that needs to be compared to the goal
        goal_state: the goal we are looking for
    Return
        the misplaced cost of the new state
    """
    cost = np.sum(new_state != goal_state)
    # sont include the blank in the cost
    cost = cost - 1
    if cost > 0: 
        return cost
    else: 
        return 0

def get_data_filepath(filepath: str) -> str:
    """
    Gets the cross platform filepath of the data file
    Args:
        filepath: the inputted file path
    Returns: 
        the platform independent filepath
    """
    current_dir = os.getcwd()
    return os.path.join(current_dir, filepath)

# driver
def main(argv):

    if len(sys.argv) == 3:        
        algorithm_name  = sys.argv[1]
        input_file_path = sys.argv[2]   
    else:
        print(f'ERROR: Pass 2 Arguments: py Assignment.py <algorithm_name> <input_file_path>')
        quit()
    try:
        file = open(get_data_filepath(input_file_path), 'r')
        read_line = file.readline()
            
        read_line = re.sub('\*', '0', str(read_line))
        cmd_init_state = [int(i) for i in read_line.split(' ')]
        init_state = np.array(cmd_init_state).reshape(3,3)
        file.close()   
    except FileNotFoundError:
        print(f"ERROR: File located at {input_file_path} was not found, please run again with a valid file")
        exit()

    goal_state = np.array([7,8,1,6,0,2,5,4,3]).reshape(3,3)
    
    # create the root node
    root_node = Board_Node(state=init_state,
                    parent=None,
                    depth=0)

    if(algorithm_name == 'dfs'):
        dfs(root_node, goal_state, 10)   
    elif(algorithm_name == 'ids'):
        ids(root_node, goal_state)   
    elif(algorithm_name == 'astar1'):
        astar(root_node, goal_state, h_function = 'num_misplaced')    
    elif(algorithm_name == 'astar2'):
        astar(root_node, goal_state, h_function = 'manhattan') 
    else:
        print('available <algorithm_name> are [dfs, ids, astar1, astar2]\n')
    
if __name__ == '__main__':
    main(sys.argv)