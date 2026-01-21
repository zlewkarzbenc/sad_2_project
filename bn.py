import networkx as nx
import boolean as bool
import random
import csv
import os
from typing import List, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

class BN():
    """
    Boolean Network (BN) class supporting synchronous and asynchronous dynamics,
    trajectory simulation, and dataset generation for BNFinder2 / DBN learning.
    """

    __bool_algebra = bool.BooleanAlgebra()

    def __init__(self, list_of_nodes: List[str], list_of_functions: List[str]):
        """
        Initialize a Boolean network.

        Args:
            list_of_nodes: List of node names (e.g., ['x0', 'x1', 'x2']).
            list_of_functions: Boolean update functions as strings, one per node.
        """
        self.num_nodes = len(list_of_nodes)
        self.node_names = list_of_nodes

        # Boolean symbols for nodes
        self.list_of_nodes = [self.__bool_algebra.Symbol(n) for n in list_of_nodes]

        # Parsed Boolean functions
        self.functions = [self.__bool_algebra.parse(f, simplify=True)
                          for f in list_of_functions]

    # -----------------------------
    # State helpers
    # -----------------------------
    def _int_to_state(self, x: int) -> Tuple[int, ...]:
        """
        Convert an integer to a Boolean state tuple.

        Example: x=3, num_nodes=3 -> (0,1,1)
        """
        binary_str = format(x, '0' + str(self.num_nodes) + 'b')
        return tuple(int(c) for c in binary_str)

    @staticmethod
    def state_to_binary_str(state: Tuple[int, ...]) -> str:
        """
        Convert a state tuple to a binary string representation.
        """
        return ''.join(str(b) for b in state)

    # -----------------------------
    # Dynamics
    # -----------------------------
    def get_neighbor_states(self, state: Tuple[int, ...]) -> set:
        """
        Compute all states reachable from the given state
        in one asynchronous update step.
        """
        true = self.__bool_algebra.TRUE
        false = self.__bool_algebra.FALSE

        # Map Boolean variables to TRUE/FALSE
        state_dict = {node: (true if v == 1 else false)
                      for node, v in zip(self.list_of_nodes, state)}

        reachable_states = set()

        # Update exactly one node at a time
        for i in range(self.num_nodes):
            new_value = self.functions[i].subs(state_dict).simplify()
            new_value = 1 if new_value == true else 0

            # Only record changes
            if new_value != state[i]:
                new_state = list(state)
                new_state[i] = new_value
                reachable_states.add(tuple(new_state))

        return reachable_states

    def synchronous_step(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Perform one synchronous update (all nodes updated simultaneously).
        """
        true = self.__bool_algebra.TRUE
        false = self.__bool_algebra.FALSE

        # dict used to transition from symbolic boolean expressions to Python booleans
        state_dict = {node: (true if v == 1 else false)
                      for node, v in zip(self.list_of_nodes, state)}

        new_state = []
        for i in range(self.num_nodes):
            # substitue symbols by their values and evaluate (simplify) the logical expression
            val = self.functions[i].subs(state_dict).simplify()
            new_state.append(1 if val == true else 0)

        return tuple(new_state)

    def asynchronous_step(self, state: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Perform one asynchronous update by randomly selecting
        one enabled node update.
        """
        neighbors = list(self.get_neighbor_states(state))

        # If no updates are possible, remain in the same state
        if not neighbors:
            return state

        return random.choice(neighbors)

    def simulate_trajectory(self,
                 initial_state: Tuple[int, ...],
                 steps: int,
                 mode: str = 'async') -> List[Tuple[int, ...]]:
        """
        Simulate a trajectory of the Boolean network in synchronous or asynchronous setting.

        Args:
            initial_state: Starting state of the network.
            steps: Number of update steps.
            mode: 'sync' or 'async'.

        Returns:
            List of visited states (including the initial state).
        """
        traj = [initial_state]
        current = initial_state

        for _ in range(steps):
            if mode == 'sync':
                current = self.synchronous_step(current)
            else:
                current = self.asynchronous_step(current)
            traj.append(current)

        return traj

    # -----------------------------
    # Dataset utilities
    # -----------------------------
    @staticmethod
    def sample_trajectory(traj: List[Tuple[int, ...]], freq: int) -> List[Tuple[int, ...]]:
        """
        Down-sample a trajectory by keeping every `freq`-th state.
        """
        return traj[::freq]
    

    #-----------------------------
    # Functions for drawing
    #----------------------------

    def generate_state_transition_system(self) -> nx.DiGraph:
        """
        Generates the asynchronous state transition system of the Boolean network.

            Returns:
                nx.DiGraph: NetworkX DiGraph object representing the asynchronous state transition system.
        """
        
        G = nx.DiGraph()

        num_states = 2 ** self.num_nodes
        for i in range(num_states):
            current_state = self._int_to_state(i)
            G.add_node(current_state)

            neighbor_states = self.get_neighbor_states(current_state)
            for neighbor in neighbor_states:
                G.add_edge(current_state, neighbor)

        return G


    def get_attractors(self) -> list[set[tuple[int]]]:
        
        """
        Computes the asynchronous attractors of the Boolean network.

            Returns:
                list[set[tuple[int]]]: A list of asynchronous attractors. Each attractor is a set of states.
        """

        sts = self.generate_state_transition_system()

        attractors = []
        for attractor in nx.attracting_components(sts):
            attractors.append(attractor)

        return attractors
    


    def draw_state_transition_system(self, highlight_attractors: bool = True) -> None:
        """
        Draws the state transition system.

            Args:
                highlight_attractors: If True, states belonging to different attractors are drawn 
                    using distinct colors.

            Returns:
                None
        """

        # The color used for non-attractor states in the state transition system
        NON_ATTRACTOR_STATE_COLOR = 'grey'

        sts = self.generate_state_transition_system()

        if highlight_attractors:
            attractors = self.get_attractors()

            sts_nodes = list(sts.nodes)

            node_colors = [NON_ATTRACTOR_STATE_COLOR for node in sts_nodes]

            colors = list(mcolors.CSS4_COLORS)
            colors.remove('white')
            colors.remove(NON_ATTRACTOR_STATE_COLOR)
            
            for attractor in attractors:
                # Select a random color for coloring the states of the attractor
                color = random.choice(colors)
                for state in attractor:
                    node_colors[sts_nodes.index(state)] = color

        nx.draw_networkx(sts,
                         with_labels=True,
                         pos=nx.spring_layout(sts, seed=42),
                         node_color = node_colors,
                         font_size=8)

        plt.show()

    def to_nx_graph(self) -> nx.DiGraph:
        """
        Convert the Boolean network to a NetworkX DiGraph.
        Nodes: self.node_names
        Edges: parent -> child if parent appears in child's Boolean function
        """
        G = nx.DiGraph()
        G.add_nodes_from(self.node_names)

        for idx, fun in enumerate(self.functions):
            child = self.node_names[idx]
            # fun.symbols gives a set of Boolean Algebra symbols used in this function
            parents = [str(s) for s in fun.symbols]  
            for p in parents:
                G.add_edge(p, child)
        return G


# Random Boolean network generator

def random_boolean_function(target: str, parents: List[str]) -> str:
    """
    Generate a random Boolean function in Disjunctive Normal Form (DNF)
    using the given parent variables.
    """
    # Constant function if no parents exist
    if not parents:
        return random.choice(['TRUE', 'FALSE'])

    terms = []

    # Generate a small number of conjunctions
    for _ in range(random.randint(1, min(3, 2 ** len(parents)))):
        lits = []
        for p in parents:
            # Randomly negate literals
            if random.random() < 0.5:
                lits.append(p)
            else:
                lits.append('~' + p)
        terms.append(' & '.join(lits))

    # Disjunction of conjunctions
    return ' | '.join(f'({t})' for t in terms)


def generate_random_bn(n: int, max_parents: int = 3) -> BN:
    """
    Generate a random Boolean network with bounded indegree.

    Args:
        n (int): Number of nodes.
        max_parents (int): Maximum number of parents per node.

    Returns:
        A randomly generated BN instance.
    """
    nodes = [f'x{i}' for i in range(n)]
    functions = []

    for i, node in enumerate(nodes):
        # Potential parents exclude the node itself
        possible_parents = nodes[:i] + nodes[i+1:]
        k = random.randint(0, min(max_parents, len(possible_parents)))
        parents = random.sample(possible_parents, k)
        functions.append(random_boolean_function(node, parents))

    return BN(nodes, functions)


def load_bnet_to_BN(file_path: str) -> BN:
    """
    Parses a .bnet file and returns a BN object with functions converted to DNF.
    """
    nodes = []
    functions = []
    
    algebra = bool.BooleanAlgebra()
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines and comments or header
            if not line or line.startswith('#') or line.lower().strip() == 'targets,factors':
                continue
                
            target, expression = [part.strip() for part in line.split(',', 1)]
            
            # Parse the expression
            parsed_expr = algebra.parse(expression)
            
            # Convert to DNF
            dnf_expr = algebra.dnf(parsed_expr)
            
            nodes.append(target)
            functions.append(str(dnf_expr))
            
    return BN(nodes, functions)


def main():
    print("Module with Boolean Network class implementation and helper functions (to be used via import).")

if __name__ == "__main__":
    main()