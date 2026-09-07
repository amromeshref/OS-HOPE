from oshope.core.states.oshope_state import OSHopeState
from oshope.core.graphs.memory.nodes.summarizer import summarizer_node
from oshope.config.config import SUMMARIZER_NODE
from oshope.utils.logger import get_logger
from langgraph.graph import StateGraph, END, START

logger = get_logger(__name__)

class MemoryGraph:
    def __init__(self):
        self.graph = self._build_memory_graph()
        self.compiled_graph = None
    
    def _build_memory_graph(self) -> StateGraph[OSHopeState]:
        """
        Build the memory graph with nodes and conditional edges based on the state of the OS-HOPE.
        """
        graph = StateGraph(OSHopeState)
        
        graph.add_node(SUMMARIZER_NODE, summarizer_node)

        # For now, we only have one node in the memory graph, so we can just route from START to it and then to END.
        graph.add_edge(START, SUMMARIZER_NODE)
        graph.add_edge(SUMMARIZER_NODE, END)

        logger.info("Memory graph built successfully.")

        return graph
    
    def compile(self):
        """
        Compile the graph for execution. This should be called after building the graph and before running it.
        """
        self.compiled_graph = self.graph.compile()

        logger.info("Memory graph compiled successfully.")

    def execute(self, initial_state: OSHopeState) -> OSHopeState:
        """
        Execute the compiled graph starting from the initial state and return the final state after execution.
        """
        logger.info("Starting execution of the memory graph.")

        final_state = self.compiled_graph.invoke(initial_state)
        final_state = OSHopeState(**final_state)
        logger.info("Completed execution of the memory graph.")
        
        return final_state