from oshope.core.states.oshope_state import OSHopeState
from oshope.core.graphs.cognition.nodes.query_clarification import (
    query_clarification_node,
)
from oshope.core.graphs.cognition.nodes.query_classification import (
    query_classification_node,
)
from oshope.core.graphs.cognition.routing.logic import (
    route_query_after_starting,
)
from oshope.config.config import (
    QUERY_CLASSIFICATION_NODE,
    QUERY_CLARIFICATION_NODE,
)

from oshope.utils.logger import get_logger
from langgraph.graph import StateGraph, END, START


logger = get_logger(__name__)


class CognitionGraph(StateGraph):
    def __init__(self):
        self.graph = self._build_cognition_graph()
        self.compiled_graph = None

    def _build_cognition_graph(self) -> StateGraph[OSHopeState]:
        """
        Builds the cognition graph for the OS-HOPE, which includes nodes for query classification and clarification. The graph defines the flow of information and decision-making based on the user's query and the results of each node's processing.
        Returns:
            StateGraph[OSHopeState]: The constructed cognition graph.
        """
        graph = StateGraph(OSHopeState)

        graph.add_node(QUERY_CLASSIFICATION_NODE, query_classification_node)
        graph.add_node(QUERY_CLARIFICATION_NODE, query_clarification_node)

        graph.add_conditional_edges(
            START,
            route_query_after_starting,
            {
                QUERY_CLARIFICATION_NODE: QUERY_CLARIFICATION_NODE,
                QUERY_CLASSIFICATION_NODE: QUERY_CLASSIFICATION_NODE,
            },
        )

        graph.add_edge(QUERY_CLASSIFICATION_NODE, END)

        graph.add_edge(QUERY_CLARIFICATION_NODE, END)

        logger.info("Cognition graph built successfully.")

        return graph

    def compile(self) -> None:
        """
        Compiles the cognition graph.
        """
        self.compiled_graph = self.graph.compile()
        logger.info("Cognition graph compiled successfully.")

    def execute(self, initial_state: OSHopeState) -> OSHopeState:
        """
        Executes the compiled cognition graph starting from the given initial state.
        Args:
            initial_state (OSHopeState): The initial state to start the graph execution from.
        Returns:
            OSHopeState: The final state after executing the graph.
        """
        logger.info("Executing Cognition Graph.")
        final_state = self.compiled_graph.invoke(initial_state)
        final_state = OSHopeState(**final_state)
        logger.info("Cognition Graph execution completed.")
        return final_state
