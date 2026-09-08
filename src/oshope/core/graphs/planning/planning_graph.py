from oshope.core.states.oshope_state import OSHopeState
from oshope.core.graphs.planning.nodes.planning import planning_node
from oshope.core.graphs.planning.nodes.user_validation import user_validation_node
from oshope.core.graphs.planning.routing.logic import (
    route_after_starting,
)
from oshope.config.config import (
    PLANNING_NODE,
    USER_VALIDATION_NODE,
)

from oshope.utils.logger import get_logger
from langgraph.graph import StateGraph, END, START

logger = get_logger(__name__)


class PlanningGraph(StateGraph):
    """
    The PlanningGraph is responsible for generating an execution plan based on the user's query and the information obtained from previous nodes (e.g., query classification and clarification). It also handles user validation of the generated plan before execution.
    """

    def __init__(self):
        self.graph = self._build_planning_graph()
        self.compiled_graph = None

    def _build_planning_graph(self) -> StateGraph[OSHopeState]:
        """
        Builds the planning graph for the OS-HOPE, which includes nodes for planning and user validation. The graph defines the flow of information and decision-making based on the user's query and the results of each node's processing.
        Returns:
            StateGraph[OSHopeState]: The constructed planning graph.
        """
        graph = StateGraph(OSHopeState)

        graph.add_node(PLANNING_NODE, planning_node)
        graph.add_node(USER_VALIDATION_NODE, user_validation_node)

        graph.add_conditional_edges(
            START,
            route_after_starting,
            {
                PLANNING_NODE: PLANNING_NODE,
                USER_VALIDATION_NODE: USER_VALIDATION_NODE,
            },
        )

        graph.add_edge(PLANNING_NODE, END)

        graph.add_edge(USER_VALIDATION_NODE, END)

        logger.info("Planning Graph built successfully.")

        return graph

    def compile(self) -> None:
        """
        Compiles the planning graph.
        """
        self.compiled_graph = self.graph.compile()
        logger.info("Planning Graph compiled successfully.")

    def execute(self, initial_state: OSHopeState) -> OSHopeState:
        """
        Executes the compiled planning graph starting from the given initial state.
        Args:
            initial_state (OSHopeState): The initial state to start the graph execution from.
        Returns:
            OSHopeState: The final state after executing the graph.
        """
        logger.info("Executing Planning Graph.")
        final_state = self.compiled_graph.invoke(initial_state)
        final_state = OSHopeState(**final_state)
        logger.info("Planning Graph execution completed.")
        return final_state
