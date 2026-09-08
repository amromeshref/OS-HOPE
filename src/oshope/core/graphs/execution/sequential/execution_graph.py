from oshope.core.states.oshope_state import OSHopeState
from oshope.core.graphs.execution.sequential.nodes.code_execution import (
    code_execution_node,
)
from oshope.core.graphs.execution.sequential.nodes.information_generation import (
    information_generation_node,
)
from oshope.core.graphs.execution.sequential.nodes.final_response import (
    final_response_node,
)
from oshope.core.graphs.execution.sequential.nodes.code_error_handling import (
    code_error_handling_node,
)
from oshope.core.graphs.execution.sequential.nodes.step_resolver import (
    step_resolver_node,
)
from oshope.core.graphs.execution.sequential.routing.logic import (
    route_after_starting,
    route_after_code_execution,
    router,
)
from oshope.config.config import (
    CODE_EXECUTION_NODE,
    INFORMATION_NODE,
    FINAL_RESPONSE_NODE,
    STEP_RESOLVER_NODE,
    CODE_ERROR_HANDLING_NODE,
)

from oshope.utils.logger import get_logger
from langgraph.graph import StateGraph, END, START


logger = get_logger(__name__)


class SequentialExecutionGraph:
    def __init__(self):
        self.graph = self._build_execution_graph()
        self.compiled_graph = None

    def _build_execution_graph(self) -> StateGraph[OSHopeState]:
        """
        Build the execution graph with nodes and conditional edges based on the state of the OS-HOPE.
        """
        graph = StateGraph(OSHopeState)

        graph.add_node(CODE_EXECUTION_NODE, code_execution_node)
        graph.add_node(INFORMATION_NODE, information_generation_node)
        graph.add_node(FINAL_RESPONSE_NODE, final_response_node)
        graph.add_node(STEP_RESOLVER_NODE, step_resolver_node)
        graph.add_node(CODE_ERROR_HANDLING_NODE, code_error_handling_node)

        graph.add_conditional_edges(
            START,
            route_after_starting,
            {
                CODE_EXECUTION_NODE: CODE_EXECUTION_NODE,
                INFORMATION_NODE: INFORMATION_NODE,
                FINAL_RESPONSE_NODE: FINAL_RESPONSE_NODE,
            },
        )

        graph.add_conditional_edges(
            CODE_EXECUTION_NODE,
            route_after_code_execution,
            {
                CODE_EXECUTION_NODE: CODE_EXECUTION_NODE,
                INFORMATION_NODE: INFORMATION_NODE,
                FINAL_RESPONSE_NODE: FINAL_RESPONSE_NODE,
                STEP_RESOLVER_NODE: STEP_RESOLVER_NODE,
                CODE_ERROR_HANDLING_NODE: CODE_ERROR_HANDLING_NODE,
            },
        )

        graph.add_conditional_edges(
            INFORMATION_NODE,
            router,
            {
                CODE_EXECUTION_NODE: CODE_EXECUTION_NODE,
                INFORMATION_NODE: INFORMATION_NODE,
                FINAL_RESPONSE_NODE: FINAL_RESPONSE_NODE,
                STEP_RESOLVER_NODE: STEP_RESOLVER_NODE,
            },
        )

        graph.add_conditional_edges(
            STEP_RESOLVER_NODE,
            router,
            {
                CODE_EXECUTION_NODE: CODE_EXECUTION_NODE,
                INFORMATION_NODE: INFORMATION_NODE,
                FINAL_RESPONSE_NODE: FINAL_RESPONSE_NODE,
                STEP_RESOLVER_NODE: STEP_RESOLVER_NODE,
            },
        )

        graph.add_edge(CODE_ERROR_HANDLING_NODE, CODE_EXECUTION_NODE)
        graph.add_edge(FINAL_RESPONSE_NODE, END)

        logger.info("Sequential execution graph built successfully.")

        return graph

    def compile(self) -> None:
        """
        Compile the execution graph to prepare it for execution.
        """
        self.compiled_graph = self.graph.compile()
        logger.info("Sequential execution graph compiled successfully.")

    def execute(self, initial_state: OSHopeState) -> OSHopeState:
        """
        Execute the compiled graph starting from the initial state and return the final state after execution.
        """
        logger.info("Executing the Sequential Execution graph.")
        final_state = self.compiled_graph.invoke(initial_state)
        final_state = OSHopeState(**final_state)
        logger.info("The Sequential Execution Graph execution completed.")
        return final_state
