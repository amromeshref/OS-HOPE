import streamlit as st
import time

from oshope.core.graphs.cognition.cognition_graph import CognitionGraph
from oshope.core.graphs.planning.planning_graph import PlanningGraph
from oshope.core.graphs.execution.sequential.execution_graph import (
    SequentialExecutionGraph,
)
from oshope.core.graphs.execution.parallel.execution_graph import (
    ParallelExecutionGraph,
)

from oshope.config.config import (
    PARALLEL_EXECUTION_ENABLED,
    RAG_ENABLED,
    DEBUG_MODE,
)

from oshope.core.graphs.memory.memory_graph import MemoryGraph
from oshope.core.states.oshope_state import OSHopeState
from oshope.tools.rag.main import RAGTool

from oshope.utils.helper_functions import (
    save_debug_state,
    format_plan_for_user,
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="OS-HOPE",
    page_icon="🖥️",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .subtitle {
        color: #777;
        font-size: 1rem;
        margin-bottom: 2rem;
    }

    .plan-box {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
    }

    .status-box {
        padding: 12px;
        border-radius: 8px;
        margin: 10px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OS-HOPE BACKEND
# ============================================================


class OSHopeBackend:

    def __init__(self, parallel_execution_enabled=True):

        self.parallel_execution_enabled = (
            parallel_execution_enabled
        )

        # ----------------------------------------------------
        # Graphs
        # ----------------------------------------------------

        self.cognition_graph = CognitionGraph()
        self.planning_graph = PlanningGraph()

        if parallel_execution_enabled:
            self.execution_graph = ParallelExecutionGraph()
        else:
            self.execution_graph = SequentialExecutionGraph()

        self.cognition_graph.compile()
        self.planning_graph.compile()
        self.execution_graph.compile()

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        if not parallel_execution_enabled:
            self.memory_graph = MemoryGraph()
            self.memory_graph.compile()

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        if RAG_ENABLED:
            self.rag_tool = RAGTool()

    # ========================================================
    # NEW STATE FROM
    # ========================================================

    def new_state_from(self, state: OSHopeState):
        """
        Equivalent to the CLI new_state_from().

        After deeper clarification is completed, the temporary
        cognition/clarification state is cleaned while preserving
        the information required for the next cognition pass.
        """

        new_state = OSHopeState()

        new_state.finalized_enhanced_query = (
            state.finalized_enhanced_query
        )

        new_state.parallel_execution_enabled = (
            self.parallel_execution_enabled
        )

        new_state.original_queries = (
            state.original_queries
        )

        new_state.turn_num = state.turn_num

        new_state.multi_turn_conversation_history = (
            state.multi_turn_conversation_history
        )

        new_state.original_queries_enhanced = (
            state.original_queries_enhanced
        )

        new_state.multi_turn_generated_responses = (
            state.multi_turn_generated_responses
        )

        new_state.clarification_attempts = (
            state.clarification_attempts
        )

        new_state.query_clarification = (
            state.query_clarification
        )

        new_state.query_classification = (
            state.query_classification
        )

        return new_state

    # ========================================================
    # COGNITION
    # ========================================================

    def run_cognition(self, state):

        """
        Execute one cognition pass.

        Equivalent to one iteration of the CLI
        handle_cognition() loop.

        If classification requires a follow-up, the caller
        pauses and waits for the user's response.
        """

        state.query_classification.requires_follow_up = False

        state = self.cognition_graph.execute(state)

        if DEBUG_MODE:
            save_debug_state(
                state,
                "cognition",
            )

        if state.query_classification.requires_follow_up:
            return state, "follow_up"

        return state, "success"

    # ========================================================
    # CLARIFICATION
    # ========================================================

    def run_clarification(self, state):
        """
        Execute ONE iteration of clarification_loop().

        CLI equivalent:

            state = self.cognition_graph.execute(state)

            if clarification is still required:
                ask user
            else:
                return state
        """

        state = self.cognition_graph.execute(state)

        if DEBUG_MODE:
            save_debug_state(
                state,
                "clarification",
            )

        if state.query_clarification.is_clarification_needed:
            return state, "clarification_needed"

        return state, "success"

    # ========================================================
    # PLANNING
    # ========================================================

    def run_planning(self, state):

        state = self.planning_graph.execute(state)

        if DEBUG_MODE:
            save_debug_state(
                state,
                "planning",
            )

        # ----------------------------------------------------
        # Planner needs clarification
        # ----------------------------------------------------

        if state.planning.requires_follow_up:

            state.query_clarification.is_clarification_needed = (
                True
            )

            return state, "clarification_needed"

        # ----------------------------------------------------
        # Information query
        # ----------------------------------------------------

        if (
            state.query_classification.query_type
            == "information"
        ):

            return state, "ready"

        # ----------------------------------------------------
        # Normal operation
        # ----------------------------------------------------

        return state, "validation_required"

    # ========================================================
    # VALIDATION
    # ========================================================

    def run_validation(self, state):

        """
        Execute ONE iteration of the CLI validation_loop().

        The important point is that validation is handled by
        the planning graph, not the cognition graph.
        """

        state = self.planning_graph.execute(state)

        if DEBUG_MODE:
            save_debug_state(
                state,
                "validation",
            )

        return state

    # ========================================================
    # EXECUTION
    # ========================================================

    def run_execution(self, state):

        start_time = time.perf_counter()

        state = self.execution_graph.execute(state)

        state.execution_time = (
            time.perf_counter() - start_time
        )

        if DEBUG_MODE:
            save_debug_state(
                state,
                "execution",
            )

        return state

    # ========================================================
    # MEMORY
    # ========================================================

    def run_memory(self, state):

        if not self.parallel_execution_enabled:

            state = self.memory_graph.execute(state)

            if DEBUG_MODE:
                save_debug_state(
                    state,
                    "memory",
                )

        return state

    # ========================================================
    # RAG
    # ========================================================

    def run_rag(self, state):

        if RAG_ENABLED:

            self.rag_tool.add_memories(
                session_id=state.turn_num,
                summaries=(
                    state.memory_extraction.summary_for_rag
                ),
            )

        return state


# ============================================================
# STATE HISTORY HELPERS
# ============================================================


def add_assistant_response_to_state_history(
    state,
    response,
):
    """
    Equivalent to the CLI:

        self.print_ai(response)
        state.multi_turn_generated_responses.append(response)
        self.append_hist(state)

    Streamlit additionally stores the response separately in
    its visual conversation.
    """

    if response is None:
        return state

    state.multi_turn_generated_responses.append(
        response
    )

    state.multi_turn_conversation_history.append(
        {
            "role": "assistant",
            "content": response,
        }
    )

    return state


def add_user_response_to_state_history(
    state,
    response,
):
    """
    Add the user's response to the state conversation history.

    original_queries is updated separately because that is the
    field used by the OS-HOPE cognition/planning pipeline.
    """

    state.multi_turn_conversation_history.append(
        {
            "role": "user",
            "content": response,
        }
    )

    return state


def add_plan_presentation_to_state_history(
    state,
):
    """
    Equivalent to the CLI plan presentation:

        plan_str = format_plan_for_user(state.planning)
        self.print_ai(plan_str)
        state.multi_turn_generated_responses.append(plan_str)
        self.append_hist(state)

    The returned plan_str is used ONLY for OS-HOPE's internal
    multi-turn history.

    Streamlit uses state.planning separately for the rich UI.
    """

    plan_str = format_plan_for_user(
        state.planning
    )

    state = add_assistant_response_to_state_history(
        state,
        plan_str,
    )

    return state, plan_str


# ============================================================
# STREAMLIT CONVERSATION HELPERS
# ============================================================


def add_text_message(
    role,
    content,
):
    """
    Add a normal text message to the Streamlit UI history.
    """

    st.session_state.conversation.append(
        {
            "role": role,
            "type": "text",
            "content": content,
        }
    )


def add_plan_message(
    planning_state,
):
    """
    Add a structured plan to the Streamlit UI history.

    This is intentionally different from plan_str.

    plan_str:
        used by OS-HOPE internal state history.

    planning_state:
        used by Streamlit to reconstruct the rich plan UI.
    """

    st.session_state.conversation.append(
        {
            "role": "assistant",
            "type": "plan",
            "planning_state": planning_state,
        }
    )


# ============================================================
# PLAN RENDERING
# ============================================================


def render_plan(planning_state):

    st.subheader("📋 Proposed Plan")

    # --------------------------------------------------------
    # Fulfillment summary
    # --------------------------------------------------------

    if planning_state.fulfillment_summary:

        st.info(
            planning_state.fulfillment_summary
        )

    st.markdown(
        f"**{len(planning_state.plan_steps)} "
        f"step{'s' if len(planning_state.plan_steps) != 1 else ''}**"
    )

    # --------------------------------------------------------
    # Steps
    # --------------------------------------------------------

    for index, step in enumerate(
        planning_state.plan_steps,
        start=1,
    ):

        step_details = step.step_details

        if step.step_type == "command":
            step_icon = "⚙️"
            step_type = "COMMAND"
        else:
            step_icon = "ℹ️"
            step_type = "INFORMATION"

        with st.container(border=True):

            col1, col2, col3 = st.columns(
                [0.12, 0.68, 0.20]
            )

            with col1:

                st.markdown(
                    f"### {step_icon} {index}"
                )

            with col2:

                st.markdown(
                    f"**{step.description}**"
                )

            with col3:

                st.caption(step_type)

            # ------------------------------------------------
            # Command
            # ------------------------------------------------

            if step.step_type == "command":

                command = step_details

                risk = command.safety_risk

                risk_labels = {
                    "low": "🟢 Low Risk",
                    "medium": "🟡 Medium Risk",
                    "high": "🔴 High Risk",
                }

                st.markdown(
                    f"**Safety:** "
                    f"{risk_labels.get(risk, risk)}"
                )

                st.markdown("**Command**")

                st.code(
                    command.command,
                    language="bash",
                )

                if command.description:

                    st.markdown(
                        f"**What it does:** "
                        f"{command.description}"
                    )

                col1, col2 = st.columns(2)

                with col1:

                    st.markdown(
                        "**Execution mode**"
                    )

                    st.write(
                        command.execution_mode
                    )

                with col2:

                    st.markdown(
                        "**Expected output**"
                    )

                    st.write(
                        command.expected_output
                        or "No output"
                    )

                # ------------------------------------------------
                # Input variables
                # ------------------------------------------------

                if command.input_variables:

                    with st.expander(
                        "Input variables"
                    ):

                        for variable in command.input_variables:

                            st.markdown(
                                f"**`{variable.variable_name}`**"
                            )

                            st.caption(
                                variable.description
                            )

                # ------------------------------------------------
                # Output variables
                # ------------------------------------------------

                if command.output_variables:

                    with st.expander(
                        "Output variables"
                    ):

                        for variable in command.output_variables:

                            st.markdown(
                                f"**`{variable.variable_name}`**"
                            )

                            st.caption(
                                variable.description
                            )

            # ------------------------------------------------
            # Information step
            # ------------------------------------------------

            elif step.step_type == "information":

                information = step_details

                st.markdown(
                    f"**Information requested:** "
                    f"{information.description}"
                )

            # ------------------------------------------------
            # Dependencies
            # ------------------------------------------------

            if step.dependencies_required:

                dependencies = ", ".join(
                    f"Step {i + 1}"
                    for i in step.dependency_step_indices
                )

                st.markdown(
                    f"🔗 **Depends on:** {dependencies}"
                )

            else:

                st.markdown(
                    "🔗 **Dependencies:** None"
                )

            # ------------------------------------------------
            # Iteration
            # ------------------------------------------------

            if step.requires_iteration:

                st.markdown(
                    "🔁 **This step requires iteration**"
                )


# ============================================================
# CONVERSATION RENDERING
# ============================================================


def render_conversation():

    for message in st.session_state.conversation:

        role = message["role"]

        with st.chat_message(role):

            if message["type"] == "text":

                st.markdown(
                    message["content"]
                )

            elif message["type"] == "plan":

                render_plan(
                    message["planning_state"]
                )


# ============================================================
# SESSION STATE
# ============================================================


def initialize_session():

    if "backend" not in st.session_state:

        st.session_state.backend = OSHopeBackend(
            parallel_execution_enabled=(
                PARALLEL_EXECUTION_ENABLED
            )
        )

    if "state" not in st.session_state:

        st.session_state.state = None

    if "conversation" not in st.session_state:

        st.session_state.conversation = []

    if "pending_state" not in st.session_state:

        st.session_state.pending_state = None

    if "interaction_state" not in st.session_state:

        st.session_state.interaction_state = None

    if "past_session_summaries" not in st.session_state:

        st.session_state.past_session_summaries = []

    if "turn_num" not in st.session_state:

        st.session_state.turn_num = 0


initialize_session()


# ============================================================
# SIDEBAR
# ============================================================


with st.sidebar:

    st.title("⚙️ OS-HOPE")

    st.markdown(
        """
        **Human-Oversight Multi-Agent OS Assistant**

        OS-HOPE generates and analyzes OS operations while
        keeping the human in control of execution.
        """
    )

    st.divider()

    st.subheader("Execution")

    parallel_enabled = st.toggle(
        "Parallel execution",
        value=(
            st.session_state
            .backend
            .parallel_execution_enabled
        ),
    )

    if (
        parallel_enabled
        != st.session_state.backend.parallel_execution_enabled
    ):

        st.session_state.backend = OSHopeBackend(
            parallel_execution_enabled=parallel_enabled
        )

        st.session_state.pending_state = None
        st.session_state.interaction_state = None

    st.divider()

    st.subheader("Session")

    if st.button(
        "🗑️ Clear conversation",
        use_container_width=True,
    ):

        st.session_state.conversation = []
        st.session_state.state = None
        st.session_state.pending_state = None
        st.session_state.interaction_state = None
        st.session_state.past_session_summaries = []
        st.session_state.turn_num = 0

        st.rerun()

    st.caption(
        f"Parallel execution: "
        f"{'Enabled' if parallel_enabled else 'Disabled'}"
    )


# ============================================================
# HEADER
# ============================================================


st.markdown(
    '<div class="main-title">🖥️ OS-HOPE</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Human-Oversight Multi-Agent Assistant for "
    "Operating-System Automation"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# DISPLAY CONVERSATION
# ============================================================

render_conversation()


# ============================================================
# PENDING PLAN VALIDATION BUTTONS
# ============================================================

if (
    st.session_state.pending_state is not None
    and st.session_state.interaction_state
    == "plan_validation"
):

    st.divider()

    st.warning(
        "Review the proposed plan carefully before "
        "allowing OS-HOPE to proceed."
    )

    col1, col2 = st.columns(2)

    with col1:

        approve = st.button(
            "✅ Approve Plan",
            use_container_width=True,
            type="primary",
        )

    with col2:

        reject = st.button(
            "❌ Reject Plan",
            use_container_width=True,
        )

    # ========================================================
    # APPROVE
    # ========================================================

    if approve:

        state = st.session_state.pending_state

        state.user_validation.user_feedback_type = (
            "approve"
        )

        state.user_validation.is_validation_required = (
            False
        )

        # ----------------------------------------------------
        # State history
        # ----------------------------------------------------

        state = add_user_response_to_state_history(
            state,
            "approve",
        )

        with st.spinner(
            "Executing approved plan..."
        ):

            state = (
                st.session_state.backend
                .run_execution(state)
            )

        response = state.generated_final_response

        state = add_assistant_response_to_state_history(
            state,
            response,
        )

        add_text_message(
            "assistant",
            response,
        )

        st.session_state.state = state
        st.session_state.pending_state = None
        st.session_state.interaction_state = None

        st.rerun()

    # ========================================================
    # REJECT
    # ========================================================

    if reject:

        state = st.session_state.pending_state

        state = add_user_response_to_state_history(
            state,
            "reject",
        )

        response = (
            "Execution cancelled. "
            "The proposed plan was not approved."
        )

        state = add_assistant_response_to_state_history(
            state,
            response,
        )

        add_text_message(
            "assistant",
            response,
        )

        st.session_state.state = state
        st.session_state.pending_state = None
        st.session_state.interaction_state = None

        st.rerun()


# ============================================================
# CHAT INPUT
# ============================================================

user_query = st.chat_input(
    "Ask OS-HOPE to help with your operating system..."
)


if user_query:

    # ========================================================
    # COGNITION FOLLOW-UP
    # ========================================================

    if (
        st.session_state.interaction_state
        == "cognition_follow_up"
    ):

        state = st.session_state.state

        # ----------------------------------------------------
        # User answered classification follow-up.
        #
        # CLI:
        #
        # follow_up = self.get_input()
        # state.original_queries.append(follow_up)
        # state = self.clarification_loop(state)
        # state = self.new_state_from(state)
        # ----------------------------------------------------

        add_text_message(
            "user",
            user_query,
        )

        state.original_queries.append(
            user_query
        )

        state = add_user_response_to_state_history(
            state,
            user_query,
        )

        # ----------------------------------------------------
        # Start deeper clarification loop.
        #
        # Streamlit performs ONE iteration here.
        # If another clarification is required, the next
        # rerun will continue from interaction_state.
        # ----------------------------------------------------

        with st.spinner(
            "Checking clarification..."
        ):

            state, clarification_status = (
                st.session_state.backend
                .run_clarification(state)
            )

        if (
            clarification_status
            == "clarification_needed"
        ):

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state

            st.session_state.interaction_state = (
                "clarification"
            )

            st.rerun()

        # ----------------------------------------------------
        # Clarification completed.
        #
        # Equivalent to:
        #
        # state = self.new_state_from(state)
        # continue
        #
        # in CLI handle_cognition().
        # ----------------------------------------------------

        state = (
            st.session_state.backend
            .new_state_from(state)
        )

        st.session_state.state = state

        # ----------------------------------------------------
        # Continue cognition immediately.
        #
        # This is the next iteration of the CLI
        # handle_cognition() loop.
        # ----------------------------------------------------

        with st.spinner(
            "Understanding your request..."
        ):

            state, cognition_status = (
                st.session_state.backend
                .run_cognition(state)
            )

        if cognition_status == "follow_up":

            response = (
                state
                .query_classification
                .generated_follow_up_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = (
                "cognition_follow_up"
            )

            st.rerun()

        # ----------------------------------------------------
        # Cognition complete.
        #
        # Continue directly to planning.
        # ----------------------------------------------------

        st.session_state.state = state

        # ====================================================
        # PLANNING
        # ====================================================

        with st.spinner(
            "Planning the requested operation..."
        ):

            state, planning_status = (
                st.session_state.backend
                .run_planning(state)
            )

        # ----------------------------------------------------
        # Planner clarification
        # ----------------------------------------------------

        if (
            planning_status
            == "clarification_needed"
        ):

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = (
                "clarification"
            )

            st.rerun()

        # ----------------------------------------------------
        # Information query
        # ----------------------------------------------------

        if planning_status == "ready":

            with st.spinner(
                "Generating response..."
            ):

                state = (
                    st.session_state.backend
                    .run_execution(state)
                )

            response = (
                state.generated_final_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = None

            st.rerun()

        # ----------------------------------------------------
        # Present plan
        # ----------------------------------------------------

        state, plan_str = (
            add_plan_presentation_to_state_history(
                state
            )
        )

        add_plan_message(
            state.planning
        )

        state.plan_presented = True

        st.session_state.state = state
        st.session_state.pending_state = state
        st.session_state.interaction_state = (
            "plan_validation"
        )

        st.rerun()

    # ========================================================
    # DEEP CLARIFICATION
    # ========================================================

    elif (
        st.session_state.interaction_state
        == "clarification"
    ):

        state = st.session_state.state

        add_text_message(
            "user",
            user_query,
        )

        state.original_queries.append(
            user_query
        )

        state = add_user_response_to_state_history(
            state,
            user_query,
        )

        # ----------------------------------------------------
        # One iteration of clarification_loop()
        # ----------------------------------------------------

        with st.spinner(
            "Processing your clarification..."
        ):

            state, clarification_status = (
                st.session_state.backend
                .run_clarification(state)
            )

        # ----------------------------------------------------
        # More clarification required
        # ----------------------------------------------------

        if (
            clarification_status
            == "clarification_needed"
        ):

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = (
                "clarification"
            )

            st.rerun()

        # ----------------------------------------------------
        # Clarification completed.
        #
        # EXACT CLI behavior:
        #
        # state = self.new_state_from(state)
        # ----------------------------------------------------

        state = (
            st.session_state.backend
            .new_state_from(state)
        )

        st.session_state.state = state

        # ----------------------------------------------------
        # Continue cognition.
        #
        # This reproduces the continuation of
        # handle_cognition().
        # ----------------------------------------------------

        with st.spinner(
            "Understanding your request..."
        ):

            state, cognition_status = (
                st.session_state.backend
                .run_cognition(state)
            )

        # ----------------------------------------------------
        # Another classification follow-up
        # ----------------------------------------------------

        if cognition_status == "follow_up":

            response = (
                state
                .query_classification
                .generated_follow_up_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = (
                "cognition_follow_up"
            )

            st.rerun()

        # ====================================================
        # PLANNING
        # ====================================================

        with st.spinner(
            "Planning the requested operation..."
        ):

            state, planning_status = (
                st.session_state.backend
                .run_planning(state)
            )

        # ----------------------------------------------------
        # Planner clarification
        # ----------------------------------------------------

        if (
            planning_status
            == "clarification_needed"
        ):

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = (
                "clarification"
            )

            st.rerun()

        # ----------------------------------------------------
        # Information query
        # ----------------------------------------------------

        if planning_status == "ready":

            with st.spinner(
                "Generating response..."
            ):

                state = (
                    st.session_state.backend
                    .run_execution(state)
                )

            response = (
                state.generated_final_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = None

            st.rerun()

        # ----------------------------------------------------
        # Present plan
        # ----------------------------------------------------

        state, plan_str = (
            add_plan_presentation_to_state_history(
                state
            )
        )

        add_plan_message(
            state.planning
        )

        state.plan_presented = True

        st.session_state.state = state
        st.session_state.pending_state = state
        st.session_state.interaction_state = (
            "plan_validation"
        )

        st.rerun()

    # ========================================================
    # PLAN VALIDATION
    # ========================================================

    elif (
        st.session_state.pending_state is not None
        and st.session_state.interaction_state
        == "plan_validation"
    ):

        # ----------------------------------------------------
        # This is NOT a new query.
        #
        # It is validation feedback.
        #
        # CLI:
        #
        # follow_up = self.get_input()
        # state.original_queries.append(follow_up)
        # state = self.validation_loop(state)
        #
        # validation_loop() executes planning_graph.
        # ----------------------------------------------------

        add_text_message(
            "user",
            user_query,
        )

        state = st.session_state.pending_state

        state.original_queries.append(
            user_query
        )

        state = add_user_response_to_state_history(
            state,
            user_query,
        )

        # ====================================================
        # VALIDATION ITERATION
        # ====================================================

        with st.spinner(
            "Processing your feedback..."
        ):

            state = (
                st.session_state.backend
                .run_validation(state)
            )

        # ====================================================
        # VALIDATION STILL REQUIRED
        # ====================================================

        if state.user_validation.is_validation_required:

            response = (
                state.user_validation.generated_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.pending_state = state
            st.session_state.state = state
            st.session_state.interaction_state = (
                "plan_validation"
            )

            st.rerun()

        # ====================================================
        # UPDATE PLAN
        # ====================================================

        if (
            state.user_validation.user_feedback_type
            == "update_plan"
        ):

            # ------------------------------------------------
            # CLI:
            #
            # state.plan_presented = False
            # continue
            # ------------------------------------------------

            state.plan_presented = False

            with st.spinner(
                "Updating the plan..."
            ):

                state, planning_status = (
                    st.session_state.backend
                    .run_planning(state)
                )

            # ------------------------------------------------
            # Planner needs clarification
            # ------------------------------------------------

            if (
                planning_status
                == "clarification_needed"
            ):

                response = (
                    state
                    .query_clarification
                    .generated_response
                )

                state = add_assistant_response_to_state_history(
                    state,
                    response,
                )

                add_text_message(
                    "assistant",
                    response,
                )

                st.session_state.state = state
                st.session_state.pending_state = None
                st.session_state.interaction_state = (
                    "clarification"
                )

                st.rerun()

            # ------------------------------------------------
            # Present updated plan
            # ------------------------------------------------

            state, plan_str = (
                add_plan_presentation_to_state_history(
                    state
                )
            )

            add_plan_message(
                state.planning
            )

            state.plan_presented = True

            st.session_state.state = state
            st.session_state.pending_state = state
            st.session_state.interaction_state = (
                "plan_validation"
            )

            st.rerun()

        # ====================================================
        # VALIDATION FINISHED
        # ====================================================

        with st.spinner(
            "Executing approved plan..."
        ):

            state = (
                st.session_state.backend
                .run_execution(state)
            )

        response = (
            state.generated_final_response
        )

        state = add_assistant_response_to_state_history(
            state,
            response,
        )

        add_text_message(
            "assistant",
            response,
        )

        st.session_state.state = state
        st.session_state.pending_state = None
        st.session_state.interaction_state = None

        st.rerun()

    # ========================================================
    # NEW QUERY
    # ========================================================

    else:

        # ----------------------------------------------------
        # Add user message to Streamlit UI
        # ----------------------------------------------------

        add_text_message(
            "user",
            user_query,
        )

        # ----------------------------------------------------
        # Create new state
        # ----------------------------------------------------

        state = OSHopeState()

        state.parallel_execution_enabled = (
            st.session_state
            .backend
            .parallel_execution_enabled
        )

        state.past_session_summaries = (
            st.session_state
            .past_session_summaries
        )

        state.original_queries.append(
            user_query
        )

        state.turn_num = (
            st.session_state.turn_num
        )

        # ----------------------------------------------------
        # State history
        # ----------------------------------------------------

        state = add_user_response_to_state_history(
            state,
            user_query,
        )

        # ====================================================
        # COGNITION
        # ====================================================

        with st.spinner(
            "Understanding your request..."
        ):

            state, cognition_status = (
                st.session_state.backend
                .run_cognition(state)
            )

        # ====================================================
        # CLASSIFICATION FOLLOW-UP
        # ====================================================

        if cognition_status == "follow_up":

            response = (
                state
                .query_classification
                .generated_follow_up_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state

            st.session_state.interaction_state = (
                "cognition_follow_up"
            )

            st.rerun()

        # ====================================================
        # PLANNING
        # ====================================================

        with st.spinner(
            "Planning the requested operation..."
        ):

            state, planning_status = (
                st.session_state.backend
                .run_planning(state)
            )

        # ====================================================
        # PLANNER CLARIFICATION
        # ====================================================

        if (
            planning_status
            == "clarification_needed"
        ):

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state

            st.session_state.interaction_state = (
                "clarification"
            )

            st.rerun()

        # ====================================================
        # INFORMATION QUERY
        # ====================================================

        if planning_status == "ready":

            with st.spinner(
                "Generating response..."
            ):

                state = (
                    st.session_state.backend
                    .run_execution(state)
                )

            response = (
                state.generated_final_response
            )

            state = add_assistant_response_to_state_history(
                state,
                response,
            )

            add_text_message(
                "assistant",
                response,
            )

            st.session_state.state = state
            st.session_state.interaction_state = None

            st.rerun()

        # ====================================================
        # PLAN PRESENTATION
        # ====================================================

        # ----------------------------------------------------
        # CLI equivalent:
        #
        # plan_str = format_plan_for_user(state.planning)
        # self.print_ai(plan_str)
        # state.plan_presented = True
        # state.multi_turn_generated_responses.append(plan_str)
        # self.append_hist(state)
        # ----------------------------------------------------

        state, plan_str = (
            add_plan_presentation_to_state_history(
                state
            )
        )

        # ----------------------------------------------------
        # Rich Streamlit representation
        # ----------------------------------------------------

        add_plan_message(
            state.planning
        )

        state.plan_presented = True

        # ----------------------------------------------------
        # Wait for validation.
        # ----------------------------------------------------

        st.session_state.state = state

        st.session_state.pending_state = state

        st.session_state.interaction_state = (
            "plan_validation"
        )

        st.rerun()