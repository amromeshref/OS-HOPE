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

        self.memory_graph = None

        if not parallel_execution_enabled:
            self.memory_graph = MemoryGraph()
            self.memory_graph.compile()

        # ----------------------------------------------------
        # RAG
        # ----------------------------------------------------

        self.rag_tool = None

        if RAG_ENABLED:
            self.rag_tool = RAGTool()

    # ========================================================
    # NEW STATE FROM
    # ========================================================

    def new_state_from(self, state: OSHopeState):

        """
        Exactly follows the CLI new_state_from().
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

        new_state.turn_num = (
            state.turn_num
        )

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
        Execute one iteration of handle_cognition().
        """

        # Exactly as CLI:
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
        Execute one iteration of clarification_loop().

        Streamlit cannot block waiting for user input, so the
        while-loop from the CLI is represented by one graph
        execution per Streamlit interaction.
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
        # Planner requires clarification
        # ----------------------------------------------------

        if state.planning.requires_follow_up:

            # Same as CLI:
            #
            # state.query_clarification.is_clarification_needed = True

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
        Execute one iteration of the validation flow.

        Validation is handled by PlanningGraph.
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

        if self.memory_graph is not None:

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

        if RAG_ENABLED and self.rag_tool is not None:

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
    Exactly follows the semantics of CLI append_hist().

    CLI:

        state.multi_turn_generated_responses.append(
            response
        )

        state.multi_turn_conversation_history.append(
            {
                "turn_num": state.turn_num,
                "user_query": state.original_queries[-1],
                "assistant_response": response,
            }
        )

        state.turn_num += 1
    """

    if response is None:
        return state

    # --------------------------------------------------------
    # First store generated response
    # --------------------------------------------------------

    state.multi_turn_generated_responses.append(
        response
    )

    # --------------------------------------------------------
    # Then append history using CURRENT turn_num
    # --------------------------------------------------------

    state.multi_turn_conversation_history.append(
        {
            "turn_num": state.turn_num,
            "user_query": state.original_queries[-1],
            "assistant_response": response,
        }
    )

    # --------------------------------------------------------
    # Finally increment turn_num
    # --------------------------------------------------------

    state.turn_num += 1

    return state


def add_user_response_to_state_history(
    state,
    response,
):
    """
    User queries are stored in original_queries.

    The CLI append_hist() only creates history records when
    an assistant response is generated, so no separate user
    record is added here.
    """

    return state


def add_plan_presentation_to_state_history(
    state,
):
    """
    Store the formatted plan exactly like an assistant response
    in the CLI history.
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
# EXECUTION COMPLETION
# ============================================================


def complete_execution(state):
    """
    Same post-execution sequence as the CLI:

        state = handle_memory(state)

        if RAG_ENABLED:
            state = handle_rag(state)

        past_session_summaries.append(
            state.memory_extraction.session_summary
        )
    """

    backend = st.session_state.backend

    # --------------------------------------------------------
    # Memory
    # --------------------------------------------------------

    state = backend.run_memory(state)

    # --------------------------------------------------------
    # RAG
    # --------------------------------------------------------

    if RAG_ENABLED:
        state = backend.run_rag(state)

    # --------------------------------------------------------
    # Session summary
    # --------------------------------------------------------

    if (
        hasattr(state, "memory_extraction")
        and state.memory_extraction is not None
    ):

        summary = (
            state.memory_extraction.session_summary
        )

        if summary:
            st.session_state.past_session_summaries.append(
                summary
            )

    return state


# ============================================================
# STREAMLIT CONVERSATION HELPERS
# ============================================================


def add_text_message(
    role,
    content,
):

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

    st.session_state.conversation.append(
        {
            "role": "assistant",
            "type": "plan",
            "planning_state": planning_state,
        }
    )


# ============================================================
# SYNCHRONIZE SESSION STATE
# ============================================================


def sync_session_state(state):

    st.session_state.state = state

    # Keep Streamlit turn_num synchronized with OSHopeState.
    st.session_state.turn_num = state.turn_num


# ============================================================
# PRESENT ASSISTANT RESPONSE
# ============================================================


def present_assistant_response(
    state,
    response,
):

    state = add_assistant_response_to_state_history(
        state,
        response,
    )

    add_text_message(
        "assistant",
        response,
    )

    sync_session_state(state)

    return state


# ============================================================
# PLAN PRESENTATION
# ============================================================


def present_plan(state):

    # --------------------------------------------------------
    # Add textual plan to internal CLI-compatible history.
    # --------------------------------------------------------

    state, _ = (
        add_plan_presentation_to_state_history(
            state
        )
    )

    # --------------------------------------------------------
    # Add structured plan to Streamlit UI.
    # --------------------------------------------------------

    add_plan_message(
        state.planning
    )

    state.plan_presented = True

    st.session_state.state = state
    st.session_state.pending_state = state
    st.session_state.interaction_state = (
        "plan_validation"
    )

    st.session_state.turn_num = state.turn_num

    st.rerun()


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
        "Parallel Execution",
        value=(
            st.session_state
            .backend
            .parallel_execution_enabled
        ),
    )

    st.caption(
        "Execute independent steps in parallel to reduce execution time."
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

            state = complete_execution(state)

        response = state.generated_final_response

        state = present_assistant_response(
            state,
            response,
        )

        st.session_state.pending_state = None
        st.session_state.interaction_state = None

        sync_session_state(state)

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

        state = present_assistant_response(
            state,
            response,
        )

        st.session_state.pending_state = None
        st.session_state.interaction_state = None

        sync_session_state(state)

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
        # CLI:
        #
        # state = clarification_loop(state)
        #
        # Execute one iteration.
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

        if clarification_status == "clarification_needed":

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = (
                "clarification_from_cognition"
            )

            sync_session_state(state)

            st.rerun()

        # ----------------------------------------------------
        # Clarification completed
        #
        # CLI:
        #
        # state = self.new_state_from(state)
        # ----------------------------------------------------

        state = (
            st.session_state.backend
            .new_state_from(state)
        )

        # ----------------------------------------------------
        # CLI returns to handle_cognition()
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

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = (
                "cognition_follow_up"
            )

            sync_session_state(state)

            st.rerun()

        # ----------------------------------------------------
        # Cognition complete -> planning
        # ----------------------------------------------------

        with st.spinner(
            "Planning the requested operation..."
        ):

            state, planning_status = (
                st.session_state.backend
                .run_planning(state)
            )

        if planning_status == "clarification_needed":

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = (
                "clarification_from_planning"
            )

            sync_session_state(state)

            st.rerun()

        if planning_status == "ready":

            with st.spinner(
                "Generating response..."
            ):

                state = (
                    st.session_state.backend
                    .run_execution(state)
                )

                state = complete_execution(state)

            response = (
                state.generated_final_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = None

            sync_session_state(state)

            st.rerun()

        present_plan(state)


    # ========================================================
    # CLARIFICATION
    # ========================================================

    elif (
        st.session_state.interaction_state
        in (
            "clarification_from_cognition",
            "clarification_from_planning",
        )
    ):

        state = st.session_state.state

        clarification_origin = (
            st.session_state.interaction_state
        )

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
        # One clarification iteration
        # ----------------------------------------------------

        with st.spinner(
            "Processing your clarification..."
        ):

            state, clarification_status = (
                st.session_state.backend
                .run_clarification(state)
            )

        # ----------------------------------------------------
        # Still needs clarification
        # ----------------------------------------------------

        if clarification_status == "clarification_needed":

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            # Preserve the origin.
            st.session_state.interaction_state = (
                clarification_origin
            )

            sync_session_state(state)

            st.rerun()

        # ----------------------------------------------------
        # Clarification finished
        # ----------------------------------------------------

        state = (
            st.session_state.backend
            .new_state_from(state)
        )

        # ====================================================
        # CLARIFICATION ORIGINATED FROM COGNITION
        # ====================================================

        if (
            clarification_origin
            == "clarification_from_cognition"
        ):

            # CLI:
            #
            # clarification_loop()
            # -> new_state_from()
            # -> handle_cognition()
            #
            # Therefore cognition runs again.

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

                state = present_assistant_response(
                    state,
                    response,
                )

                st.session_state.interaction_state = (
                    "cognition_follow_up"
                )

                sync_session_state(state)

                st.rerun()

        # ====================================================
        # CLARIFICATION ORIGINATED FROM PLANNING
        # ====================================================

        # IMPORTANT:
        #
        # Do NOT run cognition here.
        #
        # CLI:
        #
        # planning
        #   -> clarification_loop()
        #   -> new_state_from()
        #   -> planning
        #
        # ====================================================

        with st.spinner(
            "Continuing the plan..."
        ):

            state, planning_status = (
                st.session_state.backend
                .run_planning(state)
            )

        # ----------------------------------------------------
        # Planner needs another clarification
        # ----------------------------------------------------

        if planning_status == "clarification_needed":

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = (
                "clarification_from_planning"
            )

            sync_session_state(state)

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

                state = complete_execution(state)

            response = (
                state.generated_final_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = None

            sync_session_state(state)

            st.rerun()

        # ----------------------------------------------------
        # Normal operation -> present plan
        # ----------------------------------------------------

        present_plan(state)


    # ========================================================
    # PLAN VALIDATION
    # ========================================================

    elif (
        st.session_state.pending_state is not None
        and st.session_state.interaction_state
        == "plan_validation"
    ):

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

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        with st.spinner(
            "Processing your feedback..."
        ):

            state = (
                st.session_state.backend
                .run_validation(state)
            )

        # ----------------------------------------------------
        # Validation still required
        # ----------------------------------------------------

        if state.user_validation.is_validation_required:

            response = (
                state
                .user_validation
                .generated_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.pending_state = state
            st.session_state.interaction_state = (
                "plan_validation"
            )

            sync_session_state(state)

            st.rerun()

        # ====================================================
        # UPDATE PLAN
        # ====================================================

        if (
            state.user_validation.user_feedback_type
            == "update_plan"
        ):

            state.plan_presented = False

            with st.spinner(
                "Updating the plan..."
            ):

                state, planning_status = (
                    st.session_state.backend
                    .run_planning(state)
                )

            # ------------------------------------------------
            # Planner requests clarification
            # ------------------------------------------------

            if planning_status == "clarification_needed":

                response = (
                    state
                    .query_clarification
                    .generated_response
                )

                state = present_assistant_response(
                    state,
                    response,
                )

                st.session_state.pending_state = state

                st.session_state.interaction_state = (
                    "clarification_from_planning"
                )

                sync_session_state(state)

                st.rerun()

            # ------------------------------------------------
            # Present updated plan
            # ------------------------------------------------

            present_plan(state)

        # ====================================================
        # VALIDATION COMPLETE -> EXECUTION
        # ====================================================

        with st.spinner(
            "Executing approved plan..."
        ):

            state = (
                st.session_state.backend
                .run_execution(state)
            )

            state = complete_execution(state)

        response = (
            state.generated_final_response
        )

        state = present_assistant_response(
            state,
            response,
        )

        st.session_state.pending_state = None
        st.session_state.interaction_state = None

        sync_session_state(state)

        st.rerun()


    # ========================================================
    # NEW QUERY
    # ========================================================

    else:

        # ----------------------------------------------------
        # UI history
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

        # Preserve previous session summaries.
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

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = (
                "cognition_follow_up"
            )

            sync_session_state(state)

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

        if planning_status == "clarification_needed":

            response = (
                state
                .query_clarification
                .generated_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            # IMPORTANT:
            # This clarification came from planning.
            # The next user response must return directly
            # to planning.
            st.session_state.interaction_state = (
                "clarification_from_planning"
            )

            sync_session_state(state)

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

                state = complete_execution(state)

            response = (
                state.generated_final_response
            )

            state = present_assistant_response(
                state,
                response,
            )

            st.session_state.interaction_state = None

            sync_session_state(state)

            st.rerun()

        # ====================================================
        # NORMAL OPERATION -> PLAN
        # ====================================================

        present_plan(state)