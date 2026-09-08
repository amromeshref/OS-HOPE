from time import time

from oshope.core.graphs.cognition.cognition_graph import CognitionGraph
from oshope.core.graphs.planning.planning_graph import PlanningGraph
from oshope.core.graphs.execution.sequential.execution_graph import (
    SequentialExecutionGraph,
)
from oshope.core.graphs.execution.parallel.execution_graph import ParallelExecutionGraph
from oshope.config.config import PARALLEL_EXECUTION_ENABLED
from oshope.core.graphs.memory.memory_graph import MemoryGraph
from oshope.interfaces.voice_input.main import VoiceInputInterface
from oshope.core.states.oshope_state import OSHopeState
from oshope.config.config import VOICE_INPUT_ENABLED, RAG_ENABLED, DEBUG_MODE
from oshope.tools.rag.main import RAGTool
from oshope.utils.helper_functions import save_debug_state, format_plan_for_user
from rich.console import Console
from rich.markdown import Markdown
import argparse
import time


class OSHopeApp:
    def __init__(self, use_voice=False, parallel_execution_enabled=PARALLEL_EXECUTION_ENABLED):
        self.use_voice = use_voice
        self.parallel_execution_enabled = parallel_execution_enabled
        self.equal_num = 50
        self.execution_time = 0.0
        self.timer_start = None

        # Graphs
        self.cognition_graph = CognitionGraph()
        self.planning_graph = PlanningGraph()

        if self.parallel_execution_enabled:
            self.execution_graph = ParallelExecutionGraph()
        else:
            self.execution_graph = SequentialExecutionGraph()

        self.cognition_graph.compile()
        self.planning_graph.compile()
        self.execution_graph.compile()

        if not self.parallel_execution_enabled:
            self.memory_graph = MemoryGraph()
            self.memory_graph.compile()

        if self.use_voice:
            if not VOICE_INPUT_ENABLED:
                print(
                    "Voice input is not enabled in settings. Please enable it to use voice input."
                )
                exit(1)
            self.voice_input_interface = VoiceInputInterface()

        if RAG_ENABLED:
            self.rag_tool = RAGTool()

    #================= TIMER =================

    def resume_timer(self):
        self.timer_start = time.perf_counter()

    def pause_timer(self):
        if self.timer_start is not None:
            self.execution_time += time.perf_counter() - self.timer_start

            self.timer_start = None

    def reset_timer(self):
        self.execution_time = 0.0
        self.timer_start = None

    # ================= PRINT AI MESSAGE =================
    def print_ai(self, text):
        self.pause_timer()
        print("=" * self.equal_num)
        print("AI:")
        self.render(text)
        print("=" * self.equal_num)
        self.resume_timer()

    # ================= INPUT =================
    def get_input(self):
        self.pause_timer()
        inp = None
        if self.use_voice:
            inp = self.voice_input()
        else:
            print("=" * self.equal_num)
            inp = input("YOU: ")

        if inp == "exit":
            print("Exiting...")
            exit(0)

        self.resume_timer()
        
        return inp

    def voice_input(self):
        self.pause_timer()
        self.voice_input_interface.service.reset()
        self.voice_input_interface.service.start_listening()
        text = self.voice_input_interface.service.transcribe_audio()

        print("You said:", text)
        print("Type 'ok' or edit:")

        print("=" * self.equal_num)
        inp = input("YOU: ")
        print("=" * self.equal_num)

        self.resume_timer()

        return text if inp == "ok" else inp

    # ================= UTIL =================
    def render(self, text):
        console = Console()
        console.print(Markdown(text))

    def append_hist(self, state: OSHopeState):
        state.multi_turn_conversation_history.append(
            {
                "turn_num": state.turn_num,
                "user_query": state.original_queries[-1],
                "assistant_response": state.multi_turn_generated_responses[-1],
            }
        )
        state.turn_num += 1

    def new_state_from(self, state: OSHopeState):
        """Clean state reset (you repeated this MANY times)"""
        new_state = OSHopeState()

        new_state.finalized_enhanced_query = state.finalized_enhanced_query
        new_state.parallel_execution_enabled = self.parallel_execution_enabled
        new_state.original_queries = state.original_queries
        new_state.turn_num = state.turn_num
        new_state.multi_turn_conversation_history = (
            state.multi_turn_conversation_history
        )
        new_state.original_queries_enhanced = state.original_queries_enhanced
        new_state.multi_turn_generated_responses = state.multi_turn_generated_responses
        new_state.clarification_attempts = state.clarification_attempts

        new_state.query_clarification = state.query_clarification
        new_state.query_classification = state.query_classification

        return new_state

    # ================= CLARIFICATION LOOP =================
    def clarification_loop(self, state: OSHopeState):
        while True:
            state = self.cognition_graph.execute(state)

            if DEBUG_MODE:
                save_debug_state(state, "clarification")

            if not state.query_clarification.is_clarification_needed:
                return state

            self.print_ai(state.query_clarification.generated_response)
            state.multi_turn_generated_responses.append(
                state.query_clarification.generated_response
            )
            self.append_hist(state)

            follow_up = self.get_input()
            state.original_queries.append(follow_up)

    # ================= COGNITION =================
    def handle_cognition(self, state):
        while True:
            state.query_classification.requires_follow_up = False
            state = self.cognition_graph.execute(state)

            if DEBUG_MODE:
                save_debug_state(state, "cognition")

            if not state.query_classification.requires_follow_up:
                return state

            self.print_ai(state.query_classification.generated_follow_up_response)
            state.multi_turn_generated_responses.append(
                state.query_classification.generated_follow_up_response
            )
            self.append_hist(state)

            follow_up = self.get_input()
            state.original_queries.append(follow_up)

            # deeper clarification
            state = self.clarification_loop(state)

            # reset cleanly
            state = self.new_state_from(state)

    # ================= VALIDATION =================
    def validation_loop(self, state: OSHopeState):
        while True:
            state = self.planning_graph.execute(state)
            if not state.user_validation.is_validation_required:
                break
            
            self.print_ai(state.user_validation.generated_response)
            state.multi_turn_generated_responses.append(
                state.user_validation.generated_response
            )
            self.append_hist(state)

            follow_up = self.get_input()
            state.original_queries.append(follow_up)            

            if DEBUG_MODE:
                save_debug_state(state, "validation")

        return state

    # ================= PLANNING =================
    def handle_planning(self, state):
        while True:
            state = self.planning_graph.execute(state)

            if DEBUG_MODE:
                save_debug_state(state, "planning")

            # clarification needed
            if state.planning.requires_follow_up:
                state.query_clarification.is_clarification_needed = True
                state = self.clarification_loop(state)
                state = self.new_state_from(state)
                continue

            # skip validation for info queries
            if state.query_classification.query_type == "information":
                return state

            # present plan to user
            plan_str = format_plan_for_user(state.planning)
            self.print_ai(plan_str)
            state.plan_presented = True
            state.multi_turn_generated_responses.append(plan_str)
            self.append_hist(state)

            follow_up = self.get_input()
            state.original_queries.append(follow_up)

            # validation loop
            state = self.validation_loop(state)

            if state.user_validation.user_feedback_type == "update_plan":
                state.plan_presented = False
                continue

            return state

    # ================= EXECUTION =================
    def handle_execution(self, state: OSHopeState):
        state = self.execution_graph.execute(state)

        self.print_ai(state.generated_final_response)
        state.multi_turn_generated_responses.append(state.generated_final_response)
        self.append_hist(state)

        if DEBUG_MODE:
            save_debug_state(state, "execution")

        return state

    def handle_memory(self, state: OSHopeState):
        state = self.memory_graph.execute(state)

        if DEBUG_MODE:
            save_debug_state(state, "memory")

        return state

    def handle_rag(self, state: OSHopeState):
        self.rag_tool.add_memories(
            session_id=state.turn_num, summaries=state.memory_extraction.summary_for_rag
        )
        return state

    # ================= MAIN LOOP =================
    def run(self):
        past_session_summaries = []

        while True:
            self.reset_timer()

            query = self.get_input()

            self.resume_timer()

            state = OSHopeState()
            state.parallel_execution_enabled = self.parallel_execution_enabled
            state.past_session_summaries = past_session_summaries
            state.original_queries.append(query)

            state = self.handle_cognition(state)
            state = self.handle_planning(state)
            state = self.handle_execution(state)

            if not self.parallel_execution_enabled:
                state = self.handle_memory(state)

            if RAG_ENABLED:
                state = self.handle_rag(state)

            past_session_summaries.append(state.memory_extraction.session_summary)

            self.pause_timer()
            state.execution_time = self.execution_time

            save_debug_state(state, "final_state")

def main():
    parser = argparse.ArgumentParser(description="OS-HOPE CLI")

    parser.add_argument("--voice", action="store_true", help="Enable voice input")
    parser.add_argument("--no-voice", action="store_true", help="Disable voice input")

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel execution",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel execution",
    )

    args = parser.parse_args()

    if args.voice:
        use_voice = True
    elif args.no_voice:
        use_voice = False
    else:
        use_voice = False

    if args.parallel:
        use_parallel = True
    elif args.no_parallel:
        use_parallel = False
    else:
        use_parallel = PARALLEL_EXECUTION_ENABLED

    app = OSHopeApp(
        use_voice=use_voice,
        parallel_execution_enabled=use_parallel,
    )
    app.run()
