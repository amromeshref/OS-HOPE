import argparse
import time

from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.prompt import Prompt
from rich.status import Status

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
    VOICE_INPUT_ENABLED,
    RAG_ENABLED,
    DEBUG_MODE,
)

from oshope.core.graphs.memory.memory_graph import MemoryGraph
from oshope.interfaces.voice_input.main import VoiceInputInterface
from oshope.core.states.oshope_state import OSHopeState
from oshope.tools.rag.main import RAGTool
from oshope.utils.helper_functions import (
    save_debug_state,
    format_plan_for_user,
)


class OSHopeApp:
    """
    OS-HOPE command-line interface.

    Application flow:

        Cognition
            ↓
        Clarification
            ↓
        Planning
            ↓
        Plan Presentation
            ↓
        Human Validation
            ↓
        Execution
            ↓
        Memory / RAG

    Rich is used only to improve the terminal user experience.
    """

    def __init__(
        self,
        use_voice=False,
        parallel_execution_enabled=PARALLEL_EXECUTION_ENABLED,
    ):
        self.use_voice = use_voice
        self.parallel_execution_enabled = parallel_execution_enabled

        self.execution_time = 0.0
        self.timer_start = None

        self.console = Console()

        # ============================================================
        # GRAPHS
        # ============================================================

        self.cognition_graph = CognitionGraph()
        self.planning_graph = PlanningGraph()

        if self.parallel_execution_enabled:
            self.execution_graph = ParallelExecutionGraph()
        else:
            self.execution_graph = SequentialExecutionGraph()

        self.cognition_graph.compile()
        self.planning_graph.compile()
        self.execution_graph.compile()

        # Memory is only used for sequential execution,
        # preserving the original implementation.
        if not self.parallel_execution_enabled:
            self.memory_graph = MemoryGraph()
            self.memory_graph.compile()

        # ============================================================
        # VOICE
        # ============================================================

        if self.use_voice:
            if not VOICE_INPUT_ENABLED:
                self.console.print(
                    Panel(
                        "[bold red]Voice input is disabled[/bold red]\n\n"
                        "Enable VOICE_INPUT_ENABLED in the OS-HOPE settings "
                        "before using --voice.",
                        title="Configuration Error",
                        border_style="red",
                    )
                )
                raise SystemExit(1)

            self.voice_input_interface = VoiceInputInterface()

        # ============================================================
        # RAG
        # ============================================================

        if RAG_ENABLED:
            self.rag_tool = RAGTool()

    # ================================================================
    # TIMER
    # ================================================================

    def resume_timer(self):
        self.timer_start = time.perf_counter()

    def pause_timer(self):
        if self.timer_start is not None:
            self.execution_time += time.perf_counter() - self.timer_start
            self.timer_start = None

    def reset_timer(self):
        self.execution_time = 0.0
        self.timer_start = None

    # ================================================================
    # UI
    # ================================================================

    def print_banner(self):
        """Display the OS-HOPE startup banner."""

        self.console.print()

        # ------------------------------------------------------------
        # Header
        # ------------------------------------------------------------

        header = Text()
        header.append(
            "🖥️  OS-HOPE\n",
            style="bold cyan",
        )
        header.append(
            "Human-Oversight Operating-System Assistant",
            style="dim",
        )

        self.console.print(
            Panel(
                header,
                border_style="cyan",
                padding=(1, 2),
            )
        )

        # ------------------------------------------------------------
        # Configuration
        # ------------------------------------------------------------

        table = Table.grid(padding=(0, 2))

        table.add_column(
            style="dim",
            justify="right",
        )

        table.add_column()

        table.add_row(
            "Execution",
            (
                "[bold cyan]⚡ Parallel[/bold cyan]"
                if self.parallel_execution_enabled
                else "[bold yellow]→ Sequential[/bold yellow]"
            ),
        )

        table.add_row(
            "Voice",
            (
                "[bold green]✓ Enabled[/bold green]"
                if self.use_voice
                else "[dim]✗ Disabled[/dim]"
            ),
        )

        table.add_row(
            "RAG",
            (
                "[bold green]✓ Enabled[/bold green]"
                if RAG_ENABLED
                else "[dim]✗ Disabled[/dim]"
            ),
        )

        self.console.print(
            Panel(
                table,
                title="[bold]Configuration[/bold]",
                border_style="dim",
                padding=(1, 2),
            )
        )

        # ------------------------------------------------------------
        # Help
        # ------------------------------------------------------------

        self.console.print()

        self.console.print("[dim]Type [bold]exit[/bold] at any time to quit.[/dim]")

        self.console.print()

    def print_section(self, title):
        """Display a section divider."""

        self.console.print()

        self.console.print(
            Rule(
                f"[bold cyan]{title}[/bold cyan]",
                style="cyan",
            )
        )

    def print_ai(self, text):
        """Display an AI response."""

        self.pause_timer()

        self.console.print()

        self.console.print(
            Panel(
                Markdown(text),
                title="[bold cyan]🤖 OS-HOPE[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

        self.console.print()

        self.resume_timer()

    def print_info(self, text):
        self.pause_timer()

        self.console.print(f"[dim]ℹ {text}[/dim]")

        self.resume_timer()

    def print_success(self, text):
        self.pause_timer()

        self.console.print(
            Panel(
                f"[bold green]✓ {text}[/bold green]",
                border_style="green",
                padding=(0, 1),
            )
        )

        self.resume_timer()

    def print_warning(self, text):
        self.pause_timer()

        self.console.print(
            Panel(
                f"[bold yellow]⚠ {text}[/bold yellow]",
                border_style="yellow",
                padding=(0, 1),
            )
        )

        self.resume_timer()

    # ================================================================
    # INPUT
    # ================================================================

    def get_input(self):
        """
        Get input from the user.

        Supports both regular text and voice input.
        """

        self.pause_timer()

        if self.use_voice:
            inp = self.voice_input()
        else:
            self.console.print(
                "[bold green]You ›[/bold green] ",
                end="",
            )
            inp = input()

        if inp is None:
            inp = ""

        inp = inp.strip()

        if inp.lower() == "exit":
            self.console.print()

            self.console.print(
                Panel(
                    "[bold cyan]Thank you for using OS-HOPE.[/bold cyan]\n"
                    "[dim]Goodbye![/dim]",
                    border_style="cyan",
                )
            )

            raise SystemExit(0)

        self.resume_timer()

        return inp

    def voice_input(self):
        """
        Record voice input, show the transcription, and allow
        the user to confirm or edit it.
        """

        self.pause_timer()

        self.console.print()

        with Status(
            "[cyan]Listening... Speak now[/cyan]",
            console=self.console,
            spinner="dots",
        ):
            self.voice_input_interface.service.reset()
            self.voice_input_interface.service.start_listening()
            text = self.voice_input_interface.service.transcribe_audio()

        self.console.print()

        self.console.print(
            Panel(
                f"[bold]{text}[/bold]",
                title="[cyan]🎤 Transcription[/cyan]",
                border_style="cyan",
            )
        )

        self.console.print()

        choice = Prompt.ask(
            "[bold green]You[/bold green]",
            default="ok",
        )

        self.resume_timer()

        if choice.lower() == "ok":
            return text

        return choice

    # ================================================================
    # RENDER
    # ================================================================

    def render(self, text):
        self.console.print(Markdown(text))

    # ================================================================
    # HISTORY
    # ================================================================

    def append_hist(self, state: OSHopeState):
        state.multi_turn_conversation_history.append(
            {
                "turn_num": state.turn_num,
                "user_query": state.original_queries[-1],
                "assistant_response": (state.multi_turn_generated_responses[-1]),
            }
        )

        state.turn_num += 1

    # ================================================================
    # STATE
    # ================================================================

    def new_state_from(self, state: OSHopeState):
        """
        Create a clean state while preserving the conversation context.

        This follows the original OS-HOPE state-reset behavior.
        """

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

    # ================================================================
    # CLARIFICATION LOOP
    # ================================================================

    def clarification_loop(self, state: OSHopeState):

        while True:

            self.print_section("Clarification")

            state = self.cognition_graph.execute(state)

            if DEBUG_MODE:
                save_debug_state(
                    state,
                    "clarification",
                )

            if not state.query_clarification.is_clarification_needed:
                return state

            self.print_ai(state.query_clarification.generated_response)

            state.multi_turn_generated_responses.append(
                state.query_clarification.generated_response
            )

            self.append_hist(state)

            follow_up = self.get_input()

            state.original_queries.append(follow_up)

    # ================================================================
    # COGNITION
    # ================================================================

    def handle_cognition(self, state):

        self.print_section("Understanding Request")

        while True:

            state.query_classification.requires_follow_up = False

            with Status(
                "[cyan]Analyzing your request...[/cyan]",
                console=self.console,
                spinner="dots",
            ):
                state = self.cognition_graph.execute(state)

            if DEBUG_MODE:
                save_debug_state(
                    state,
                    "cognition",
                )

            if not state.query_classification.requires_follow_up:
                return state

            self.print_ai(state.query_classification.generated_follow_up_response)

            state.multi_turn_generated_responses.append(
                state.query_classification.generated_follow_up_response
            )

            self.append_hist(state)

            follow_up = self.get_input()

            state.original_queries.append(follow_up)

            # Deeper clarification
            state = self.clarification_loop(state)

            # Reset cleanly
            state = self.new_state_from(state)

    # ================================================================
    # VALIDATION
    # ================================================================

    def print_validation_header(self):

        self.console.print()

        self.console.print(
            Panel(
                "[bold yellow]Human approval required[/bold yellow]\n\n"
                "OS-HOPE will not execute the proposed commands "
                "until you provide your response.",
                title="🛡️  Human Oversight",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    def validation_loop(self, state: OSHopeState):

        while True:

            state = self.planning_graph.execute(state)

            if not state.user_validation.is_validation_required:
                break

            self.print_validation_header()

            self.print_ai(state.user_validation.generated_response)

            state.multi_turn_generated_responses.append(
                state.user_validation.generated_response
            )

            self.append_hist(state)

            follow_up = self.get_input()

            state.original_queries.append(follow_up)

            if DEBUG_MODE:
                save_debug_state(
                    state,
                    "validation",
                )

        return state

    # ================================================================
    # PLANNING DISPLAY
    # ================================================================

    def _risk_style(self, risk):
        """Return Rich styling information for a command risk level."""

        risk = str(risk).lower()

        risk_styles = {
            "low": {
                "style": "green",
                "icon": "🟢",
            },
            "medium": {
                "style": "yellow",
                "icon": "🟡",
            },
            "high": {
                "style": "red",
                "icon": "🔴",
            },
        }

        return risk_styles.get(
            risk,
            {
                "style": "dim",
                "icon": "⚪",
            },
        )

    def _display_variables(
        self,
        variables,
        title,
    ):
        """
        Display command input/output variables.

        Returns immediately if there are no variables.
        """

        if not variables:
            return

        table = Table(
            show_header=True,
            header_style="bold cyan",
            box=None,
            padding=(0, 1),
        )

        table.add_column(
            "Variable",
            style="bold",
            no_wrap=True,
        )

        table.add_column(
            "Description",
        )

        for variable in variables:
            table.add_row(
                f"${variable.variable_name}",
                variable.description,
            )

        self.console.print(
            Panel(
                table,
                title=f"[bold]{title}[/bold]",
                border_style="dim",
                padding=(0, 1),
            )
        )

    def _display_command_step(
        self,
        step,
        step_number,
    ):
        """Render a command step."""

        command = step.step_details

        risk = self._risk_style(command.safety_risk)

        # ------------------------------------------------------------
        # Metadata
        # ------------------------------------------------------------

        metadata = Table.grid(padding=(0, 2))

        metadata.add_column(
            style="dim",
            justify="right",
        )

        metadata.add_column(
            style="bold",
        )

        metadata.add_row(
            "Type",
            "Command",
        )

        metadata.add_row(
            "Risk",
            (
                f"[{risk['style']}]"
                f"{risk['icon']} "
                f"{str(command.safety_risk).capitalize()}"
                f"[/{risk['style']}]"
            ),
        )

        metadata.add_row(
            "Execution",
            ("⏳ Blocking" if command.execution_mode == "blocking" else "↗ Background"),
        )

        if step.requires_iteration:
            metadata.add_row(
                "Iteration",
                "[yellow]↻ Multiple iterations[/yellow]",
            )

        # ------------------------------------------------------------
        # Dependencies
        # ------------------------------------------------------------

        if step.dependencies_required:

            if step.dependency_step_indices:

                dependency_text = ", ".join(
                    f"Step {index + 1}" for index in step.dependency_step_indices
                )

            else:
                dependency_text = "Previous step(s)"

            metadata.add_row(
                "Dependencies",
                f"[yellow]{dependency_text}[/yellow]",
            )

        else:

            metadata.add_row(
                "Dependencies",
                "[green]None[/green]",
            )

        # ------------------------------------------------------------
        # Command
        # ------------------------------------------------------------

        command_panel = Panel(
            Text(
                command.command,
                style="bold",
            ),
            title="[bold cyan]Command[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

        # ------------------------------------------------------------
        # Details
        # ------------------------------------------------------------

        details = Table.grid(padding=(0, 1))

        details.add_column(
            style="bold",
            width=18,
        )

        details.add_column()

        details.add_row(
            "What it does",
            command.description,
        )

        expected_output = (
            command.expected_output if command.expected_output else "No output"
        )

        details.add_row(
            "Expected output",
            expected_output,
        )

        # ------------------------------------------------------------
        # Step panel
        # ------------------------------------------------------------

        border_style = {
            "high": "red",
            "medium": "yellow",
            "low": "cyan",
        }.get(
            str(command.safety_risk).lower(),
            "dim",
        )

        self.console.print(
            Panel(
                Group(
                    metadata,
                    Text(""),
                    command_panel,
                    Text(""),
                    details,
                ),
                title=(
                    f"[bold white]Step {step_number}[/bold white]  "
                    f"[dim]{step.description}[/dim]"
                ),
                border_style=border_style,
                padding=(1, 1),
            )
        )

        # ------------------------------------------------------------
        # Variables
        # ------------------------------------------------------------

        self._display_variables(
            command.input_variables,
            "Input Variables",
        )

        self._display_variables(
            command.output_variables,
            "Output Variables",
        )

    def _display_information_step(
        self,
        step,
        step_number,
    ):
        """Render an information step."""

        details = step.step_details

        metadata = Table.grid(padding=(0, 2))

        metadata.add_column(
            style="dim",
            justify="right",
        )

        metadata.add_column(
            style="bold",
        )

        metadata.add_row(
            "Type",
            "Information",
        )

        if step.requires_iteration:
            metadata.add_row(
                "Iteration",
                "[yellow]↻ Multiple iterations[/yellow]",
            )

        # ------------------------------------------------------------
        # Dependencies
        # ------------------------------------------------------------

        if step.dependencies_required:

            if step.dependency_step_indices:

                dependency_text = ", ".join(
                    f"Step {index + 1}" for index in step.dependency_step_indices
                )

            else:
                dependency_text = "Previous step(s)"

            metadata.add_row(
                "Dependencies",
                f"[yellow]{dependency_text}[/yellow]",
            )

        else:

            metadata.add_row(
                "Dependencies",
                "[green]None[/green]",
            )

        # ------------------------------------------------------------
        # Information
        # ------------------------------------------------------------

        information_panel = Panel(
            Markdown(details.description),
            title="[bold cyan]Information[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )

        # ------------------------------------------------------------
        # Step panel
        # ------------------------------------------------------------

        self.console.print(
            Panel(
                Group(
                    metadata,
                    Text(""),
                    information_panel,
                ),
                title=(
                    f"[bold white]Step {step_number}[/bold white]  "
                    f"[dim]{step.description}[/dim]"
                ),
                border_style="cyan",
                padding=(1, 1),
            )
        )

    def display_plan(self, state):
        """
        Display the structured execution plan directly from
        state.planning.

        This function is responsible only for terminal presentation.
        It does not generate the conversation-history representation.
        """

        planning = state.planning

        self.console.print()

        # ============================================================
        # PLAN HEADER
        # ============================================================

        self.console.print(
            Panel(
                "[bold yellow]The following plan will be used to "
                "fulfill your request.[/bold yellow]\n\n"
                "[dim]Review each step, command, dependency, "
                "execution mode, and safety level carefully.[/dim]",
                title="[bold yellow]📋 Proposed Plan[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # ============================================================
        # FULFILLMENT SUMMARY
        # ============================================================

        if planning.fulfillment_summary:

            self.console.print(
                Panel(
                    Markdown(planning.fulfillment_summary),
                    title="[bold cyan]Plan Summary[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )

        # ============================================================
        # STEPS
        # ============================================================

        if not planning.plan_steps:

            self.console.print(
                Panel(
                    "[yellow]No execution steps were generated.[/yellow]",
                    title="[bold]Plan[/bold]",
                    border_style="yellow",
                )
            )

        else:

            for index, step in enumerate(
                planning.plan_steps,
                start=1,
            ):

                if step.step_type == "command":

                    self._display_command_step(
                        step,
                        index,
                    )

                elif step.step_type == "information":

                    self._display_information_step(
                        step,
                        index,
                    )

        # ============================================================
        # EXECUTION STRATEGY
        # ============================================================

        if self.parallel_execution_enabled:

            execution_mode = (
                "[bold cyan]⚡ Parallel Execution[/bold cyan]\n\n"
                "[dim]"
                "Independent steps may execute concurrently, "
                "while dependency relationships are preserved."
                "[/dim]"
            )

        else:

            execution_mode = (
                "[bold yellow]→ Sequential Execution[/bold yellow]\n\n"
                "[dim]"
                "Steps will execute one after another according "
                "to their dependencies."
                "[/dim]"
            )

        self.console.print(
            Panel(
                execution_mode,
                title="[bold]Execution Strategy[/bold]",
                border_style="dim",
                padding=(1, 2),
            )
        )

        # ============================================================
        # HUMAN OVERSIGHT
        # ============================================================

        self.console.print()

        self.console.print(
            Panel(
                "[bold yellow]Human approval is required.[/bold yellow]\n\n"
                "OS-HOPE will not execute the proposed commands "
                "without your approval.\n\n"
                "[dim]"
                "You can approve the plan, request changes, "
                "or cancel the request."
                "[/dim]",
                title="[bold yellow]🛡️ Human Oversight[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        self.console.print()

    # ================================================================
    # PLANNING
    # ================================================================

    def handle_planning(self, state):

        self.print_section("Planning")

        while True:

            with Status(
                "[cyan]Generating execution plan...[/cyan]",
                console=self.console,
                spinner="dots",
            ):
                state = self.planning_graph.execute(state)

            if DEBUG_MODE:
                save_debug_state(
                    state,
                    "planning",
                )

            # --------------------------------------------------------
            # Clarification required
            # --------------------------------------------------------

            if state.planning.requires_follow_up:

                state.query_clarification.is_clarification_needed = True

                state = self.clarification_loop(state)

                state = self.new_state_from(state)

                continue

            # --------------------------------------------------------
            # Information query
            # --------------------------------------------------------

            if state.query_classification.query_type == "information":
                return state

            # --------------------------------------------------------
            # Present the generated plan
            # --------------------------------------------------------

            self.display_plan(state)

            state.plan_presented = True

            # --------------------------------------------------------
            # Store plan in conversation history
            #
            # IMPORTANT:
            # format_plan_for_user() is NOT used for terminal
            # presentation. It is retained only because the
            # conversation history expects a text response.
            # --------------------------------------------------------

            plan_str = format_plan_for_user(state.planning)

            state.multi_turn_generated_responses.append(plan_str)

            self.append_hist(state)

            # --------------------------------------------------------
            # Ask for human decision
            # --------------------------------------------------------

            self.console.print(
                Panel(
                    "[bold green]Approve[/bold green]  "
                    "→ execute this plan\n"
                    "[bold yellow]Update[/bold yellow]   "
                    "→ modify the plan\n"
                    "[bold red]Cancel[/bold red]   "
                    "→ stop this request",
                    title="[bold]Plan Decision[/bold]",
                    border_style="dim",
                    padding=(1, 2),
                )
            )

            follow_up = self.get_input()

            state.original_queries.append(follow_up)

            # --------------------------------------------------------
            # Validation
            # --------------------------------------------------------

            state = self.validation_loop(state)

            # --------------------------------------------------------
            # User requested a plan update
            # --------------------------------------------------------

            if state.user_validation.user_feedback_type == "update_plan":

                state.plan_presented = False

                self.print_info("Updating the plan based on your feedback...")

                continue

            # --------------------------------------------------------
            # Plan approved
            # --------------------------------------------------------

            return state

    # ================================================================
    # EXECUTION
    # ================================================================

    def handle_execution(self, state):

        self.print_section("Execution")

        mode = "parallel" if self.parallel_execution_enabled else "sequential"

        self.console.print(f"[dim]Execution mode: " f"[bold]{mode}[/bold][/dim]")

        self.console.print()

        with Status(
            "[cyan]Executing approved plan...[/cyan]",
            console=self.console,
            spinner="dots",
        ):
            state = self.execution_graph.execute(state)

        self.print_success("Execution completed.")

        self.print_ai(state.generated_final_response)

        state.multi_turn_generated_responses.append(state.generated_final_response)

        self.append_hist(state)

        if DEBUG_MODE:
            save_debug_state(
                state,
                "execution",
            )

        return state

    # ================================================================
    # MEMORY
    # ================================================================

    def handle_memory(self, state):

        with Status(
            "[cyan]Updating conversation memory...[/cyan]",
            console=self.console,
            spinner="dots",
        ):
            state = self.memory_graph.execute(state)

        if DEBUG_MODE:
            save_debug_state(
                state,
                "memory",
            )

        return state

    # ================================================================
    # RAG
    # ================================================================

    def handle_rag(self, state):

        self.rag_tool.add_memories(
            session_id=state.turn_num,
            summaries=state.memory_extraction.summary_for_rag,
        )

        return state

    # ================================================================
    # FINAL SUMMARY
    # ================================================================

    def display_execution_summary(self, state):

        self.pause_timer()

        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
        )

        table.add_column(style="dim")

        table.add_column(style="bold")

        table.add_row(
            "Execution mode",
            ("⚡ Parallel" if self.parallel_execution_enabled else "→ Sequential"),
        )

        table.add_row(
            "Elapsed time",
            f"{state.execution_time:.2f}s",
        )

        if RAG_ENABLED:

            table.add_row(
                "Memory",
                "✓ Updated",
            )

        self.console.print()

        self.console.print(
            Panel(
                table,
                title="[bold green]✓ OS-HOPE Complete[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

        self.console.print()

    # ================================================================
    # MAIN LOOP
    # ================================================================

    def run(self):

        past_session_summaries = []

        self.print_banner()

        while True:

            self.reset_timer()

            # --------------------------------------------------------
            # User query
            # --------------------------------------------------------

            query = self.get_input()

            if not query:

                self.print_warning("Please enter a request.")

                continue

            self.resume_timer()

            # --------------------------------------------------------
            # Create state
            # --------------------------------------------------------

            state = OSHopeState()

            state.parallel_execution_enabled = self.parallel_execution_enabled

            state.past_session_summaries = past_session_summaries

            state.original_queries.append(query)

            # --------------------------------------------------------
            # Cognition
            # --------------------------------------------------------

            state = self.handle_cognition(state)

            # --------------------------------------------------------
            # Planning
            # --------------------------------------------------------

            state = self.handle_planning(state)

            # --------------------------------------------------------
            # Execution
            # --------------------------------------------------------

            state = self.handle_execution(state)

            # --------------------------------------------------------
            # Memory
            # --------------------------------------------------------

            if not self.parallel_execution_enabled:

                state = self.handle_memory(state)

            # --------------------------------------------------------
            # RAG
            # --------------------------------------------------------

            if RAG_ENABLED:

                state = self.handle_rag(state)

            # --------------------------------------------------------
            # Session summary
            # --------------------------------------------------------

            past_session_summaries.append(state.memory_extraction.session_summary)

            # --------------------------------------------------------
            # Timing
            # --------------------------------------------------------

            self.pause_timer()

            state.execution_time = self.execution_time

            # --------------------------------------------------------
            # Final debug state
            # --------------------------------------------------------

            if DEBUG_MODE:

                save_debug_state(
                    state,
                    "final_state",
                )

            # --------------------------------------------------------
            # Summary
            # --------------------------------------------------------

            self.display_execution_summary(state)


# ====================================================================
# ARGUMENT PARSING
# ====================================================================


def main():

    parser = argparse.ArgumentParser(description="OS-HOPE CLI")

    parser.add_argument(
        "--voice",
        action="store_true",
        help="Enable voice input",
    )

    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Disable voice input",
    )

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

    # ================================================================
    # VOICE
    # ================================================================

    if args.voice:

        use_voice = True

    elif args.no_voice:

        use_voice = False

    else:

        use_voice = False

    # ================================================================
    # EXECUTION MODE
    # ================================================================

    if args.parallel:

        use_parallel = True

    elif args.no_parallel:

        use_parallel = False

    else:

        use_parallel = PARALLEL_EXECUTION_ENABLED

    # ================================================================
    # START APPLICATION
    # ================================================================

    app = OSHopeApp(
        use_voice=use_voice,
        parallel_execution_enabled=use_parallel,
    )

    app.run()


if __name__ == "__main__":
    main()
