from oshope.tools.command_execution import run_command
from dataclasses import dataclass, field
from typing import List
import threading


@dataclass
class CommandRequest:

    step_index: int
    commands: List[str]
    execution_modes: List[str]

    result: List[tuple[bool, str, str]] = field(default_factory=list)
    error: Exception = None

    event: threading.Event = field(default_factory=threading.Event)


class CommandBatchCoordinator:

    def __init__(self, expected_size: int):

        self.expected_size = expected_size

        self.lock = threading.Lock()

        self.requests = {}

        self.all_submitted = threading.Event()

        self.results = {}

    def submit(self, step_index: int, commands: List[str], execution_modes: List[str]):

        request = CommandRequest(
            step_index=step_index,
            commands=commands,
            execution_modes=execution_modes,
        )

        with self.lock:
            self.requests[step_index] = request

            if len(self.requests) == self.expected_size:
                self.all_submitted.set()

        # wait for execution result
        request.event.wait()

        if request.error:
            raise request.error

        return self.requests[step_index].result

    def execute_all(self):

        # wait until all commands submitted
        self.all_submitted.wait()

        # execute in strict order
        for step_index in sorted(self.requests.keys()):

            request: CommandRequest = self.requests[step_index]

            try:
                results = []
                for i in range(len(request.commands)):
                    if request.commands[i] != "ignore":
                        success, output, error = run_command(
                            request.commands[i],
                            request.execution_modes[i],
                        )
                    else:
                        success = True
                        output = "ignored"
                        error = "no errors"

                    results.append((success, output, error))

                self.requests[step_index].result = results

            except Exception as e:
                request.error = e

            finally:
                request.event.set()
