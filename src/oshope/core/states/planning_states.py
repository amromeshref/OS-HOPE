from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, Union


class InformationStep(BaseModel):
    description: str = Field(
        ..., description="A human-readable description of the information to provide."
    )


class VariableDetails(BaseModel):
    variable_name: str = Field(..., description="The name of the variable.")

    description: str = Field(
        ...,
        description="A clear and concise explanation of what this variable represents and how it should be used.",
    )


class CommandStep(BaseModel):
    command: str = Field(..., description="The command to execute.")

    description: str = Field(
        ...,
        description="A short, human-readable explanation of what this command does.",
    )

    expected_output: str = Field(
        default="",
        description="What the command is expected to produce. If the command produce no output. set it as 'no output'",
    )

    safety_risk: Literal["low", "medium", "high"] = Field(
        ..., description="The safety risk level of executing this command."
    )

    input_variables: List[VariableDetails] = Field(
        default_factory=list,
        description=(
            "A list of input variables required for executing the command, along with their descriptions. "
            "If the command does not require any input variables, this should be an empty list."
        ),
    )

    output_variables: List[VariableDetails] = Field(
        default_factory=list,
        description=(
            "A list of output variables produced by executing the command, along with their descriptions. "
            "If the command does not produce any output variables, this should be an empty list."
        ),
    )

    execution_mode: Literal["blocking", "background"] = Field(
        ...,
        description=(
            "The execution mode of the command, either 'blocking' or 'background'."
            "'blocking' should be used when the command must complete and its output is needed for subsequent steps. "
            "'background' should be used when the command starts a long-running or GUI process and does not need to block execution of subsequent steps."
        ),
    )


class Step(BaseModel):
    description: str = Field(
        ..., description="A human-readable description of the step."
    )

    step_type: Literal["information", "command"] = Field(
        ..., description="The type of the step, either 'information' or 'command'."
    )

    step_details: Union[InformationStep, CommandStep] = Field(
        ...,
        description="The details of the step, which can be either information details or command details based on the step type.",
    )

    requires_iteration: bool = Field(
        default=False,
        description=(
            "Indicates if this step requires multiple iterations to complete. "
            "Set to true if the step needs to be executed multiple times with different inputs. Otherwise, set to false."
        ),
    )

    dependencies_required: bool = Field(
        default=False,
        description=(
            "Indicates if this step has dependencies on the outputs of previous steps. "
            "Set to true if the step requires outputs from previous steps to be executed or fulfilled. Otherwise, set to false."
        ),
    )

    dependency_step_indices: List[int] = Field(
        default_factory=list,
        description=(
            "A list of indices of the previous steps that this step depends on. "
            "This should only be populated if dependencies_required is true. Otherwise, this should be an empty list."
        ),
    )

    status: Literal["pending", "running", "completed", "failed"] = Field(
        default="pending", description="Execution status of the step."
    )

    model_config = ConfigDict(extra="allow")


class PlanningState(BaseModel):
    """
    Represents the execution plan generated for a user query.
    """

    fulfillment_summary: str = Field(
        default="",
        description=(
            "A high-level explanation of how the user's request will be fulfilled. "
            "This may describe reasoning steps, conceptual answers, or execution logic."
        ),
    )

    plan_steps: List[Step] = Field(
        default_factory=list,
        description=(
            "A list of steps that outline the plan for fulfilling the user's request. Each step includes a description, type (information or command), and relevant details. "
            "The steps should be ordered in the sequence they will be executed or fulfilled."
        ),
    )

    requires_follow_up: bool = Field(
        default=False,
        description=(
            "Indicates if the plan requires a follow-up question for clarification "
            "This field should be set to true if any required information for performing the plan is missing, and we need to ask the user for clarification before proceeding."
        ),
    )

    follow_up_reasoning: str = Field(
        default="",
        description=(
            "The reasoning for why a follow-up question is needed, if applicable. "
            "This should explain what information is missing and why it's necessary for executing the plan."
        ),
    )

    # generated_follow_up_response: str = Field(
    #     default=None,
    #     description="The response generated by the LLM for follow-up purposes, if applicable. It should be sent to the user to ask for clarification."
    # )


class UserValidationState(BaseModel):
    user_feedback_type: Literal[
        "approved", "rejected", "needs_clarification", "update_plan", "pending"
    ] = Field(
        default="pending",
        description=(
            "The user's feedback on the generated plan after validation. "
            "It can be 'approved' if the user approves the plan, "
            "'rejected' if the user rejects the plan, "
            "'needs_clarification' if the user asks for clarification for the current plan without providing specific feedback on what needs to be clarified, "
            "or 'update_plan' if the user provides specific feedback on what needs to be updated in the plan. "
            "`pending` indicates that the user has not responded to the validation question yet, so we cannot proceed without the user's feedback on the plan."
        ),
    )

    user_feedback: List[str] = Field(
        default_factory=list,
        description=(
            "A list of specific feedback points provided by the user regarding what needs to be updated in the plan. "
            "This feedback would be used to update the current plan."
        ),
    )

    is_validation_required: bool = Field(
        default=False,
        description=(
            "Indicates whether validation process is finished and we can proceed to execute the plan or not. "
            "This should be set to true when the user approves the plan, or if the user rejects the plan but does not provide any specific feedback on what needs to be updated (in this case, we can consider the user's rejection as a disapproval of the current plan without asking for an updated plan, so we can end the validation process and proceed to generate a final response based on the user's disapproval). "
            "This should be set to false if the user asks for clarification or provides specific feedback on what needs to be updated, since in both cases we need to update the plan and ask for validation again until we get an approval or a rejection without specific feedback. "
            "This should also be set to false if the user did not respond to the validation question yet, since we cannot proceed without the user's feedback on the plan."
        ),
    )

    validation_reasoning: str = Field(
        default="",
        description=("The reasoning behind why validation is required or not."),
    )

    generated_response: str = Field(
        default="",
        description=(
            "The response generated by the LLM for user validation purposes. It should be sent to the user to validate the generated execution/information plan. "
            "If the user asked for clarification for the current plan, this field should contain a response to answer the user's clarification question. Otherwise, it should contain a response to validate the current plan without asking any clarification questions."
        ),
    )
