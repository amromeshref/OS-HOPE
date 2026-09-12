from oshope_evaluation.runners.pydantic_schemas import DataPointEvaluation
from oshope.core.states.oshope_state import OSHopeState
from datetime import datetime
from typing import Literal
from pathlib import Path
import pandas as pd
import json

PARENT_DIR = Path(__file__).parent.parent.parent.parent
OS_ASISTANT_OUTPUT_DIR = (
    PARENT_DIR / "src/oshope_evaluation/outputs/os-assistant-output"
)
EVALUATION_OUTPUT_DIR = PARENT_DIR / "src/oshope_evaluation/outputs/evaluation-output"
REPORTS_DIR = PARENT_DIR / "src/oshope_evaluation/reports"

evaluation_files = list(EVALUATION_OUTPUT_DIR.glob("*.json"))
# print(evaluation_files)


class EvaluationAnalysis:
    def __init__(self):

        ######## Classification Evaluation Metrics ########

        # Query type prediction metrics
        self.correct_query_type_prediction_count = 0
        self.query_type_prediction_score = 0.0

        # Clarification prediction metrics
        self.correct_requires_clarification_prediction_count = 0
        self.requires_clarification_prediction_score = 0.0

        self.classification_score = 0.0

        ######## Clarification Evaluation Metrics ########
        self.total_clarification_needed_count = 0
        self.clarification_score = 0.0

        ######## Planning Evaluation Metrics ########

        # Step correctness metrics
        self.total_generated_steps_count = 0
        self.total_correctly_generated_steps_count = 0
        self.step_correctness_score = 0.0

        # Step completeness metrics
        self.total_required_steps_count = 0
        self.total_fulfilling_steps_count = 0
        self.step_completion_coverage_score = 0.0

        # Dependency accuracy metrics
        self.total_generated_dependency_steps_count = 0
        self.total_correctly_generated_dependency_steps_count = 0
        self.dependency_accuracy_score = 0.0

        # Step type accuracy metrics
        self.total_correctly_generated_step_types_count = 0
        self.step_type_accuracy_score = 0.0

        # Iteration handling metrics
        self.total_generated_steps_require_iteration_count = 0
        self.correctly_generated_iterative_steps_count = 0
        self.iteration_handling_accuracy_score = 0.0

        # Redundancy metrics
        self.total_redundant_steps_count = 0
        self.redundancy_score = 0.0

        self.planning_score = 0.0

        ######## User Validation Evaluation Metrics ########
        self.total_user_validations_count = 0
        self.user_validation_score = 0.0

        ######## Step Resolution Evaluation Metrics ########

        # Resolution success metrics
        self.total_step_resolution_count = 0
        self.total_correctly_predicted_resolution_success = 0
        self.resolution_success_accuracy_score = 0.0

        # Placeholder metrics
        self.total_generated_placeholders_count = 0
        self.total_correctly_resolved_placeholders_count = 0
        self.placeholder_resolution_accuracy_score = 0.0

        # Resolved steps count metrics
        self.total_expected_generated_resolved_steps_count = 0
        self.total_actual_generated_resolved_steps_count = 0
        self.resolved_steps_count_accuracy_score = 0.0

        self.step_resolution_score = 0.0

        ######## Information Generation Evaluation Metrics ########
        self.total_generated_information_responses_count = 0
        self.information_generation_score = 0.0

        ######## Command Execution Evaluation Metrics ########

        # Execution success prediction metrics
        self.total_command_executions_count = 0
        self.total_correctly_predicted_execution_success_count = 0
        self.execution_success_prediction_score = 0.0

        self.command_execution_score = 0.0

        ######## Command Error Handling Evaluation Metrics ########

        # Can recover prediction metrics
        self.total_command_errors_count = 0
        self.total_correctly_predicted_can_recover_count = 0
        self.can_recover_prediction_score = 0.0

        # Recovery command execution success metrics
        self.total_generated_recoveries_count = 0
        self.total_succesfully_executed_recoveries_count = 0
        self.successfully_executed_recoveries_score = 0.0

        self.command_error_handling_score = 0.0

        ######### Final Response Evaluation Metrics ########
        self.final_response_score = 0.0

        ######### Overall Evaluation Metrics ########
        self.overall_score = 0.0

        ######### Additional Metrics ########
        self.average_execution_time_parallel = 0.0
        self.average_execution_time_sequential = 0.0

    def load_evaluation_results(self, path: Path) -> DataPointEvaluation:
        with open(path, "r") as f:
            evaluation_data = json.load(f)

        return DataPointEvaluation.model_validate(evaluation_data)

    def load_os_assistant_output(self, path: Path) -> OSHopeState:
        with open(path, "r") as f:
            os_assistant_output = json.load(f)

        return OSHopeState.model_validate(os_assistant_output)

    def analyze_classification_evaluation(self, eval_data: DataPointEvaluation):
        self.correct_query_type_prediction_count += int(
            eval_data.classification_evaluation.is_query_type_correct
        )
        self.correct_requires_clarification_prediction_count += int(
            eval_data.classification_evaluation.is_clarification_prediction_correct
        )

    def analyze_clarification_evaluation(self, eval_data: DataPointEvaluation):
        if eval_data.clarification_evaluation.evaluation_needed:
            self.total_clarification_needed_count += 1
            self.clarification_score += (
                eval_data.clarification_evaluation.evaluation_score
            )

    def analyze_planning_evaluation(self, eval_data: DataPointEvaluation):
        # Step correctness metrics
        self.total_generated_steps_count += (
            eval_data.planning_evaluation.generated_steps_count
        )
        self.total_correctly_generated_steps_count += (
            eval_data.planning_evaluation.correctly_generated_steps_count
        )

        # Step completeness metrics
        self.total_required_steps_count += (
            eval_data.planning_evaluation.num_required_steps
        )
        self.total_fulfilling_steps_count += (
            eval_data.planning_evaluation.num_fulfilling_steps
        )

        # Dependency accuracy metrics
        self.total_generated_dependency_steps_count += (
            eval_data.planning_evaluation.total_generated_dependency_steps
        )
        self.total_correctly_generated_dependency_steps_count += (
            eval_data.planning_evaluation.correctly_generated_dependency_steps
        )

        # Step type accuracy metrics
        self.total_correctly_generated_step_types_count += (
            eval_data.planning_evaluation.correctly_generated_information_steps_count
            + eval_data.planning_evaluation.correctly_generated_command_steps_count
        )

        # Iteration handling metrics
        self.total_generated_steps_require_iteration_count += (
            eval_data.planning_evaluation.total_generated_steps_require_iteration
        )
        self.correctly_generated_iterative_steps_count += (
            eval_data.planning_evaluation.correctly_generated_iterative_steps
        )

        # Redundancy metrics
        self.total_redundant_steps_count += (
            eval_data.planning_evaluation.num_redundant_steps
        )

    def analyze_user_validation_evaluation(self, eval_data: DataPointEvaluation):
        if eval_data.user_validation_evaluation.evaluation_needed:
            self.total_user_validations_count += 1
            self.user_validation_score += (
                eval_data.user_validation_evaluation.evaluation_score
            )

    def analyze_step_resolution_evaluation(self, eval_data: DataPointEvaluation):
        for step_resolution_eval in eval_data.step_resolver_evaluation:
            if step_resolution_eval.evaluation_needed:
                # Resolution success metrics
                self.total_step_resolution_count += 1
                self.total_correctly_predicted_resolution_success += int(
                    (
                        step_resolution_eval.expected_resolution_success
                        == step_resolution_eval.generated_resolution_success
                    )
                )

                # Placeholder metrics
                self.total_generated_placeholders_count += (
                    step_resolution_eval.total_generated_placeholders
                )
                self.total_correctly_resolved_placeholders_count += (
                    step_resolution_eval.correctly_resolved_placeholders
                )

                # Resolved steps count metrics
                self.total_expected_generated_resolved_steps_count += (
                    step_resolution_eval.expected_generated_resolved_steps_count
                )
                self.total_actual_generated_resolved_steps_count += (
                    step_resolution_eval.actual_generated_resolved_steps_count
                )

    def analyze_information_generation_evaluation(self, eval_data: DataPointEvaluation):
        for info_gen_eval in eval_data.information_generation_evaluation:
            if info_gen_eval.evaluation_needed:
                self.total_generated_information_responses_count += 1
                self.information_generation_score += info_gen_eval.evaluation_score

    def analyze_command_execution_evaluation(self, eval_data: DataPointEvaluation):
        for cmd_exec_eval in eval_data.command_execution_analysis_evaluation:
            if cmd_exec_eval.evaluation_needed:
                self.total_command_executions_count += 1
                self.total_correctly_predicted_execution_success_count += int(
                    (cmd_exec_eval.expected_success == cmd_exec_eval.predicted_success)
                )

    def analyze_command_error_handling_evaluation(self, eval_data: DataPointEvaluation):
        for cmd_error_eval in eval_data.command_error_handling_evaluation:
            if cmd_error_eval.evaluation_needed:
                self.total_command_errors_count += 1
                self.total_correctly_predicted_can_recover_count += int(
                    (
                        cmd_error_eval.expected_can_recover
                        == cmd_error_eval.predicted_can_recover
                    )
                )
                self.total_succesfully_executed_recoveries_count += int(
                    cmd_error_eval.recovery_command_executed_successfully
                )

                if cmd_error_eval.predicted_can_recover:
                    self.total_generated_recoveries_count += 1

    def analyze_final_response_evaluation(self, eval_data: DataPointEvaluation):
        self.final_response_score += (
            eval_data.final_response_evaluation.evaluation_score
        )

    def compute_scores(self, num_data_points: int):
        # Classification
        self.query_type_prediction_score = (
            self.correct_query_type_prediction_count / num_data_points
            if num_data_points > 0
            else 0.0
        )
        self.requires_clarification_prediction_score = (
            self.correct_requires_clarification_prediction_count / num_data_points
            if num_data_points > 0
            else 0.0
        )
        self.classification_score = (
            self.query_type_prediction_score
            + self.requires_clarification_prediction_score
        ) / 2

        # Clarification
        self.clarification_score = (
            self.clarification_score / self.total_clarification_needed_count
            if self.total_clarification_needed_count > 0
            else 0.0
        )
        self.clarification_score /= 10

        # Planning
        self.step_correctness_score = (
            self.total_correctly_generated_steps_count
            / self.total_generated_steps_count
            if self.total_generated_steps_count > 0
            else 0.0
        )
        self.step_completion_coverage_score = (
            self.total_fulfilling_steps_count / self.total_required_steps_count
            if self.total_required_steps_count > 0
            else 0.0
        )
        self.dependency_accuracy_score = (
            self.total_correctly_generated_dependency_steps_count
            / self.total_generated_dependency_steps_count
            if self.total_generated_dependency_steps_count > 0
            else 0.0
        )
        self.step_type_accuracy_score = (
            self.total_correctly_generated_step_types_count
            / self.total_generated_steps_count
            if self.total_generated_steps_count > 0
            else 0.0
        )
        self.iteration_handling_accuracy_score = (
            self.correctly_generated_iterative_steps_count
            / self.total_generated_steps_require_iteration_count
            if self.total_generated_steps_require_iteration_count > 0
            else 0.0
        )
        self.redundancy_score = (
            (1 - (self.total_redundant_steps_count / self.total_generated_steps_count))
            if self.total_generated_steps_count > 0
            else 0.0
        )
        self.planning_score = (
            self.step_correctness_score
            + self.step_completion_coverage_score
            + self.dependency_accuracy_score
            + self.step_type_accuracy_score
            + self.iteration_handling_accuracy_score
            + self.redundancy_score
        ) / 6

        # User Validation
        self.user_validation_score = (
            self.user_validation_score / self.total_user_validations_count
            if self.total_user_validations_count > 0
            else 0.0
        )
        self.user_validation_score /= 10

        # Step Resolution
        self.resolution_success_accuracy_score = (
            self.total_correctly_predicted_resolution_success
            / self.total_step_resolution_count
            if self.total_step_resolution_count > 0
            else 0.0
        )
        self.placeholder_resolution_accuracy_score = (
            self.total_correctly_resolved_placeholders_count
            / self.total_generated_placeholders_count
            if self.total_generated_placeholders_count > 0
            else 0.0
        )
        self.resolved_steps_count_accuracy_score = (
            self.total_actual_generated_resolved_steps_count
            / self.total_expected_generated_resolved_steps_count
            if self.total_expected_generated_resolved_steps_count > 0
            else 0.0
        )
        self.step_resolution_score = (
            self.resolution_success_accuracy_score
            + self.placeholder_resolution_accuracy_score
            + self.resolved_steps_count_accuracy_score
        ) / 3

        # Information Generation
        self.information_generation_score = (
            self.information_generation_score
            / self.total_generated_information_responses_count
            if self.total_generated_information_responses_count > 0
            else 0.0
        )
        self.information_generation_score /= 10

        # Command Execution
        self.execution_success_prediction_score = (
            self.total_correctly_predicted_execution_success_count
            / self.total_command_executions_count
            if self.total_command_executions_count > 0
            else 0.0
        )
        self.command_execution_score = self.execution_success_prediction_score

        # Command Error Handling
        self.can_recover_prediction_score = (
            self.total_correctly_predicted_can_recover_count
            / self.total_command_errors_count
            if self.total_command_errors_count > 0
            else 0.0
        )
        self.successfully_executed_recoveries_score = (
            self.total_succesfully_executed_recoveries_count
            / self.total_generated_recoveries_count
            if self.total_generated_recoveries_count > 0
            else 0.0
        )
        self.command_error_handling_score = (
            self.can_recover_prediction_score
            + self.successfully_executed_recoveries_score
        ) / 2

        # Final Response
        self.final_response_score = (
            self.final_response_score / num_data_points if num_data_points > 0 else 0.0
        )
        self.final_response_score /= 10

        # Overall Score (simple average of all category scores)
        self.overall_score = (
            self.classification_score
            + self.clarification_score
            + self.planning_score
            + self.user_validation_score
            + self.step_resolution_score
            + self.information_generation_score
            + self.command_execution_score
            + self.command_error_handling_score
            + self.final_response_score
        ) / 9

        # Execution Time
        self.average_execution_time_parallel = self.get_average_execution_time(
            mode="parallel"
        )
        self.average_execution_time_sequential = self.get_average_execution_time(
            mode="sequential"
        )

    def get_average_execution_time(
        self, mode: Literal["parallel", "sequential"] = "sequential"
    ) -> float:
        if mode == "parallel":
            file_paths = OS_ASISTANT_OUTPUT_DIR / "parallel"
            os_assistant_out_files = list(file_paths.glob("*.json"))
        elif mode == "sequential":
            file_paths = OS_ASISTANT_OUTPUT_DIR / "sequential"
            os_assistant_out_files = list(file_paths.glob("*.json"))
        else:
            raise ValueError("Invalid mode. Choose either 'parallel' or 'sequential'.")

        total_execution_time = 0.0
        count = len(os_assistant_out_files)
        for out_file in os_assistant_out_files:
            os_assistant_output = self.load_os_assistant_output(out_file)
            total_execution_time += (
                os_assistant_output.execution_time
                - os_assistant_output.total_retry_wait_time
            )

        return total_execution_time / count if count > 0 else 0.0

    def analyze_evaluation_results(self):
        evaluation_files = list(EVALUATION_OUTPUT_DIR.glob("*.json"))

        for eval_file in evaluation_files:
            eval_data = self.load_evaluation_results(eval_file)

            self.analyze_classification_evaluation(eval_data)
            self.analyze_clarification_evaluation(eval_data)
            self.analyze_planning_evaluation(eval_data)
            self.analyze_user_validation_evaluation(eval_data)
            self.analyze_step_resolution_evaluation(eval_data)
            self.analyze_information_generation_evaluation(eval_data)
            self.analyze_command_execution_evaluation(eval_data)
            self.analyze_command_error_handling_evaluation(eval_data)
            self.analyze_final_response_evaluation(eval_data)

        self.compute_scores(num_data_points=len(evaluation_files))

    def export_excel_report(self):
        """
        Export evaluation metrics into a clean Excel report.
        """

        report_data = [
            # ================= Classification =================
            {
                "Category": "Classification",
                "Metric": "Query Type Prediction Score",
                "Value": self.query_type_prediction_score,
            },
            {
                "Category": "Classification",
                "Metric": "Requires Clarification Prediction Score",
                "Value": self.requires_clarification_prediction_score,
            },
            {
                "Category": "Classification",
                "Metric": "Overall Classification Score",
                "Value": self.classification_score,
            },
            # ================= Clarification =================
            {
                "Category": "Clarification",
                "Metric": "Clarification Score",
                "Value": self.clarification_score,
            },
            # ================= Planning =================
            {
                "Category": "Planning",
                "Metric": "Step Correctness Score",
                "Value": self.step_correctness_score,
            },
            {
                "Category": "Planning",
                "Metric": "Step Completion Coverage Score",
                "Value": self.step_completion_coverage_score,
            },
            {
                "Category": "Planning",
                "Metric": "Dependency Accuracy Score",
                "Value": self.dependency_accuracy_score,
            },
            {
                "Category": "Planning",
                "Metric": "Step Type Accuracy Score",
                "Value": self.step_type_accuracy_score,
            },
            {
                "Category": "Planning",
                "Metric": "Iteration Handling Accuracy Score",
                "Value": self.iteration_handling_accuracy_score,
            },
            {
                "Category": "Planning",
                "Metric": "Redundancy Score",
                "Value": self.redundancy_score,
            },
            {
                "Category": "Planning",
                "Metric": "Overall Planning Score",
                "Value": self.planning_score,
            },
            # ================= User Validation =================
            {
                "Category": "User Validation",
                "Metric": "User Validation Score",
                "Value": self.user_validation_score,
            },
            # ================= Step Resolution =================
            {
                "Category": "Step Resolution",
                "Metric": "Resolution Success Accuracy",
                "Value": self.resolution_success_accuracy_score,
            },
            {
                "Category": "Step Resolution",
                "Metric": "Placeholder Resolution Accuracy",
                "Value": self.placeholder_resolution_accuracy_score,
            },
            {
                "Category": "Step Resolution",
                "Metric": "Resolved Steps Count Accuracy",
                "Value": self.resolved_steps_count_accuracy_score,
            },
            {
                "Category": "Step Resolution",
                "Metric": "Overall Step Resolution Score",
                "Value": self.step_resolution_score,
            },
            # ================= Information Generation =================
            {
                "Category": "Information Generation",
                "Metric": "Information Generation Score",
                "Value": self.information_generation_score,
            },
            # ================= Command Execution =================
            {
                "Category": "Command Execution",
                "Metric": "Execution Success Prediction Score",
                "Value": self.execution_success_prediction_score,
            },
            {
                "Category": "Command Execution",
                "Metric": "Overall Command Execution Score",
                "Value": self.command_execution_score,
            },
            # ================= Command Error Handling =================
            {
                "Category": "Command Error Handling",
                "Metric": "Can Recover Prediction Score",
                "Value": self.can_recover_prediction_score,
            },
            {
                "Category": "Command Error Handling",
                "Metric": "Successfully Executed Recoveries Score",
                "Value": self.successfully_executed_recoveries_score,
            },
            {
                "Category": "Command Error Handling",
                "Metric": "Overall Command Error Handling Score",
                "Value": self.command_error_handling_score,
            },
            # ================= Final Response =================
            {
                "Category": "Final Response",
                "Metric": "Final Response Score",
                "Value": self.final_response_score,
            },
            # ================= Overall =================
            {
                "Category": "Overall",
                "Metric": "Overall Score",
                "Value": self.overall_score,
            },
            # ================= Timing =================
            {
                "Category": "Execution Time",
                "Metric": "Average Parallel Execution Time",
                "Value": self.average_execution_time_parallel,
            },
            {
                "Category": "Execution Time",
                "Metric": "Average Sequential Execution Time",
                "Value": self.average_execution_time_sequential,
            },
        ]

        df = pd.DataFrame(report_data)

        timestamp = datetime.now().strftime("%Y-%m-%d-%I-%M-%S")

        report_path = REPORTS_DIR / f"evaluation_report_{timestamp}.xlsx"

        with pd.ExcelWriter(report_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Evaluation Report", index=False)

            worksheet = writer.sheets["Evaluation Report"]

            # Auto-size columns
            for column_cells in worksheet.columns:
                length = max(
                    len(str(cell.value)) if cell.value else 0 for cell in column_cells
                )
                worksheet.column_dimensions[column_cells[0].column_letter].width = (
                    length + 5
                )

        print(f"Excel report saved to: {report_path}")


def main():
    analysis = EvaluationAnalysis()
    analysis.analyze_evaluation_results()
    analysis.export_excel_report()


if __name__ == "__main__":
    main()
