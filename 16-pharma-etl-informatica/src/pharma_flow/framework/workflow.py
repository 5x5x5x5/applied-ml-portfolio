"""
Informatica Workflow Manager equivalent.

A Workflow orchestrates the execution of multiple Sessions (and other tasks)
with dependency management, conditional branching, timers, and error handling.

Task types (mirroring Informatica Workflow Manager):
  - SessionTask: Execute an ETL session
  - DecisionTask: Conditional branching based on variables/conditions
  - TimerTask: Wait until a scheduled time or for a duration
  - EventWaitTask: Wait for an external event (file arrival, signal)
  - CommandTask: Execute a shell command
  - EmailTask: Send email notification
  - AssignmentTask: Set workflow variable values

Workflows support:
  - Sequential and parallel execution paths
  - Link conditions (success, failure, expression)
  - Fail-over paths
  - Workflow variables
  - Suspend/resume
"""

from __future__ import annotations

import smtplib
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import UTC, datetime
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

import structlog

from pharma_flow.framework.session import PerformanceStats, Session, SessionStatus

logger = structlog.get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"
    DISABLED = "disabled"


class LinkCondition(str, Enum):
    """Condition type for task links (like Informatica link conditions)."""

    ON_SUCCESS = "on_success"
    ON_FAILURE = "on_failure"
    UNCONDITIONAL = "unconditional"
    EXPRESSION = "expression"


@dataclass
class TaskLink:
    """Directed link between two tasks with a condition."""

    from_task: str
    to_task: str
    condition: LinkCondition = LinkCondition.ON_SUCCESS
    expression: str = ""  # Used when condition == EXPRESSION


# ---------------------------------------------------------------------------
# Abstract Task
# ---------------------------------------------------------------------------


class WorkflowTask(ABC):
    """Base class for all workflow tasks."""

    name: str
    status: TaskStatus
    enabled: bool
    retry_count: int
    retry_interval_sec: int

    @abstractmethod
    def run(self, context: WorkflowContext) -> TaskStatus:
        """Execute the task and return its final status."""
        ...


@dataclass
class WorkflowContext:
    """
    Shared context passed to all tasks in a workflow.

    Holds workflow variables, task outputs, and global configuration.
    """

    variables: dict[str, Any] = field(default_factory=dict)
    task_results: dict[str, TaskStatus] = field(default_factory=dict)
    task_outputs: dict[str, Any] = field(default_factory=dict)
    workflow_name: str = ""
    start_time: datetime | None = None
    smtp_server: str = "localhost"
    smtp_port: int = 25
    notification_email_from: str = "pharmaflow@company.com"
    notification_email_to: list[str] = field(default_factory=list)

    def set_variable(self, name: str, value: Any) -> None:
        self.variables[name] = value

    def get_variable(self, name: str, default: Any = None) -> Any:
        return self.variables.get(name, default)

    def evaluate_expression(self, expression: str) -> bool:
        """Evaluate a condition expression against workflow variables."""
        try:
            return bool(eval(expression, {"__builtins__": {}}, self.variables))  # noqa: S307
        except Exception as exc:
            logger.error("expression_eval_failed", expression=expression, error=str(exc))
            return False


# ---------------------------------------------------------------------------
# SessionTask
# ---------------------------------------------------------------------------


@dataclass
class SessionTask(WorkflowTask):
    """Execute an ETL Session."""

    name: str = ""
    session: Session | None = None
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    retry_count: int = 0
    retry_interval_sec: int = 60
    fail_parent_on_error: bool = True

    def run(self, context: WorkflowContext) -> TaskStatus:
        if not self.enabled:
            self.status = TaskStatus.DISABLED
            return self.status

        log = logger.bind(task=self.name, task_type="SessionTask")
        self.status = TaskStatus.RUNNING

        if self.session is None:
            log.error("no_session_configured")
            self.status = TaskStatus.FAILED
            return self.status

        attempts = 0
        max_attempts = self.retry_count + 1

        while attempts < max_attempts:
            attempts += 1
            try:
                log.info("session_starting", attempt=attempts)
                stats = self.session.execute()
                context.task_outputs[self.name] = stats

                if stats.status == SessionStatus.SUCCEEDED:
                    self.status = TaskStatus.SUCCEEDED
                    log.info("session_succeeded", stats=stats.summary())
                    return self.status
                else:
                    raise RuntimeError(f"Session ended with status: {stats.status.value}")

            except Exception as exc:
                log.warning(
                    "session_attempt_failed",
                    attempt=attempts,
                    max_attempts=max_attempts,
                    error=str(exc),
                )
                if attempts < max_attempts:
                    time.sleep(self.retry_interval_sec)

        self.status = TaskStatus.FAILED
        log.error("session_all_retries_exhausted", task=self.name)
        return self.status


# ---------------------------------------------------------------------------
# DecisionTask
# ---------------------------------------------------------------------------


@dataclass
class DecisionTask(WorkflowTask):
    """
    Conditional branching (like Informatica Decision task).

    Evaluates a condition and routes to success or failure paths.
    """

    name: str = ""
    condition_expression: str = ""
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    retry_count: int = 0
    retry_interval_sec: int = 0

    def run(self, context: WorkflowContext) -> TaskStatus:
        if not self.enabled:
            self.status = TaskStatus.DISABLED
            return self.status

        log = logger.bind(task=self.name, task_type="DecisionTask")
        self.status = TaskStatus.RUNNING

        result = context.evaluate_expression(self.condition_expression)
        log.info("decision_evaluated", expression=self.condition_expression, result=result)

        if result:
            self.status = TaskStatus.SUCCEEDED
        else:
            self.status = TaskStatus.FAILED  # Failure path = condition was False

        context.task_outputs[self.name] = {"condition_result": result}
        return self.status


# ---------------------------------------------------------------------------
# TimerTask
# ---------------------------------------------------------------------------


@dataclass
class TimerTask(WorkflowTask):
    """
    Timer task (like Informatica Timer).

    Either waits until a specific datetime or for a fixed duration.
    """

    name: str = ""
    wait_until: datetime | None = None
    wait_seconds: int = 0
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    retry_count: int = 0
    retry_interval_sec: int = 0

    def run(self, context: WorkflowContext) -> TaskStatus:
        if not self.enabled:
            self.status = TaskStatus.DISABLED
            return self.status

        log = logger.bind(task=self.name, task_type="TimerTask")
        self.status = TaskStatus.RUNNING

        if self.wait_until:
            now = datetime.now(tz=UTC)
            delta = (self.wait_until - now).total_seconds()
            if delta > 0:
                log.info("timer_waiting_until", target=self.wait_until.isoformat(), seconds=delta)
                time.sleep(delta)
        elif self.wait_seconds > 0:
            log.info("timer_waiting_duration", seconds=self.wait_seconds)
            time.sleep(self.wait_seconds)

        self.status = TaskStatus.SUCCEEDED
        return self.status


# ---------------------------------------------------------------------------
# EventWaitTask
# ---------------------------------------------------------------------------


@dataclass
class EventWaitTask(WorkflowTask):
    """
    Event wait task (like Informatica Event-Wait).

    Waits for a file to appear or a workflow variable to be set.
    """

    name: str = ""
    wait_for_file: str = ""
    wait_for_variable: str = ""
    poll_interval_sec: int = 30
    timeout_sec: int = 3600
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    retry_count: int = 0
    retry_interval_sec: int = 0

    def run(self, context: WorkflowContext) -> TaskStatus:
        if not self.enabled:
            self.status = TaskStatus.DISABLED
            return self.status

        log = logger.bind(task=self.name, task_type="EventWaitTask")
        self.status = TaskStatus.RUNNING
        elapsed = 0

        while elapsed < self.timeout_sec:
            # Check for file
            if self.wait_for_file:
                path = Path(self.wait_for_file)
                if path.exists():
                    log.info("event_file_arrived", path=self.wait_for_file)
                    self.status = TaskStatus.SUCCEEDED
                    return self.status

            # Check for variable
            if self.wait_for_variable:
                val = context.get_variable(self.wait_for_variable)
                if val:
                    log.info("event_variable_set", variable=self.wait_for_variable, value=val)
                    self.status = TaskStatus.SUCCEEDED
                    return self.status

            time.sleep(self.poll_interval_sec)
            elapsed += self.poll_interval_sec
            log.debug("event_polling", elapsed=elapsed, timeout=self.timeout_sec)

        log.warning("event_timeout", timeout=self.timeout_sec)
        self.status = TaskStatus.FAILED
        return self.status


# ---------------------------------------------------------------------------
# CommandTask
# ---------------------------------------------------------------------------


@dataclass
class CommandTask(WorkflowTask):
    """Execute a shell command."""

    name: str = ""
    command: str = ""
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    retry_count: int = 0
    retry_interval_sec: int = 30

    def run(self, context: WorkflowContext) -> TaskStatus:
        if not self.enabled:
            self.status = TaskStatus.DISABLED
            return self.status

        import subprocess

        log = logger.bind(task=self.name, task_type="CommandTask")
        self.status = TaskStatus.RUNNING

        try:
            result = subprocess.run(  # noqa: S603
                self.command,
                shell=True,  # noqa: S602
                capture_output=True,
                text=True,
                timeout=600,
            )
            context.task_outputs[self.name] = {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

            if result.returncode == 0:
                self.status = TaskStatus.SUCCEEDED
                log.info("command_succeeded", returncode=0)
            else:
                self.status = TaskStatus.FAILED
                log.error("command_failed", returncode=result.returncode, stderr=result.stderr)

        except Exception as exc:
            self.status = TaskStatus.FAILED
            log.error("command_exception", error=str(exc))

        return self.status


# ---------------------------------------------------------------------------
# EmailTask
# ---------------------------------------------------------------------------


@dataclass
class EmailTask(WorkflowTask):
    """Send email notification (like Informatica Email task)."""

    name: str = ""
    to_addresses: list[str] = field(default_factory=list)
    subject: str = ""
    body: str = ""
    include_session_stats: bool = True
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    retry_count: int = 0
    retry_interval_sec: int = 30

    def run(self, context: WorkflowContext) -> TaskStatus:
        if not self.enabled:
            self.status = TaskStatus.DISABLED
            return self.status

        log = logger.bind(task=self.name, task_type="EmailTask")
        self.status = TaskStatus.RUNNING

        recipients = self.to_addresses or context.notification_email_to
        if not recipients:
            log.warning("no_email_recipients")
            self.status = TaskStatus.SUCCEEDED
            return self.status

        # Build email body
        body_parts = [self.body]
        if self.include_session_stats:
            body_parts.append("\n--- Session Statistics ---")
            for task_name, output in context.task_outputs.items():
                if isinstance(output, PerformanceStats):
                    body_parts.append(f"\n{task_name}:")
                    for k, v in output.summary().items():
                        body_parts.append(f"  {k}: {v}")
                elif isinstance(output, dict):
                    body_parts.append(f"\n{task_name}: {output}")

        full_body = "\n".join(body_parts)

        try:
            msg = MIMEText(full_body)
            msg["Subject"] = self._resolve_subject(context)
            msg["From"] = context.notification_email_from
            msg["To"] = ", ".join(recipients)

            with smtplib.SMTP(context.smtp_server, context.smtp_port) as server:
                server.send_message(msg)

            self.status = TaskStatus.SUCCEEDED
            log.info("email_sent", to=recipients, subject=msg["Subject"])

        except Exception as exc:
            # Email failures are typically non-fatal
            log.warning("email_send_failed", error=str(exc))
            self.status = TaskStatus.SUCCEEDED  # Don't fail workflow over email

        return self.status

    def _resolve_subject(self, context: WorkflowContext) -> str:
        """Resolve variables in the subject line."""
        subject = self.subject or f"PharmaFlow Workflow: {context.workflow_name}"
        for var_name, var_val in context.variables.items():
            subject = subject.replace(f"$${var_name}", str(var_val))
        return subject


# ---------------------------------------------------------------------------
# AssignmentTask
# ---------------------------------------------------------------------------


@dataclass
class AssignmentTask(WorkflowTask):
    """Set workflow variable values."""

    name: str = ""
    assignments: dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    enabled: bool = True
    retry_count: int = 0
    retry_interval_sec: int = 0

    def run(self, context: WorkflowContext) -> TaskStatus:
        if not self.enabled:
            self.status = TaskStatus.DISABLED
            return self.status

        for var_name, value in self.assignments.items():
            context.set_variable(var_name, value)
            logger.info("variable_assigned", name=var_name, value=value)

        self.status = TaskStatus.SUCCEEDED
        return self.status


# ---------------------------------------------------------------------------
# Workflow
# ---------------------------------------------------------------------------


@dataclass
class Workflow:
    """
    Informatica Workflow equivalent.

    Orchestrates tasks with dependency management, parallel execution,
    conditional branching, and error handling.
    """

    name: str
    description: str = ""
    tasks: dict[str, WorkflowTask] = field(default_factory=dict)
    links: list[TaskLink] = field(default_factory=list)
    start_task: str = ""
    context: WorkflowContext = field(default_factory=WorkflowContext)
    max_parallel: int = 4
    _execution_order: list[list[str]] = field(default_factory=list, repr=False)

    def add_task(self, task: WorkflowTask) -> Workflow:
        """Register a task in the workflow."""
        self.tasks[task.name] = task
        return self

    def add_link(
        self,
        from_task: str,
        to_task: str,
        condition: LinkCondition = LinkCondition.ON_SUCCESS,
        expression: str = "",
    ) -> Workflow:
        """Add a directed link between tasks."""
        self.links.append(TaskLink(from_task, to_task, condition, expression))
        return self

    def _get_successors(self, task_name: str) -> list[TaskLink]:
        """Get all outgoing links from a task."""
        return [link for link in self.links if link.from_task == task_name]

    def _get_predecessors(self, task_name: str) -> list[TaskLink]:
        """Get all incoming links to a task."""
        return [link for link in self.links if link.to_task == task_name]

    def _evaluate_link(self, link: TaskLink) -> bool:
        """Determine if a link's condition is satisfied."""
        from_status = self.context.task_results.get(link.from_task, TaskStatus.PENDING)

        if link.condition == LinkCondition.UNCONDITIONAL:
            return True
        if link.condition == LinkCondition.ON_SUCCESS:
            return from_status == TaskStatus.SUCCEEDED
        if link.condition == LinkCondition.ON_FAILURE:
            return from_status == TaskStatus.FAILED
        if link.condition == LinkCondition.EXPRESSION:
            return self.context.evaluate_expression(link.expression)
        return False

    def _can_run_task(self, task_name: str) -> bool:
        """Check if all predecessor conditions are met for a task."""
        predecessors = self._get_predecessors(task_name)
        if not predecessors:
            return task_name == self.start_task

        return any(self._evaluate_link(link) for link in predecessors)

    def _build_execution_plan(self) -> list[list[str]]:
        """
        Build a topological execution plan with parallel groups.

        Tasks with no dependencies on each other run in the same group (parallel).
        """
        # Find tasks with satisfied prerequisites
        completed: set[str] = set()
        remaining = set(self.tasks.keys())
        plan: list[list[str]] = []

        # Start task first
        if self.start_task and self.start_task in remaining:
            plan.append([self.start_task])
            completed.add(self.start_task)
            remaining.discard(self.start_task)

        max_iterations = len(self.tasks) + 1
        iteration = 0

        while remaining and iteration < max_iterations:
            iteration += 1
            ready: list[str] = []

            for task_name in list(remaining):
                predecessors = self._get_predecessors(task_name)
                if not predecessors:
                    continue
                # All predecessors must be completed
                pred_tasks = {link.from_task for link in predecessors}
                if pred_tasks.issubset(completed):
                    ready.append(task_name)

            if ready:
                plan.append(ready)
                for t in ready:
                    completed.add(t)
                    remaining.discard(t)
            else:
                # No tasks ready -- remaining are unreachable
                break

        self._execution_order = plan
        return plan

    def execute(self) -> dict[str, TaskStatus]:
        """
        Execute the workflow.

        Returns dict mapping task names to their final status.
        """
        log = logger.bind(workflow=self.name)
        self.context.workflow_name = self.name
        self.context.start_time = datetime.now(tz=UTC)

        log.info("workflow_starting", tasks=list(self.tasks.keys()))

        plan = self._build_execution_plan()
        log.info("execution_plan", plan=plan)

        for group in plan:
            # Determine which tasks in this group should actually run
            runnable = [t for t in group if self._can_run_task(t)]
            skipped = [t for t in group if t not in runnable]

            for t in skipped:
                self.context.task_results[t] = TaskStatus.SKIPPED
                self.tasks[t].status = TaskStatus.SKIPPED
                log.info("task_skipped", task=t)

            if not runnable:
                continue

            if len(runnable) == 1:
                # Sequential execution
                self._run_single_task(runnable[0])
            else:
                # Parallel execution
                self._run_parallel_tasks(runnable)

        # Final status
        results = dict(self.context.task_results)

        # Send failure notification if any task failed
        failed_tasks = [t for t, s in results.items() if s == TaskStatus.FAILED]
        if failed_tasks:
            log.error("workflow_has_failures", failed_tasks=failed_tasks)
            self._send_failure_notification(failed_tasks)
        else:
            log.info("workflow_completed_successfully")

        return results

    def _run_single_task(self, task_name: str) -> None:
        """Execute a single task with retry logic."""
        task = self.tasks[task_name]
        log = logger.bind(task=task_name, task_type=type(task).__name__)

        log.info("task_starting")
        status = task.run(self.context)
        self.context.task_results[task_name] = status
        log.info("task_completed", status=status.value)

    def _run_parallel_tasks(self, task_names: list[str]) -> None:
        """Execute multiple tasks in parallel using thread pool."""
        log = logger.bind(parallel_group=task_names)
        log.info("parallel_group_starting")

        with ThreadPoolExecutor(max_workers=min(len(task_names), self.max_parallel)) as executor:
            futures = {}
            for task_name in task_names:
                task = self.tasks[task_name]
                futures[executor.submit(task.run, self.context)] = task_name

            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    status = future.result()
                    self.context.task_results[task_name] = status
                except Exception as exc:
                    self.context.task_results[task_name] = TaskStatus.FAILED
                    logger.error("parallel_task_exception", task=task_name, error=str(exc))

        log.info("parallel_group_complete")

    def _send_failure_notification(self, failed_tasks: list[str]) -> None:
        """Send email notification for workflow failures."""
        if not self.context.notification_email_to:
            return

        email_task = EmailTask(
            name="_failure_notification",
            subject=f"[ALERT] PharmaFlow Workflow '{self.name}' FAILED",
            body=(
                f"Workflow '{self.name}' completed with failures.\n"
                f"Failed tasks: {', '.join(failed_tasks)}\n"
                f"Time: {datetime.now(tz=UTC).isoformat()}\n"
            ),
            include_session_stats=True,
        )
        email_task.run(self.context)

    def get_status_report(self) -> dict[str, Any]:
        """Generate a human-readable status report."""
        return {
            "workflow": self.name,
            "start_time": (
                self.context.start_time.isoformat() if self.context.start_time else None
            ),
            "tasks": {
                name: {
                    "type": type(task).__name__,
                    "status": self.context.task_results.get(name, TaskStatus.PENDING).value,
                    "enabled": task.enabled,
                }
                for name, task in self.tasks.items()
            },
            "variables": dict(self.context.variables),
        }
