"""Built-in tools for QMCP.

These tools demonstrate the tool pattern and provide basic functionality.
All tools are:
- Stateless
- Deterministic where possible
- Single-responsibility
"""

from typing import Any

from qmcp.tools.registry import tool_registry


@tool_registry.register(
    name="echo",
    description="Echo the input message back. Useful for testing.",
    input_schema={
        "type": "object",
        "properties": {
            "message": {
                "type": "string",
                "description": "The message to echo back",
            }
        },
        "required": ["message"],
    },
)
def echo(params: dict[str, Any]) -> str:
    """Echo tool - returns the input message."""
    return params.get("message", "")


@tool_registry.register(
    name="planner",
    description="Create a step-by-step execution plan from a goal.",
    input_schema={
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "The goal to create a plan for",
            },
            "context": {
                "type": "string",
                "description": "Optional context to inform planning",
            },
        },
        "required": ["goal"],
    },
)
def planner(params: dict[str, Any]) -> dict[str, Any]:
    """Planner tool - creates a structured execution plan.

    In a real implementation, this might use an LLM.
    For now, it returns a template plan.
    """
    goal = params.get("goal", "")
    context = params.get("context", "")

    # Template plan - in production, this would be more sophisticated
    steps = [
        {"step": 1, "action": "Analyze requirements", "description": f"Understand: {goal}"},
        {"step": 2, "action": "Identify dependencies", "description": "List prerequisites"},
        {"step": 3, "action": "Create execution order", "description": "Sequence tasks"},
        {"step": 4, "action": "Validate plan", "description": "Check for gaps"},
    ]

    if context:
        steps.insert(1, {"step": 1.5, "action": "Review context", "description": context})

    return {
        "goal": goal,
        "steps": steps,
        "estimated_steps": len(steps),
    }


@tool_registry.register(
    name="executor",
    description="Execute an approved plan. Returns execution summary.",
    input_schema={
        "type": "object",
        "properties": {
            "plan": {
                "type": "object",
                "description": "The plan to execute",
            },
            "dry_run": {
                "type": "boolean",
                "description": "If true, simulate execution without side effects",
            },
        },
        "required": ["plan"],
    },
)
def executor(params: dict[str, Any]) -> dict[str, Any]:
    """Executor tool - executes a plan.

    In a real implementation, this would perform actual actions.
    For now, it returns a mock execution report.
    """
    plan = params.get("plan", {})
    dry_run = params.get("dry_run", True)

    steps = plan.get("steps", [])
    executed = []

    for step in steps:
        executed.append({
            "step": step.get("step"),
            "action": step.get("action"),
            "status": "simulated" if dry_run else "completed",
        })

    return {
        "mode": "dry_run" if dry_run else "live",
        "steps_executed": len(executed),
        "results": executed,
        "success": True,
    }


@tool_registry.register(
    name="reviewer",
    description="Review results and provide a summary assessment.",
    input_schema={
        "type": "object",
        "properties": {
            "result": {
                "type": "object",
                "description": "The result to review",
            },
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Review criteria to check",
            },
        },
        "required": ["result"],
    },
)
def reviewer(params: dict[str, Any]) -> dict[str, Any]:
    """Reviewer tool - assesses results against criteria.

    In a real implementation, this might use an LLM for analysis.
    For now, it returns a template review.
    """
    result = params.get("result", {})
    criteria = params.get("criteria", ["completeness", "correctness"])

    checks = []
    for criterion in criteria:
        checks.append({
            "criterion": criterion,
            "status": "passed",
            "notes": f"Checked {criterion}",
        })

    return {
        "reviewed_result": result,
        "checks": checks,
        "overall_status": "approved",
        "recommendation": "Proceed with next steps",
    }
