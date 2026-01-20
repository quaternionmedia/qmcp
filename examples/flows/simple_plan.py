"""Simple planning flow demonstrating MCP tool invocation.

This example shows how to use the MCP client from a Metaflow flow
to invoke tools and process the results.

Usage:
    # First start the MCP server
    uv run qmcp serve

    # Then run the flow
    uv run python examples/flows/simple_plan.py run \
        --goal "Deploy a web service" \
        --mcp-url "http://localhost:3333"

    # On Windows, run via Docker
    docker compose -f docker-compose.flows.yml run --rm flow-runner \
        examples/flows/simple_plan.py run \
        --goal "Deploy a web service" \
        --mcp-url "http://host.docker.internal:3333"
"""

import os

from metaflow import FlowSpec, Parameter, step

from qmcp.client import MCPClient


class SimplePlanFlow(FlowSpec):
    """A simple flow that creates and executes a plan using MCP tools.

    Steps:
    1. start: Initialize and check server health
    2. plan: Use the planner tool to create an execution plan
    3. execute: Use the executor tool to run the plan (dry run)
    4. review: Use the reviewer tool to assess the results
    5. end: Output the final summary
    """

    goal = Parameter(
        "goal",
        help="The goal to plan for",
        default="Deploy a new microservice",
    )

    mcp_url = Parameter(
        "mcp-url",
        help="URL of the MCP server",
        default=os.getenv("MCP_URL", "http://localhost:3333"),
    )

    @step
    def start(self):
        """Initialize the flow and verify MCP server connectivity."""
        print(f"Goal: {self.goal}")
        print(f"MCP Server: {self.mcp_url}")

        # Check server health
        with MCPClient(self.mcp_url) as client:
            health = client.health()
            print(f"Server status: {health['status']}")
            print(f"Server version: {health['version']}")

            # List available tools
            tools = client.list_tools()
            print(f"Available tools: {[t.name for t in tools]}")

        self.next(self.plan)

    @step
    def plan(self):
        """Create an execution plan using the planner tool."""
        print(f"Creating plan for: {self.goal}")

        with MCPClient(self.mcp_url) as client:
            result = client.invoke_tool(
                "planner",
                {"goal": self.goal},
                correlation_id=f"flow-{self.run_id}",
            )

            if result.error:
                raise RuntimeError(f"Planner failed: {result.error}")

            self.plan_result = result.result
            self.plan_invocation_id = result.invocation_id

        print(f"Plan created with {self.plan_result['estimated_steps']} steps")
        for step_info in self.plan_result["steps"]:
            print(f"  Step {step_info['step']}: {step_info['action']}")

        self.next(self.execute)

    @step
    def execute(self):
        """Execute the plan using the executor tool (dry run)."""
        print("Executing plan (dry run mode)...")

        with MCPClient(self.mcp_url) as client:
            result = client.invoke_tool(
                "executor",
                {"plan": self.plan_result, "dry_run": True},
                correlation_id=f"flow-{self.run_id}",
            )

            if result.error:
                raise RuntimeError(f"Executor failed: {result.error}")

            self.execution_result = result.result
            self.execution_invocation_id = result.invocation_id

        print(f"Execution mode: {self.execution_result['mode']}")
        print(f"Steps executed: {self.execution_result['steps_executed']}")
        print(f"Success: {self.execution_result['success']}")

        self.next(self.review)

    @step
    def review(self):
        """Review the execution results using the reviewer tool."""
        print("Reviewing execution results...")

        with MCPClient(self.mcp_url) as client:
            result = client.invoke_tool(
                "reviewer",
                {
                    "result": self.execution_result,
                    "criteria": ["completeness", "correctness", "efficiency"],
                },
                correlation_id=f"flow-{self.run_id}",
            )

            if result.error:
                raise RuntimeError(f"Reviewer failed: {result.error}")

            self.review_result = result.result
            self.review_invocation_id = result.invocation_id

        print(f"Overall status: {self.review_result['overall_status']}")
        print(f"Recommendation: {self.review_result['recommendation']}")

        self.next(self.end)

    @step
    def end(self):
        """Output the final summary."""
        print("\n" + "=" * 50)
        print("FLOW COMPLETE")
        print("=" * 50)
        print(f"Goal: {self.goal}")
        print(f"Plan steps: {self.plan_result['estimated_steps']}")
        print(f"Execution success: {self.execution_result['success']}")
        print(f"Review status: {self.review_result['overall_status']}")
        print("\nInvocation IDs for audit:")
        print(f"  Plan: {self.plan_invocation_id}")
        print(f"  Execute: {self.execution_invocation_id}")
        print(f"  Review: {self.review_invocation_id}")


if __name__ == "__main__":
    SimplePlanFlow()
