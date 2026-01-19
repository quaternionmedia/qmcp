"""Approved deployment flow demonstrating Human-in-the-Loop.

This example shows how to use the MCP client to request human approval
before proceeding with a deployment operation.

Usage:
    # First start the MCP server
    uv run qmcp serve

    # In another terminal, run the flow
    uv run python examples/flows/approved_deploy.py run \\
        --service "api-gateway" \\
        --environment "production"

    # The flow will pause waiting for approval. Submit via API:
    curl -X POST http://localhost:3333/v1/human/responses \\
        -H "Content-Type: application/json" \\
        -d '{"request_id": "<request-id>", "response": "approve", "responded_by": "operator@example.com"}'
"""

from metaflow import FlowSpec, step, Parameter

from qmcp.client import MCPClient, HumanRequestExpiredError


class ApprovedDeployFlow(FlowSpec):
    """A deployment flow that requires human approval before proceeding.

    Steps:
    1. start: Validate inputs and check server health
    2. plan: Create a deployment plan
    3. request_approval: Create human approval request and wait
    4. deploy: Execute deployment (or abort if rejected)
    5. end: Output final status
    """

    service = Parameter(
        "service",
        help="Name of the service to deploy",
        required=True,
    )

    environment = Parameter(
        "environment",
        help="Target environment (staging, production)",
        default="staging",
    )

    mcp_url = Parameter(
        "mcp-url",
        help="URL of the MCP server",
        default="http://localhost:3333",
    )

    approval_timeout = Parameter(
        "approval-timeout",
        help="Timeout in seconds for human approval",
        default=3600,
        type=int,
    )

    @step
    def start(self):
        """Validate inputs and check server health."""
        print(f"Service: {self.service}")
        print(f"Environment: {self.environment}")
        print(f"MCP Server: {self.mcp_url}")

        # Validate environment
        if self.environment not in ("staging", "production"):
            raise ValueError(f"Invalid environment: {self.environment}")

        # Check server health
        with MCPClient(self.mcp_url) as client:
            health = client.health()
            print(f"Server status: {health['status']}")

        self.next(self.plan)

    @step
    def plan(self):
        """Create a deployment plan."""
        print(f"Creating deployment plan for {self.service}...")

        with MCPClient(self.mcp_url) as client:
            result = client.invoke_tool(
                "planner",
                {
                    "goal": f"Deploy {self.service} to {self.environment}",
                    "context": f"Service: {self.service}, Environment: {self.environment}",
                },
                correlation_id=f"deploy-{self.run_id}",
            )

            self.deployment_plan = result.result

        print(f"Plan created with {self.deployment_plan['estimated_steps']} steps:")
        for step_info in self.deployment_plan["steps"]:
            print(f"  {step_info['step']}. {step_info['action']}")

        self.next(self.request_approval)

    @step
    def request_approval(self):
        """Request human approval for the deployment."""
        # Only require approval for production
        if self.environment != "production":
            print("Staging deployment - no approval required")
            self.approved = True
            self.approval_response = None
            self.next(self.deploy)
            return

        print("\n" + "=" * 50)
        print("HUMAN APPROVAL REQUIRED")
        print("=" * 50)

        request_id = f"deploy-{self.service}-{self.run_id}"

        with MCPClient(self.mcp_url) as client:
            # Create the approval request
            request = client.create_human_request(
                request_id=request_id,
                request_type="approval",
                prompt=f"Approve deployment of '{self.service}' to PRODUCTION?",
                options=["approve", "reject"],
                context={
                    "service": self.service,
                    "environment": self.environment,
                    "plan_steps": self.deployment_plan["estimated_steps"],
                    "flow_run_id": self.run_id,
                },
                timeout_seconds=self.approval_timeout,
                correlation_id=f"deploy-{self.run_id}",
            )

            print(f"Approval request created: {request.id}")
            print(f"Waiting for human response (timeout: {self.approval_timeout}s)...")
            print(f"\nTo approve, run:")
            print(f'  curl -X POST {self.mcp_url}/v1/human/responses \\')
            print(f'    -H "Content-Type: application/json" \\')
            print(f"    -d '{{\"request_id\": \"{request_id}\", \"response\": \"approve\", \"responded_by\": \"your@email.com\"}}'")
            print()

            try:
                response = client.wait_for_human_response(
                    request_id=request_id,
                    poll_interval=5.0,
                )

                self.approved = response.response == "approve"
                self.approval_response = response
                print(f"Response received: {response.response}")
                print(f"Responded by: {response.responded_by}")

            except HumanRequestExpiredError:
                print("Approval request expired - aborting deployment")
                self.approved = False
                self.approval_response = None

        self.next(self.deploy)

    @step
    def deploy(self):
        """Execute the deployment (if approved)."""
        if not self.approved:
            print("Deployment NOT approved - skipping execution")
            self.deployment_result = {
                "status": "aborted",
                "reason": "not_approved" if self.approval_response else "expired",
            }
            self.next(self.end)
            return

        print(f"Executing deployment of {self.service} to {self.environment}...")

        with MCPClient(self.mcp_url) as client:
            # Execute in dry_run mode for safety in examples
            dry_run = self.environment == "production"

            result = client.invoke_tool(
                "executor",
                {
                    "plan": self.deployment_plan,
                    "dry_run": dry_run,
                },
                correlation_id=f"deploy-{self.run_id}",
            )

            self.deployment_result = result.result

        print(f"Deployment complete:")
        print(f"  Mode: {self.deployment_result['mode']}")
        print(f"  Steps: {self.deployment_result['steps_executed']}")
        print(f"  Success: {self.deployment_result['success']}")

        self.next(self.end)

    @step
    def end(self):
        """Output final status."""
        print("\n" + "=" * 50)
        print("DEPLOYMENT FLOW COMPLETE")
        print("=" * 50)
        print(f"Service: {self.service}")
        print(f"Environment: {self.environment}")
        print(f"Approved: {self.approved}")

        if hasattr(self, "deployment_result"):
            status = self.deployment_result.get("status", "executed")
            print(f"Status: {status}")

            if self.deployment_result.get("success"):
                print("✅ Deployment successful")
            elif status == "aborted":
                print("⚠️ Deployment aborted")
            else:
                print("❌ Deployment failed")


if __name__ == "__main__":
    ApprovedDeployFlow()
