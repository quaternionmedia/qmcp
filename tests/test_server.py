"""Tests for the MCP server endpoints."""



# Client fixture is inherited from conftest.py


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_ok(self, client):
        """Test that health endpoint returns healthy status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data


class TestToolsEndpoint:
    """Tests for the tool discovery endpoint."""

    def test_list_tools(self, client):
        """Test listing available tools."""
        response = client.get("/v1/tools")
        assert response.status_code == 200
        data = response.json()
        assert "tools" in data
        assert len(data["tools"]) > 0

    def test_tools_have_required_fields(self, client):
        """Test that tool definitions have required fields."""
        response = client.get("/v1/tools")
        data = response.json()
        for tool in data["tools"]:
            assert "name" in tool
            assert "description" in tool
            assert "input_schema" in tool

    def test_builtin_tools_present(self, client):
        """Test that builtin tools are registered."""
        response = client.get("/v1/tools")
        data = response.json()
        tool_names = [t["name"] for t in data["tools"]]
        assert "echo" in tool_names
        assert "planner" in tool_names
        assert "executor" in tool_names
        assert "reviewer" in tool_names


class TestToolInvocation:
    """Tests for tool invocation endpoint."""

    def test_invoke_echo_tool(self, client):
        """Test invoking the echo tool."""
        response = client.post(
            "/v1/tools/echo",
            json={"input": {"message": "Hello, MCP!"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"] == "Hello, MCP!"
        assert data["error"] is None

    def test_invoke_planner_tool(self, client):
        """Test invoking the planner tool."""
        response = client.post(
            "/v1/tools/planner",
            json={"input": {"goal": "Deploy a service"}},
        )
        assert response.status_code == 200
        data = response.json()
        assert "goal" in data["result"]
        assert "steps" in data["result"]

    def test_invoke_nonexistent_tool(self, client):
        """Test invoking a tool that doesn't exist."""
        response = client.post(
            "/v1/tools/nonexistent",
            json={"input": {}},
        )
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_invoke_executor_dry_run(self, client):
        """Test executor tool in dry run mode."""
        plan = {
            "goal": "Test",
            "steps": [{"step": 1, "action": "Test action"}],
        }
        response = client.post(
            "/v1/tools/executor",
            json={"input": {"plan": plan, "dry_run": True}},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["result"]["mode"] == "dry_run"
        assert data["result"]["success"] is True


class TestInvocationHistory:
    """Tests for invocation history endpoints."""

    def test_invocation_logged(self, client):
        """Test that tool invocations are logged."""
        # Invoke a tool
        client.post("/v1/tools/echo", json={"input": {"message": "test"}})

        # Check history
        response = client.get("/v1/invocations")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 1
        assert any(inv["tool_name"] == "echo" for inv in data["invocations"])

    def test_invocation_returns_id(self, client):
        """Test that tool invocation returns an invocation ID."""
        response = client.post("/v1/tools/echo", json={"input": {"message": "test"}})
        data = response.json()
        assert "invocation_id" in data
        assert data["invocation_id"] is not None

    def test_filter_by_tool_name(self, client):
        """Test filtering invocations by tool name."""
        # Create some invocations
        client.post("/v1/tools/echo", json={"input": {"message": "1"}})
        client.post("/v1/tools/planner", json={"input": {"goal": "test"}})
        client.post("/v1/tools/echo", json={"input": {"message": "2"}})

        # Filter by echo
        response = client.get("/v1/invocations", params={"tool_name": "echo"})
        data = response.json()
        assert all(inv["tool_name"] == "echo" for inv in data["invocations"])

    def test_get_single_invocation(self, client):
        """Test getting a single invocation by ID."""
        # Create an invocation
        response = client.post("/v1/tools/echo", json={"input": {"message": "test"}})
        invocation_id = response.json()["invocation_id"]

        # Get it by ID
        response = client.get(f"/v1/invocations/{invocation_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == invocation_id
        assert data["tool_name"] == "echo"

    def test_get_nonexistent_invocation(self, client):
        """Test getting a non-existent invocation."""
        response = client.get("/v1/invocations/nonexistent-id")
        assert response.status_code == 404
