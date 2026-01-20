"""Tests for Human-in-the-Loop (HITL) endpoints."""



# Client fixture is inherited from conftest.py


class TestCreateHumanRequest:
    """Tests for creating human requests."""

    def test_create_approval_request(self, client):
        """Test creating a basic approval request."""
        response = client.post(
            "/v1/human/requests",
            json={
                "id": "approve-deploy-001",
                "request_type": "approval",
                "prompt": "Approve deployment to production?",
                "options": ["approve", "reject"],
                "timeout_seconds": 3600,
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["id"] == "approve-deploy-001"
        assert data["request_type"] == "approval"
        assert data["status"] == "pending"
        assert data["expires_at"] is not None

    def test_create_input_request(self, client):
        """Test creating an input request (no options)."""
        response = client.post(
            "/v1/human/requests",
            json={
                "id": "input-reason-001",
                "request_type": "input",
                "prompt": "Please provide a reason for this change:",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["request_type"] == "input"

    def test_duplicate_request_id_rejected(self, client):
        """Test that duplicate request IDs are rejected."""
        # Create first request
        client.post(
            "/v1/human/requests",
            json={
                "id": "duplicate-001",
                "request_type": "approval",
                "prompt": "First request",
            },
        )

        # Try to create duplicate
        response = client.post(
            "/v1/human/requests",
            json={
                "id": "duplicate-001",
                "request_type": "approval",
                "prompt": "Duplicate request",
            },
        )
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"]

    def test_request_with_context(self, client):
        """Test creating a request with additional context."""
        response = client.post(
            "/v1/human/requests",
            json={
                "id": "context-001",
                "request_type": "review",
                "prompt": "Review this document",
                "context": {
                    "document_id": "doc-123",
                    "author": "alice@example.com",
                    "changes": ["Added section 2", "Fixed typo"],
                },
            },
        )
        assert response.status_code == 201


class TestListHumanRequests:
    """Tests for listing human requests."""

    def test_list_requests(self, client):
        """Test listing all requests."""
        # Create some requests
        client.post(
            "/v1/human/requests",
            json={"id": "list-001", "request_type": "approval", "prompt": "Request 1"},
        )
        client.post(
            "/v1/human/requests",
            json={"id": "list-002", "request_type": "input", "prompt": "Request 2"},
        )

        response = client.get("/v1/human/requests")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] >= 2

    def test_filter_by_status(self, client):
        """Test filtering requests by status."""
        client.post(
            "/v1/human/requests",
            json={"id": "filter-001", "request_type": "approval", "prompt": "Test"},
        )

        response = client.get("/v1/human/requests", params={"status": "pending"})
        assert response.status_code == 200
        data = response.json()
        assert all(r["status"] == "pending" for r in data["requests"])

    def test_filter_by_type(self, client):
        """Test filtering requests by type."""
        client.post(
            "/v1/human/requests",
            json={"id": "type-001", "request_type": "approval", "prompt": "Approval"},
        )
        client.post(
            "/v1/human/requests",
            json={"id": "type-002", "request_type": "input", "prompt": "Input"},
        )

        response = client.get("/v1/human/requests", params={"request_type": "approval"})
        assert response.status_code == 200
        data = response.json()
        assert all(r["request_type"] == "approval" for r in data["requests"])


class TestGetHumanRequest:
    """Tests for getting a single human request."""

    def test_get_request(self, client):
        """Test getting a request by ID."""
        client.post(
            "/v1/human/requests",
            json={"id": "get-001", "request_type": "approval", "prompt": "Test"},
        )

        response = client.get("/v1/human/requests/get-001")
        assert response.status_code == 200
        data = response.json()
        assert data["request"]["id"] == "get-001"
        assert data["response"] is None  # No response yet

    def test_get_nonexistent_request(self, client):
        """Test getting a non-existent request."""
        response = client.get("/v1/human/requests/nonexistent")
        assert response.status_code == 404


class TestSubmitHumanResponse:
    """Tests for submitting human responses."""

    def test_submit_response(self, client):
        """Test submitting a response to a pending request."""
        # Create request
        client.post(
            "/v1/human/requests",
            json={
                "id": "respond-001",
                "request_type": "approval",
                "prompt": "Approve?",
                "options": ["approve", "reject"],
            },
        )

        # Submit response
        response = client.post(
            "/v1/human/responses",
            json={
                "request_id": "respond-001",
                "response": "approve",
                "responded_by": "alice@example.com",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["response"] == "approve"
        assert data["responded_by"] == "alice@example.com"

    def test_response_updates_request_status(self, client):
        """Test that responding updates the request status."""
        client.post(
            "/v1/human/requests",
            json={"id": "status-001", "request_type": "approval", "prompt": "Test"},
        )

        client.post(
            "/v1/human/responses",
            json={"request_id": "status-001", "response": "approved"},
        )

        # Check request status
        response = client.get("/v1/human/requests/status-001")
        data = response.json()
        assert data["request"]["status"] == "responded"
        assert data["response"] is not None

    def test_cannot_respond_twice(self, client):
        """Test that a request cannot be responded to twice."""
        client.post(
            "/v1/human/requests",
            json={"id": "double-001", "request_type": "approval", "prompt": "Test"},
        )

        # First response
        client.post(
            "/v1/human/responses",
            json={"request_id": "double-001", "response": "yes"},
        )

        # Second response should fail
        response = client.post(
            "/v1/human/responses",
            json={"request_id": "double-001", "response": "no"},
        )
        assert response.status_code == 409
        assert "already been responded" in response.json()["detail"]

    def test_response_validates_options(self, client):
        """Test that responses are validated against allowed options."""
        client.post(
            "/v1/human/requests",
            json={
                "id": "validate-001",
                "request_type": "approval",
                "prompt": "Choose",
                "options": ["yes", "no"],
            },
        )

        response = client.post(
            "/v1/human/responses",
            json={"request_id": "validate-001", "response": "maybe"},
        )
        assert response.status_code == 400
        assert "must be one of" in response.json()["detail"]

    def test_respond_to_nonexistent_request(self, client):
        """Test responding to a non-existent request."""
        response = client.post(
            "/v1/human/responses",
            json={"request_id": "nonexistent", "response": "yes"},
        )
        assert response.status_code == 404


class TestHITLWorkflow:
    """End-to-end tests for the HITL workflow."""

    def test_complete_approval_workflow(self, client):
        """Test a complete approval workflow."""
        # 1. Create approval request
        create_response = client.post(
            "/v1/human/requests",
            json={
                "id": "workflow-001",
                "request_type": "approval",
                "prompt": "Approve deployment to production?",
                "options": ["approve", "reject"],
                "context": {"service": "api-gateway", "environment": "prod"},
            },
        )
        assert create_response.status_code == 201

        # 2. Poll for response (should be pending)
        poll_response = client.get("/v1/human/requests/workflow-001")
        assert poll_response.status_code == 200
        assert poll_response.json()["request"]["status"] == "pending"
        assert poll_response.json()["response"] is None

        # 3. Submit human response
        submit_response = client.post(
            "/v1/human/responses",
            json={
                "request_id": "workflow-001",
                "response": "approve",
                "responded_by": "ops@example.com",
                "response_metadata": {"reason": "Looks good"},
            },
        )
        assert submit_response.status_code == 201

        # 4. Poll again (should have response)
        final_response = client.get("/v1/human/requests/workflow-001")
        assert final_response.status_code == 200
        data = final_response.json()
        assert data["request"]["status"] == "responded"
        assert data["response"]["response"] == "approve"
