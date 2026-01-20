"""Tests for the tool registry."""


from qmcp.tools.registry import Tool, ToolRegistry


class TestTool:
    """Tests for the Tool class."""

    def test_tool_creation(self):
        """Test basic tool creation."""
        tool = Tool(
            name="test",
            description="A test tool",
            handler=lambda params: params.get("value"),
        )
        assert tool.name == "test"
        assert tool.description == "A test tool"

    def test_tool_invocation(self):
        """Test tool invocation."""
        tool = Tool(
            name="double",
            description="Doubles a number",
            handler=lambda params: params.get("n", 0) * 2,
        )
        result = tool.invoke({"n": 5})
        assert result == 10

    def test_tool_to_definition(self):
        """Test conversion to MCP definition."""
        tool = Tool(
            name="test",
            description="A test tool",
            handler=lambda params: None,
            input_schema={"type": "object"},
        )
        definition = tool.to_definition()
        assert definition.name == "test"
        assert definition.description == "A test tool"
        assert definition.input_schema == {"type": "object"}


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_decorator(self):
        """Test tool registration via decorator."""
        registry = ToolRegistry()

        @registry.register("greet", "Greet someone")
        def greet(params):
            return f"Hello, {params.get('name', 'World')}!"

        tool = registry.get("greet")
        assert tool is not None
        assert tool.name == "greet"
        assert tool.invoke({"name": "Test"}) == "Hello, Test!"

    def test_list_tools(self):
        """Test listing registered tools."""
        registry = ToolRegistry()

        @registry.register("a", "Tool A")
        def tool_a(params):
            return "a"

        @registry.register("b", "Tool B")
        def tool_b(params):
            return "b"

        tools = registry.list_tools()
        assert len(tools) == 2
        names = [t.name for t in tools]
        assert "a" in names
        assert "b" in names

    def test_get_nonexistent_tool(self):
        """Test getting a tool that doesn't exist."""
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_definitions(self):
        """Test listing tool definitions for discovery."""
        registry = ToolRegistry()

        @registry.register("test", "Test tool", input_schema={"type": "object"})
        def test_tool(params):
            return None

        definitions = registry.list_definitions()
        assert len(definitions) == 1
        assert definitions[0].name == "test"
