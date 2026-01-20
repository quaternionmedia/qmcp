"""Tests for agent framework capability mixins."""

import pytest

from qmcp.agentframework.mixins import (
    MemoryConfig,
    MemoryEntry,
    MemoryMixin,
    MixinRegistry,
    ReasoningMixin,
    StructuredOutputConfig,
    StructuredOutputMixin,
    ToolUseConfig,
    ToolUseMixin,
)


class TestMixinRegistry:
    """Tests for MixinRegistry."""

    def test_get_registered_mixin(self):
        retrieved = MixinRegistry.get("tool_use")
        assert retrieved is ToolUseMixin

    def test_create_mixin_with_config(self):
        instance = MixinRegistry.create("tool_use", {"max_tool_calls_per_turn": 3})
        assert isinstance(instance, ToolUseMixin)
        assert instance.tool_config.max_tool_calls_per_turn == 3

    def test_create_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown mixin"):
            MixinRegistry.create("nonexistent_mixin")

    def test_list_all(self):
        mixins = MixinRegistry.list_all()
        assert len(mixins) > 0

        names = [mixin["name"] for mixin in mixins]
        assert "tool_use" in names
        assert "memory" in names

    def test_resolve_dependencies(self):
        ordered = MixinRegistry.resolve_dependencies(["human_in_loop", "memory"])

        tool_idx = ordered.index("tool_use")
        hitl_idx = ordered.index("human_in_loop")
        assert tool_idx < hitl_idx


class TestToolUseMixin:
    """Tests for ToolUseMixin."""

    def test_default_config(self):
        mixin = ToolUseMixin()
        assert mixin.tool_config.max_tool_calls_per_turn == 5
        assert mixin.tool_config.require_confirmation is False

    def test_custom_config(self):
        config = ToolUseConfig(
            allowed_tools=["web_search", "calculator"],
            max_tool_calls_per_turn=3,
        )
        mixin = ToolUseMixin(config)
        assert mixin.tool_config.allowed_tools == ["web_search", "calculator"]

    @pytest.mark.asyncio
    async def test_on_start_resets_counter(self):
        mixin = ToolUseMixin()
        mixin._tool_call_count = 10

        await mixin.on_start({})

        assert mixin._tool_call_count == 0

    @pytest.mark.asyncio
    async def test_on_tool_call_allowed(self):
        config = ToolUseConfig(allowed_tools=["calculator"])
        mixin = ToolUseMixin(config)

        result = await mixin.on_tool_call("calculator", {"x": 1})
        assert result is None

    @pytest.mark.asyncio
    async def test_on_tool_call_denied(self):
        config = ToolUseConfig(allowed_tools=["calculator"])
        mixin = ToolUseMixin(config)

        with pytest.raises(PermissionError, match="not in allowed list"):
            await mixin.on_tool_call("web_search", {})

    @pytest.mark.asyncio
    async def test_rate_limit_exceeded(self):
        config = ToolUseConfig(max_tool_calls_per_turn=2)
        mixin = ToolUseMixin(config)

        await mixin.on_tool_call("tool1", {})
        await mixin.on_tool_call("tool2", {})

        with pytest.raises(RuntimeError, match="Exceeded max tool calls"):
            await mixin.on_tool_call("tool3", {})


class TestMemoryMixin:
    """Tests for MemoryMixin."""

    def test_default_config(self):
        mixin = MemoryMixin()
        assert mixin.memory_config.max_memories == 100
        assert mixin.memory_config.auto_summarize is True

    @pytest.mark.asyncio
    async def test_remember(self):
        mixin = MemoryMixin()

        entry = await mixin.remember("Test memory", importance=0.8, tags=["test"])

        assert entry.content == "Test memory"
        assert entry.importance == 0.8
        assert "test" in entry.tags

    @pytest.mark.asyncio
    async def test_recall(self):
        mixin = MemoryMixin()

        await mixin.remember("Memory 1", importance=0.5)
        await mixin.remember("Memory 2", importance=0.9)
        await mixin.remember("Memory 3", importance=0.3)

        recalled = await mixin.recall(limit=2)

        assert len(recalled) == 2
        assert recalled[0].importance >= recalled[1].importance

    @pytest.mark.asyncio
    async def test_recall_with_tags(self):
        mixin = MemoryMixin()

        await mixin.remember("Tagged", tags=["important"])
        await mixin.remember("Untagged")

        recalled = await mixin.recall(tags=["important"])

        assert len(recalled) == 1
        assert recalled[0].content == "Tagged"

    @pytest.mark.asyncio
    async def test_forget(self):
        mixin = MemoryMixin()

        await mixin.remember("To forget")
        await mixin.remember("To keep")

        result = await mixin.forget("To forget")

        assert result is True
        assert len(mixin._memories) == 1

    def test_memory_pruning(self):
        config = MemoryConfig(max_memories=2)
        mixin = MemoryMixin(config)

        mixin._memories = [
            MemoryEntry(content="Memory 1", importance=0.3),
            MemoryEntry(content="Memory 2", importance=0.9),
            MemoryEntry(content="Memory 3", importance=0.5),
        ]

        mixin._prune_memories()

        assert len(mixin._memories) == 2
        contents = [entry.content for entry in mixin._memories]
        assert "Memory 1" not in contents


class TestReasoningMixin:
    """Tests for ReasoningMixin."""

    def test_available_styles(self):
        assert "step_by_step" in ReasoningMixin.REASONING_PROMPTS
        assert "tree_of_thought" in ReasoningMixin.REASONING_PROMPTS
        assert "socratic" in ReasoningMixin.REASONING_PROMPTS

    @pytest.mark.asyncio
    async def test_on_invoke_injects_prompt(self):
        mixin = ReasoningMixin()

        request = {"messages": [{"role": "user", "content": "Test question"}]}
        result = await mixin.on_invoke(request)

        assert result["messages"][0]["role"] == "system"
        assert "step by step" in result["messages"][0]["content"].lower()

    @pytest.mark.asyncio
    async def test_on_response_extracts_reasoning(self):
        mixin = ReasoningMixin()

        response = {"content": "1. First step\n2. Second step\n3. Conclusion"}
        result = await mixin.on_response(response)

        assert "_reasoning" in result
        assert len(result["_reasoning"]["steps"]) >= 2


class TestStructuredOutputMixin:
    """Tests for StructuredOutputMixin."""

    @pytest.mark.asyncio
    async def test_on_invoke_adds_schema(self):
        schema = {
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }
        config = StructuredOutputConfig(output_schema=schema)
        mixin = StructuredOutputMixin(config)

        request = {"messages": [{"role": "user", "content": "Question"}]}
        result = await mixin.on_invoke(request)

        assert "JSON" in result["messages"][0]["content"]

    def test_extract_json_direct(self):
        mixin = StructuredOutputMixin()

        result = mixin._extract_json('{"key": "value"}')

        assert result == {"key": "value"}

    def test_extract_json_from_markdown(self):
        mixin = StructuredOutputMixin()

        content = (
            "Here is the answer:\n"
            "```json\n"
            '{"key": "value"}\n'
            "```\n"
        )
        result = mixin._extract_json(content)

        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_validation_passes(self):
        schema = {"type": "object", "required": ["answer"]}
        config = StructuredOutputConfig(output_schema=schema, strict_validation=True)
        mixin = StructuredOutputMixin(config)

        response = {"content": '{"answer": "test"}'}
        result = await mixin.on_response(response)

        assert result["_structured_output"]["answer"] == "test"

    @pytest.mark.asyncio
    async def test_validation_fails(self):
        schema = {"type": "object", "required": ["missing_field"]}
        config = StructuredOutputConfig(
            output_schema=schema,
            strict_validation=True,
            auto_repair=False,
        )
        mixin = StructuredOutputMixin(config)

        response = {"content": '{"other": "value"}'}

        with pytest.raises(ValueError, match="Missing required field"):
            await mixin.on_response(response)
