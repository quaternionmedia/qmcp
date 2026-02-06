"""Council deliberation flow for multi-perspective decision making.

A council topology where an arbiter presides over a plurality/majority seeking
council of diverse agent perspectives. Each member contributes their unique
viewpoint until consensus or majority is reached.

Council Members:
    - Council Manager (Arbiter): Facilitates, synthesizes, decides
    - Relatable Storyteller: Frames issues in narrative form
    - Infinite Dreamer: Explores possibilities without constraint
    - Pragmatic Strategist: Focuses on practical implementation
    - Sanity Check: Validates feasibility, catches issues
    - Tidy Archivist: Maintains context, references history
    - Brutal Efficist: Cuts through complexity for efficiency
    - Eager Accomplisher: Drives toward completion
    - Technical Reflector: Provides deep technical analysis

Usage:
    # Install flow dependencies
    uv sync --extra flows

    # Run with local Ollama (default - optimized for speed)
    uv run python examples/flows/council_deliberation.py run \
        --question "Should we migrate from REST to GraphQL?"

    # Run with context
    uv run python examples/flows/council_deliberation.py run \
        --question "Should we migrate from REST to GraphQL?" \
        --context "E-commerce platform with 50+ microservices"

    # Run with OpenAI (slower but higher quality)
    uv run python examples/flows/council_deliberation.py run \
        --question "Should we adopt a microservices architecture?" \
        --llm-base-url "https://api.openai.com/v1" \
        --llm-model "gpt-4o-mini" \
        --llm-api-key "$OPENAI_API_KEY"
"""

from __future__ import annotations

import json
import os
from enum import Enum
from typing import Any

from metaflow import FlowSpec, Parameter, current, step
from pydantic import BaseModel, Field


# =============================================================================
# Council Member Personas & System Prompts
# =============================================================================


class CouncilRole(str, Enum):
    """Council member roles."""

    ARBITER = "arbiter"
    STORYTELLER = "storyteller"
    DREAMER = "dreamer"
    STRATEGIST = "strategist"
    SANITY_CHECK = "sanity_check"
    ARCHIVIST = "archivist"
    EFFICIST = "efficist"
    ACCOMPLISHER = "accomplisher"
    REFLECTOR = "reflector"


# Concise prompts optimized for local LLM speed
COUNCIL_PROMPTS: dict[CouncilRole, str] = {
    CouncilRole.ARBITER: """You are the Council Arbiter. Be concise.
Responsibilities: Synthesize viewpoints, track consensus, make decisions.
Style: Impartial and decisive.""",
    CouncilRole.STORYTELLER: """You are the Storyteller. Be concise.
Role: Frame issues as human stories and analogies.
Style: Narrative-driven, use "imagine if..." framing.""",
    CouncilRole.DREAMER: """You are the Dreamer. Be concise.
Role: Explore possibilities without constraint, challenge assumptions.
Style: Visionary, use "What if..." framing.""",
    CouncilRole.STRATEGIST: """You are the Strategist. Be concise.
Role: Ground ideas in reality with timelines and resources.
Style: Practical, use "Here's how..." framing.""",
    CouncilRole.SANITY_CHECK: """You are the Sanity Check. Be concise.
Role: Find risks, edge cases, and overlooked problems.
Style: Skeptical, use "But have we considered..." framing.""",
    CouncilRole.ARCHIVIST: """You are the Archivist. Be concise.
Role: Reference past decisions and historical patterns.
Style: Scholarly, use "Previously we..." framing.""",
    CouncilRole.EFFICIST: """You are the Efficist. Be concise.
Role: Cut complexity, find the simplest path.
Style: Direct, use "The simplest approach is..." framing.""",
    CouncilRole.ACCOMPLISHER: """You are the Accomplisher. Be concise.
Role: Drive toward action and shipping.
Style: Action-oriented, use "Let's ship..." framing.""",
    CouncilRole.REFLECTOR: """You are the Technical Reflector. Be concise.
Role: Provide technical analysis on architecture and scalability.
Style: Rigorous, use "Technically..." framing.""",
}


# =============================================================================
# Output Models (simplified for local LLM compatibility)
# =============================================================================

# Valid positions - use strings for better local LLM compatibility
VALID_POSITIONS = ["strongly_for", "for", "neutral", "against", "strongly_against"]


class CouncilContribution(BaseModel):
    """A single council member's contribution."""

    role: str = Field(default="council_member", description="Council role name")
    position: str = Field(
        default="neutral",
        description="Position: strongly_for, for, neutral, against, strongly_against",
    )
    reasoning: str = Field(default="No reasoning provided", description="Main argument")
    key_points: list[str] = Field(default_factory=list, description="Key points")
    concerns: list[str] = Field(default_factory=list, description="Concerns")
    suggestions: list[str] = Field(default_factory=list, description="Suggestions")


class RoundSynthesis(BaseModel):
    """Arbiter's synthesis after each round."""

    round_number: int = Field(default=1)
    consensus_level: float = Field(default=0.5, description="0-1 consensus level")
    majority_position: str = Field(default="neutral", description="Majority position")
    key_agreements: list[str] = Field(default_factory=list)
    key_disagreements: list[str] = Field(default_factory=list)
    recommendation_to_continue: bool = Field(default=True)


class CouncilDecision(BaseModel):
    """Final council decision."""

    question: str = Field(default="")
    final_position: str = Field(default="neutral")
    confidence: float = Field(default=0.5)
    consensus_reached: bool = Field(default=False)
    rounds_taken: int = Field(default=1)
    summary: str = Field(default="Decision pending", description="Summary")
    rationale: str = Field(default="", description="Rationale")
    key_factors: list[str] = Field(default_factory=list)
    dissenting_views: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)


# =============================================================================
# Council Deliberation Flow
# =============================================================================


class CouncilDeliberationFlow(FlowSpec):
    """Multi-agent council deliberation for complex decisions.

    The council follows a structured deliberation process:
    1. Each member contributes their perspective (round-robin)
    2. Arbiter synthesizes positions and checks for consensus
    3. If no consensus, another round with refined positions
    4. Arbiter makes final decision after max_rounds or consensus
    """

    question = Parameter(
        "question",
        help="The question for the council to deliberate",
        required=True,
    )
    context = Parameter(
        "context",
        help="Additional context for the deliberation",
        default="",
    )
    max_rounds = Parameter(
        "max-rounds",
        help="Maximum deliberation rounds (default: 2 for speed)",
        type=int,
        default=2,
    )
    consensus_threshold = Parameter(
        "consensus-threshold",
        help="Proportion required for consensus (0.5-1.0)",
        type=float,
        default=0.67,
    )
    llm_base_url = Parameter(
        "llm-base-url",
        help="OpenAI-compatible base URL (default: local Ollama)",
        default=os.getenv("LLM_BASE_URL", "http://localhost:11434/v1"),
    )
    llm_model = Parameter(
        "llm-model",
        help="Model name (default: llama3.2 for speed)",
        default=os.getenv("LLM_MODEL", "llama3.2"),
    )
    llm_api_key = Parameter(
        "llm-api-key",
        help="API key if required",
        default=os.getenv("LLM_API_KEY", "local"),
    )
    llm_temperature = Parameter(
        "llm-temperature",
        help="Generation temperature (lower=faster, more deterministic)",
        type=float,
        default=float(os.getenv("LLM_TEMPERATURE", "0.3")),
    )
    llm_max_tokens = Parameter(
        "llm-max-tokens",
        help="Max tokens per response (lower=faster)",
        type=int,
        default=int(os.getenv("LLM_MAX_TOKENS", "512")),
    )
    # MCP URL parameter for compatibility with Docker runner infrastructure
    # (not used by this flow, but accepted for CLI compatibility)
    mcp_url = Parameter(
        "mcp-url",
        help="MCP server URL (not used by council flow, for CLI compatibility)",
        default=os.getenv("MCP_URL", "http://localhost:3333"),
    )

    def _build_agent(self, system_prompt: str, output_type: type[BaseModel]):
        """Build a PydanticAI agent with the given configuration."""
        from openai import AsyncOpenAI
        from pydantic_ai import Agent
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        from pydantic_ai.settings import ModelSettings

        # Create custom OpenAI client for local LLM
        client = AsyncOpenAI(
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )
        provider = OpenAIProvider(openai_client=client)
        model = OpenAIChatModel(self.llm_model, provider=provider)

        # Add JSON instruction to help local LLMs
        json_instruction = "\n\nIMPORTANT: Respond ONLY with valid JSON. No explanation or text before/after."

        return Agent(
            model=model,
            system_prompt=system_prompt + json_instruction,
            output_type=output_type,
            retries=3,  # More retries for local LLMs
            model_settings=ModelSettings(
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            ),
        )

    def _get_council_contribution(
        self,
        role: CouncilRole,
        question: str,
        context: str,
        previous_contributions: list[dict[str, Any]],
        round_number: int,
    ) -> CouncilContribution:
        """Get a contribution from a council member."""
        agent = self._build_agent(
            system_prompt=COUNCIL_PROMPTS[role],
            output_type=CouncilContribution,
        )

        prompt_parts = [
            f"QUESTION FOR COUNCIL: {question}",
        ]

        if context:
            prompt_parts.append(f"\nCONTEXT: {context}")

        prompt_parts.append(f"\nThis is round {round_number} of deliberation.")

        if previous_contributions:
            # Only show last 3 positions for speed
            prompt_parts.append("\nOTHER POSITIONS:")
            for contrib in previous_contributions[-3:]:
                prompt_parts.append(f"- {contrib['role'].upper()}: {contrib['position']}")

        prompt_parts.append(
            "\n\nProvide your perspective as the "
            f"{role.value.replace('_', ' ').title()}."
        )

        result = agent.run_sync("\n".join(prompt_parts))
        contribution = result.output
        contribution.role = role.value
        return contribution

    def _synthesize_round(
        self,
        question: str,
        round_number: int,
        contributions: list[CouncilContribution],
    ) -> RoundSynthesis:
        """Arbiter synthesizes the round's contributions."""
        agent = self._build_agent(
            system_prompt=COUNCIL_PROMPTS[CouncilRole.ARBITER],
            output_type=RoundSynthesis,
        )

        prompt_parts = [
            f"QUESTION: {question}",
            f"\nROUND {round_number} CONTRIBUTIONS:",
        ]

        # Compact format for speed
        for contrib in contributions:
            prompt_parts.append(f"- {contrib.role.upper()}: {contrib.position}")

        prompt_parts.append(
            "\n\nSynthesize: assess consensus (0-1), majority position, "
            "key agreements/disagreements, recommend if another round helps."
        )

        result = agent.run_sync("\n".join(prompt_parts))
        synthesis = result.output
        synthesis.round_number = round_number
        return synthesis

    def _make_final_decision(
        self,
        question: str,
        context: str,
        all_rounds: list[dict[str, Any]],
    ) -> CouncilDecision:
        """Arbiter makes the final decision based on all deliberation."""
        agent = self._build_agent(
            system_prompt=COUNCIL_PROMPTS[CouncilRole.ARBITER],
            output_type=CouncilDecision,
        )

        prompt_parts = [
            f"QUESTION: {question}",
        ]

        if context:
            prompt_parts.append(f"CONTEXT: {context}")

        prompt_parts.append(f"\nDELIBERATION SUMMARY ({len(all_rounds)} rounds):")

        for round_data in all_rounds:
            round_num = round_data["round_number"]
            synthesis = round_data["synthesis"]
            prompt_parts.append(f"\n## Round {round_num}")
            prompt_parts.append(f"Consensus level: {synthesis['consensus_level']:.0%}")
            prompt_parts.append(f"Majority position: {synthesis['majority_position']}")
            if synthesis.get("key_agreements"):
                prompt_parts.append(
                    f"Agreements: {', '.join(synthesis['key_agreements'][:3])}"
                )
            if synthesis.get("key_disagreements"):
                prompt_parts.append(
                    f"Disagreements: {', '.join(synthesis['key_disagreements'][:3])}"
                )

        prompt_parts.append(
            "\n\nAs the Arbiter, render the final council decision. "
            "Provide a clear position, rationale, and recommended actions."
        )

        result = agent.run_sync("\n".join(prompt_parts))
        decision = result.output
        decision.question = question
        decision.rounds_taken = len(all_rounds)
        return decision

    @step
    def start(self):
        """Initialize the council deliberation."""
        self.run_id = current.run_id
        print(f"Council Deliberation: {self.run_id}")
        print(f"Question: {self.question}")
        print(f"Max rounds: {self.max_rounds}")
        print(f"Consensus threshold: {self.consensus_threshold:.0%}")
        print(f"Model: {self.llm_model}")

        self.all_rounds: list[dict[str, Any]] = []
        self.current_round = 0
        self.consensus_reached = False

        self.next(self.deliberate)

    @step
    def deliberate(self):
        """Run deliberation rounds until consensus or max rounds."""
        speaking_order = [
            CouncilRole.STORYTELLER,
            CouncilRole.DREAMER,
            CouncilRole.STRATEGIST,
            CouncilRole.SANITY_CHECK,
            CouncilRole.ARCHIVIST,
            CouncilRole.EFFICIST,
            CouncilRole.ACCOMPLISHER,
            CouncilRole.REFLECTOR,
        ]

        while self.current_round < self.max_rounds and not self.consensus_reached:
            self.current_round += 1
            print(f"\n{'='*60}")
            print(f"ROUND {self.current_round}")
            print("=" * 60)

            # Gather contributions from each council member
            contributions: list[CouncilContribution] = []
            contribution_dicts: list[dict[str, Any]] = []

            for role in speaking_order:
                print(f"\n{role.value.replace('_', ' ').title()} is speaking...")

                contrib = self._get_council_contribution(
                    role=role,
                    question=self.question,
                    context=self.context,
                    previous_contributions=contribution_dicts,
                    round_number=self.current_round,
                )
                contributions.append(contrib)
                contribution_dicts.append(contrib.model_dump())

                print(f"  Position: {contrib.position}")
                print(f"  {contrib.reasoning[:100]}...")

            # Arbiter synthesizes the round
            print(f"\nArbiter synthesizing round {self.current_round}...")
            synthesis = self._synthesize_round(
                question=self.question,
                round_number=self.current_round,
                contributions=contributions,
            )

            round_data = {
                "round_number": self.current_round,
                "contributions": [c.model_dump() for c in contributions],
                "synthesis": synthesis.model_dump(),
            }
            self.all_rounds.append(round_data)

            print(f"\nRound {self.current_round} Summary:")
            print(f"  Consensus level: {synthesis.consensus_level:.0%}")
            print(f"  Majority position: {synthesis.majority_position}")

            # Check if consensus reached
            if synthesis.consensus_level >= self.consensus_threshold:
                self.consensus_reached = True
                print(f"\nConsensus reached at {synthesis.consensus_level:.0%}!")
            elif not synthesis.recommendation_to_continue:
                print("\nArbiter recommends concluding deliberation.")
                break

        self.next(self.decide)

    @step
    def decide(self):
        """Arbiter renders the final decision."""
        print(f"\n{'='*60}")
        print("FINAL DECISION")
        print("=" * 60)

        self.decision = self._make_final_decision(
            question=self.question,
            context=self.context,
            all_rounds=self.all_rounds,
        )
        self.decision.consensus_reached = self.consensus_reached

        print(f"\nPosition: {self.decision.final_position}")
        print(f"Confidence: {self.decision.confidence:.0%}")
        print(f"Consensus: {'Yes' if self.decision.consensus_reached else 'No'}")
        print(f"\nSummary: {self.decision.summary}")

        self.next(self.end)

    @step
    def end(self):
        """Output the complete deliberation record."""
        print(f"\n{'='*60}")
        print("COUNCIL DELIBERATION COMPLETE")
        print("=" * 60)

        # Full decision output
        print(f"\nQuestion: {self.decision.question}")
        print(f"Rounds: {self.decision.rounds_taken}")
        print(f"Final Position: {self.decision.final_position}")
        print(f"Confidence: {self.decision.confidence:.0%}")

        print(f"\nRationale:\n{self.decision.rationale}")

        if self.decision.key_factors:
            print("\nKey Factors:")
            for factor in self.decision.key_factors:
                print(f"  - {factor}")

        if self.decision.recommended_actions:
            print("\nRecommended Actions:")
            for action in self.decision.recommended_actions:
                print(f"  - {action}")

        if self.decision.dissenting_views:
            print("\nDissenting Views:")
            for view in self.decision.dissenting_views:
                print(f"  - {view}")

        # Save full deliberation record
        self.deliberation_record = {
            "question": self.question,
            "context": self.context,
            "config": {
                "max_rounds": self.max_rounds,
                "consensus_threshold": self.consensus_threshold,
                "model": self.llm_model,
            },
            "rounds": self.all_rounds,
            "decision": self.decision.model_dump(),
        }

        print(f"\n\nFull record saved to self.deliberation_record")
        print(f"Access with: flow.data.deliberation_record")


if __name__ == "__main__":
    CouncilDeliberationFlow()
