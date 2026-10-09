"""Consent-gated execution and authoring for saved, composable topologies."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from qmcp import orchestration as plane
from qmcp.agentframework.models.enums import TopologyType
from qmcp.integrations.agents import AgentOutcome, AgentRuntime, Brief, OnEvent
from qmcp.topology_designs import split_reference

NAME = "topology-design"
MAX_COMPONENTS = 12
MAX_DESIGNS = 8
MAX_MODEL_REQUESTS = 48
MODEL_REQUEST_TIMEOUT = 90
MODEL_TOKENS = 512


def _local_runtime() -> AgentRuntime:
    """A bounded local reader; its tools cannot write or run shell commands."""
    from qmcp.integrations.agents.adapters.ollama import Runtime

    return Runtime(
        max_steps=1,
        recover_timeouts=False,
        max_tokens=MODEL_TOKENS,
        timeout=MODEL_REQUEST_TIMEOUT,
    )


def _contains_phrase(words: list[str], phrase: str) -> bool:
    target = re.findall(r"\w+", phrase.casefold())
    return bool(target) and any(
        words[index:index + len(target)] == target
        for index in range(len(words) - len(target) + 1)
    )


def _request_count(outcome: AgentOutcome) -> int | None:
    """The model requests an outcome reports, or None when it is not a real count."""
    calls = outcome.detail.get("model_calls")
    ok = isinstance(calls, int) and not isinstance(calls, bool) and calls >= 0
    return calls if ok else None


def _entry(name: str, outcome: AgentOutcome, calls: int | None, *,
           unknown: bool = False) -> dict[str, Any]:
    return {
        "name": name,
        "text": outcome.text,
        "model_calls": calls or 0,
        "model_call_count_unknown": calls is None or unknown,
    }


def _unsound(outcome: AgentOutcome, calls: int | None) -> bool:
    """Whether a model run failed, spent, said nothing, or broke its request limit."""
    return (
        not outcome.succeeded
        or outcome.spent != 0
        or not outcome.text.strip()
        or calls is None
        or calls > 2
    )


def _routed(node: _Design, task: str) -> list[dict[str, Any]]:
    """The components of a delegation whose declared route terms appear in the task."""
    words = re.findall(r"\w+", task.casefold())
    return [
        item for item in node.components
        if any(_contains_phrase(words, term) for term in item.get("route_terms", []))
    ]


@dataclass(frozen=True)
class _Design:
    project: str
    name: str
    kind: str
    version: str
    config: dict[str, Any]
    components: tuple[dict[str, Any], ...]
    children: tuple[_Design, ...]


class TopologyDesignRuntime:
    """Apply one approved design mutation or run an approved saved topology.

    A run snapshots the full design tree before consent. Immediately after
    consent it reads that tree again and refuses to run if any referenced
    design or shared component changed in the meantime.
    """

    name = NAME
    would_spend = 0
    uses_history = False

    def __init__(
        self,
        client: Any,
        command: Any,
        *,
        runtime_factory: Callable[[], AgentRuntime] = _local_runtime,
    ) -> None:
        self.client = client
        self.request = command
        self.runtime_factory = runtime_factory
        self._snapshot: _Design | None = None
        self._fingerprint: str | None = None
        self._request_ceiling = 0
        self._made = 0
        if command.action == "run":
            self._snapshot = self._load_tree(command.name, command.project)
            design_count, component_count = self._counts(self._snapshot)
            if design_count > MAX_DESIGNS:
                raise ValueError(
                    f"the composed design has {design_count} nodes; the limit is "
                    f"{MAX_DESIGNS}"
                )
            if component_count > MAX_COMPONENTS:
                raise ValueError(
                    f"the composed design references {component_count} components; "
                    f"the limit is {MAX_COMPONENTS}"
                )
            self._validate_runtime_config(self._snapshot)
            self._refuse_before_consent(self._snapshot)
            self._fingerprint = self._fingerprint_tree(self._snapshot)
            self._request_ceiling = self._forecast(self._snapshot)
            if self._request_ceiling == 0:
                raise ValueError(
                    "this topology has no components or runnable composed designs"
                )
            if self._request_ceiling > MAX_MODEL_REQUESTS:
                raise ValueError(
                    f"this design could make {self._request_ceiling} local-model "
                    f"requests; the limit is {MAX_MODEL_REQUESTS}"
                )
        self.command = self._consent_description()

    def _refuse_before_consent(self, node: _Design) -> None:
        """Raise for what the harness would refuse, so nobody approves a refused run."""
        kind = TopologyType(node.kind)
        # The design's own config, so an advisory council reads its own
        # declaration and the deciding one is refused here, before consent.
        capability = plane.capability_for(kind, node.config)
        refusal = plane.refuses(kind, self.request.prompt, node.config)
        if capability is None:
            raise ValueError(
                "the capability plane has no declaration for "
                f"topology {node.name!r}"
            )
        if refusal:
            raise ValueError(f"topology {node.name!r} refused: {refusal}")
        if not capability.voice_runnable:
            raise ValueError(
                f"topology {node.name!r} of type {node.kind} has no voice run"
            )
        if capability.voice_spends or capability.voice_writes or capability.voice_decides:
            raise ValueError(
                f"the voice path for {node.kind} is not declared read-only and "
                "report-only"
            )
        if kind is TopologyType.DELEGATION and not _routed(node, self.request.prompt):
            raise ValueError(
                f"topology {node.name!r} would not route the task: no component's "
                "declared route terms match it"
            )
        for child in node.children:
            self._refuse_before_consent(child)

    @staticmethod
    def _counts(node: _Design) -> tuple[int, int]:
        child_counts = [TopologyDesignRuntime._counts(child) for child in node.children]
        return (
            1 + sum(count[0] for count in child_counts),
            len(node.components) + sum(count[1] for count in child_counts),
        )

    def _validate_runtime_config(self, node: _Design) -> None:
        from qmcp.agentframework.models.entities.topologies import config_class_for

        if (
            node.kind != TopologyType.CROSS_CHECK.value
            and not node.components
            and not node.children
        ):
            raise ValueError(
                f"topology {node.name!r} has no components or composed designs"
            )
        config_type = config_class_for(TopologyType(node.kind))
        if config_type is None:
            raise ValueError(f"topology type {node.kind!r} has no config model")
        effective = config_type.model_validate(node.config).model_dump(mode="json")
        default = config_type().model_dump(mode="json")
        supported = {"components", "compose"}
        if node.kind == TopologyType.CROSS_CHECK.value:
            supported.add("num_checkers")
        if node.kind == TopologyType.COUNCIL.value:
            supported.update({"speaking_order", "arbiter_can_override"})
        unsupported = [
            key for key, value in effective.items()
            if key not in supported and value != default.get(key)
        ]
        if unsupported:
            raise ValueError(
                f"topology {node.name!r} sets options this voice runner does not "
                f"implement: {', '.join(sorted(unsupported))}"
            )
        for child in node.children:
            self._validate_runtime_config(child)

    def _get_component(self, name: str, project: str) -> dict[str, Any]:
        return self.client.get_topology_component(name, project=project)

    def _load_tree(
        self, name: str, project: str, seen: tuple[tuple[str, str], ...] = ()
    ) -> _Design:
        if (project, name) in seen:
            raise ValueError(f"composition cycle found through topology {name!r}")
        if len(seen) >= MAX_DESIGNS:
            raise ValueError(f"composed designs exceed the {MAX_DESIGNS}-design limit")
        payload = self.client.get_topology(name, project=project)
        if not isinstance(payload, dict) or not all(
            isinstance(payload.get(key), str)
            for key in ("name", "topology_type", "version")
        ):
            raise ValueError(f"topology {name!r} returned an invalid design record")
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"topology {name!r} has no valid configuration")
        refs = config.get("components", [])
        if (
            not isinstance(refs, list)
            or any(
                not isinstance(ref, dict)
                or not isinstance(ref.get("name"), str)
                or not isinstance(ref.get("route_terms", []), list)
                or not all(isinstance(term, str) for term in ref.get("route_terms", []))
                for ref in refs
            )
        ):
            raise ValueError(f"topology {name!r} has malformed component references")
        components = []
        for ref in refs:
            component = self._get_component(ref["name"], project)
            if (
                not isinstance(component, dict)
                or not all(
                    isinstance(component.get(key), str)
                    for key in ("name", "description", "instruction", "version")
                )
            ):
                raise ValueError(
                    f"component {ref['name']!r} returned an invalid component record"
                )
            components.append({
                **component,
                "route_terms": list(ref.get("route_terms", [])),
            })
        child_names = config.get("compose", [])
        if not isinstance(child_names, list) or not all(
            isinstance(child, str) and child for child in child_names
        ):
            raise ValueError(f"topology {name!r} has malformed composition references")
        children = []
        for child in child_names:
            child_project, child_name = split_reference(child, project)
            children.append(
                self._load_tree(
                    child_name, child_project, (*seen, (project, name))
                )
            )
        return _Design(
            project=project,
            name=payload["name"],
            kind=payload["topology_type"],
            version=payload["version"],
            config=config,
            components=tuple(components),
            children=tuple(children),
        )

    @staticmethod
    def _fingerprint_tree(tree: _Design) -> str:
        def materialise(node: _Design) -> dict[str, Any]:
            return {
                "project": node.project,
                "name": node.name,
                "kind": node.kind,
                "version": node.version,
                "config": node.config,
                "components": node.components,
                "children": [materialise(child) for child in node.children],
            }

        encoded = json.dumps(materialise(tree), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _forecast(self, node: _Design) -> int:
        if node.kind == TopologyType.CROSS_CHECK.value:
            count = len(node.components) or int(node.config.get("num_checkers", 3))
            if not 2 <= count <= 10:
                raise ValueError("a crosscheck needs between 2 and 10 checkers")
            own = count * 4
        else:
            own = len(node.components) * 2
            if node.kind in {"debate", "ensemble", "council"}:
                own += 2
        return own + sum(self._forecast(child) for child in node.children)

    def _consent_description(self) -> str:
        action = self.request.action
        if action != "run":
            details = {
                "create_topology": (
                    f"create saved {self.request.kind} topology {self.request.name}"
                ),
                "create_component": (
                    f"create shared component {self.request.name} with instruction "
                    f"{self.request.instruction!r}"
                ),
                "edit_component": (
                    f"change shared component {self.request.name} instruction to "
                    f"{self.request.instruction!r}"
                ),
                "add_component": (
                    f"add shared component {self.request.component} to topology "
                    f"{self.request.name}"
                ),
                "remove_component": (
                    f"remove shared component {self.request.component} from topology "
                    f"{self.request.name}"
                ),
                "route_component": (
                    f"set route terms for shared component {self.request.component} "
                    f"in topology {self.request.name} to {self.request.route_terms!r}"
                ),
                "compose": (
                    f"compose topology {self.request.name} with topology "
                    f"{self.request.child}"
                ),
            }
            return f"Project {self.request.project}: {details[action]}."

        assert self._snapshot is not None
        node = self._snapshot
        plan: list[str] = []

        def describe(current: _Design, depth: int = 0) -> None:
            indent = " " * depth
            plan.append(
                f"{indent}Topology {current.name}, type {current.kind}, "
                f"version {current.version}."
            )
            if current.kind == TopologyType.CROSS_CHECK.value:
                plan.append(
                    f"{indent}Checker count: "
                    f"{len(current.components) or current.config.get('num_checkers', 3)}."
                )
            if current.kind == TopologyType.COUNCIL.value:
                plan.append(
                    f"{indent}Advisory speaking order: "
                    f"{', '.join(current.config.get('speaking_order', []))}."
                )
            if current.components:
                for item in current.components:
                    routes = item.get("route_terms", [])
                    plan.append(
                        f"{indent}Component {item['name']} v{item.get('version')}: "
                        f"{item.get('description', 'no description')}. "
                        f"Instruction: {item['instruction']!r}. "
                        f"Route terms: {', '.join(routes) if routes else 'none'}."
                    )
            else:
                plan.append(f"{indent}Components: none.")
            for child in current.children:
                describe(child, depth + 2)

        describe(node)
        details = [
            f"Task: {self.request.prompt}",
            "Approved design tree: " + " ".join(plan),
            "This is a local-model, read-only advisory report; it cannot change files "
            "or perform a human-only act.",
            f"Hard ceiling: {self._request_ceiling} model requests, at most "
            f"{MAX_MODEL_REQUESTS} for the entire composed run; each request is "
            f"limited to {MODEL_TOKENS} tokens and {MODEL_REQUEST_TIMEOUT} seconds.",
        ]
        return " ".join(details)

    def run(self, brief: Brief, on_event: OnEvent | None = None) -> AgentOutcome:
        if self.request.action != "run":
            return self._mutate()
        return self._run_design(brief, on_event)

    def _mutate(self) -> AgentOutcome:
        request = self.request
        if request.action == "create_topology":
            result = self.client.create_topology({
                "project": request.project,
                "name": request.name,
                "description": "Voice-authored advisory topology design.",
                "topology_type": request.kind,
                "config": {},
            })
            text = f"Created topology {result['name']} of type {result['topology_type']}."
        elif request.action == "create_component":
            result = self.client.create_topology_component({
                "project": request.project,
                "name": request.name,
                "description": "Shared voice-authored topology component.",
                "instruction": request.instruction,
            })
            text = f"Created shared component {result['name']}."
        elif request.action == "edit_component":
            result = self.client.update_topology_component(
                request.name, {"instruction": request.instruction},
                project=request.project,
            )
            text = f"Updated shared component {result['name']}."
        elif request.action in {
            "add_component", "remove_component", "route_component", "compose"
        }:
            result = self.client.get_topology(request.name, project=request.project)
            config = result["config"]
            if request.action == "add_component":
                component = self._get_component(request.component, request.project)
                references = list(config.get("components", []))
                if any(item["name"] == component["name"] for item in references):
                    raise ValueError(
                        f"component {component['name']!r} is already in "
                        f"topology {request.name!r}"
                    )
                references.append({"name": component["name"], "route_terms": []})
                config["components"] = references
                text = f"Added shared component {component['name']} to topology {request.name}."
            elif request.action == "remove_component":
                references = list(config.get("components", []))
                remaining = [
                    item for item in references
                    if item["name"] != request.component.lower()
                ]
                if len(remaining) == len(references):
                    raise ValueError(
                        f"component {request.component!r} is not in topology "
                        f"{request.name!r}"
                    )
                config["components"] = remaining
                text = f"Removed shared component {request.component.lower()} from topology {request.name}."
            elif request.action == "route_component":
                if result["topology_type"] != TopologyType.DELEGATION.value:
                    raise ValueError(
                        "route terms apply only to delegation topologies"
                    )
                terms = [
                    " ".join(term.strip().lower().split())
                    for term in re.split(r"\s+or\s+", request.route_terms, flags=re.IGNORECASE)
                ]
                if any(not term for term in terms) or len(set(terms)) != len(terms):
                    raise ValueError("route terms must be non-empty and unique")
                references = list(config.get("components", []))
                matched = False
                for item in references:
                    if item["name"] == request.component.lower():
                        item["route_terms"] = terms
                        matched = True
                if not matched:
                    raise ValueError(
                        f"component {request.component!r} is not in topology "
                        f"{request.name!r}"
                    )
                config["components"] = references
                text = (
                    f"Set {request.component} route terms to {', '.join(terms)} "
                    f"in topology {request.name}."
                )
            else:
                children = list(config.get("compose", []))
                if request.child in children:
                    raise ValueError(
                        f"topology {request.child!r} is already composed into "
                        f"{request.name!r}"
                    )
                children.append(request.child)
                config["compose"] = children
                text = f"Composed topology {request.child} into topology {request.name}."
            self.client.update_topology(
                request.name, {"config": config}, project=request.project
            )
        else:
            raise ValueError(f"unsupported topology action {request.action!r}")
        return AgentOutcome(
            text=text,
            exit_code=0,
            spent=0,
            detail={"topology_action": request.action, "project": request.project},
        )

    def _run_design(self, brief: Brief, on_event: OnEvent | None) -> AgentOutcome:
        assert self._snapshot is not None
        current = self._load_tree(self.request.name, self.request.project)
        if self._fingerprint_tree(current) != self._fingerprint:
            return AgentOutcome(
                text="The saved topology or one of its shared components changed "
                "after approval was requested. Nothing ran; review and approve again.",
                exit_code=1,
                spent=0,
                detail={"topology": self.request.name, "model_calls": 0,
                        "refused": "design changed after consent"},
            )

        started = time.monotonic()
        self._made = 0
        outputs, error = self._execute(self._snapshot, self.request.prompt, brief, on_event)
        calls = sum(item["model_calls"] for item in outputs)
        text = error or "\n\n".join(
            f"{item['name']}: {item['text']}" for item in outputs
        )
        return AgentOutcome(
            text=text or "The topology completed without a report.",
            exit_code=1 if error else 0,
            elapsed_seconds=time.monotonic() - started,
            spent=0,
            detail={"topology": self.request.name, "topology_type": self._snapshot.kind,
                    "model_calls": calls, "max_model_calls": MAX_MODEL_REQUESTS,
                    "planned_model_calls": self._request_ceiling,
                    "model_call_count_unknown": any(
                        item.get("model_call_count_unknown") for item in outputs
                    ),
                    "components": [item["name"] for item in outputs]},
        )

    def _execute(
        self,
        node: _Design,
        task: str,
        brief: Brief,
        on_event: OnEvent | None,
        prior: str = "",
    ) -> tuple[list[dict[str, Any]], str | None]:
        if node.kind == TopologyType.CROSS_CHECK.value:
            outputs, error = self._cross_check(node, task, brief, on_event)
            own = "\n".join(f"{item['name']}: {item['text']}" for item in outputs)
            prior = f"{prior}\n{own}" if prior else own
        else:
            outputs, error = self._components(node, task, brief, on_event, prior)
            prior = "\n".join(
                [prior, *(f"{item['name']}: {item['text']}" for item in outputs)]
            )
        if error:
            return outputs, error
        for child in node.children:
            child_outputs, error = self._execute(
                child, task, brief, on_event, prior=prior
            )
            outputs.extend(child_outputs)
            if error:
                return outputs, error
            prior += "\n" + "\n".join(
                f"{item['name']}: {item['text']}" for item in child_outputs
            )
        if node.kind in {"debate", "ensemble", "council"} and outputs:
            return self._summarise(node, task, brief, on_event, outputs)
        return outputs, None

    def _account(self, calls: int | None) -> str | None:
        """Count requests as they are reported; refuse the next step past the ceiling."""
        self._made += calls or 0
        if self._made > MAX_MODEL_REQUESTS:
            return (f"Topology execution exceeded its {MAX_MODEL_REQUESTS}-request "
                    "ceiling and stopped.")
        return None

    def _cross_check(
        self, node: _Design, task: str, brief: Brief, on_event: OnEvent | None
    ) -> tuple[list[dict[str, Any]], str | None]:
        from qmcp.integrations.agents.crosscheck import CrossCheckRuntime

        checkers = node.components
        runtime = CrossCheckRuntime(
            task,
            runtime_factory=self.runtime_factory,
            perspectives=tuple(item["instruction"] for item in checkers) or None,
            checker_count=len(checkers) or int(node.config.get("num_checkers", 3)),
        )
        outcome = runtime.run(
            Brief(instruction=task, cwd=brief.cwd, project=brief.project),
            on_event=on_event,
        )
        calls = _request_count(outcome)
        result = _entry(
            node.name, outcome, calls,
            unknown=bool(outcome.detail.get("unknown_model_call_counts")),
        )
        if not outcome.succeeded:
            return [result], outcome.text
        if calls is None:
            return [result], "The crosscheck did not report a reliable model-request count."
        return [result], self._account(calls)

    def _components(
        self, node: _Design, task: str, brief: Brief,
        on_event: OnEvent | None, prior: str,
    ) -> tuple[list[dict[str, Any]], str | None]:
        outputs: list[dict[str, Any]] = []
        components = list(node.components)
        if node.kind == TopologyType.DELEGATION.value:
            components = _routed(node, task)
            if not components:
                return [], (
                    f"Topology {node.name} did not route the task: no component's "
                    "declared route terms matched it."
                )
        elif node.kind == TopologyType.COUNCIL.value:
            order = node.config.get("speaking_order", [])
            positions = {name: index for index, name in enumerate(order)}
            components.sort(key=lambda item: positions.get(item["name"], len(positions)))

        is_ensemble = node.kind == TopologyType.ENSEMBLE.value
        for component in components:
            instruction = (
                f"You are the {component['name']} component. {component['instruction']}\n\n"
                f"Task: {task}\n\n"
                "Provide analysis only. Do not choose, approve, authorize, modify "
                "files, run commands, or perform an act reserved for a person."
            )
            if prior and not is_ensemble:
                instruction += f"\n\nEarlier component reports:\n{prior}"
            outcome = self.runtime_factory().run(
                Brief(instruction=instruction, cwd=brief.cwd, project=brief.project),
                on_event=on_event,
            )
            calls = _request_count(outcome)
            outputs.append(_entry(component["name"], outcome, calls))
            if _unsound(outcome, calls):
                return outputs, (
                    f"Component {component['name']} failed, reported provider "
                    "spending, or did not report a request count within its "
                    "two-request limit."
                )
            if error := self._account(calls):
                return outputs, error
            if not is_ensemble:
                prior += f"\n{component['name']}: {outcome.text}"
        return outputs, None

    def _summarise(
        self, node: _Design, task: str, brief: Brief, on_event: OnEvent | None,
        outputs: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], str | None]:
        reports = "\n".join(f"{item['name']}: {item['text']}" for item in outputs)
        instruction = (
            f"Summarise these advisory reports about this task: {task}\n\n"
            f"{reports}\n\n"
            "Report agreement, disagreement and uncertainty. Do not choose, "
            "approve, adjudicate, or instruct anyone to perform a human-only act."
        )
        outcome = self.runtime_factory().run(
            Brief(instruction=instruction, cwd=brief.cwd, project=brief.project),
            on_event=on_event,
        )
        calls = _request_count(outcome)
        outputs.append(_entry(f"{node.name} advisory summary", outcome, calls))
        if _unsound(outcome, calls):
            return outputs, (
                f"The {node.kind} report synthesis failed, reported provider "
                "spending, or exceeded its two-request limit."
            )
        return outputs, self._account(calls)


__all__ = ["MAX_MODEL_REQUESTS", "NAME", "TopologyDesignRuntime"]
