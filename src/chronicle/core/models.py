from __future__ import annotations

from typing import Literal

from .pydantic_compat import BaseModel, ConfigDict, Field


SymbolType = Literal["function", "class", "method"]
Intent = Literal["search", "explain", "debug", "edit", "architecture"]
ChangeType = Literal["logic", "validation", "error_handling", "api_change", "refactor", "unknown"]
ModelClass = Literal["small", "medium", "large", "local"]
BenchmarkConfidence = Literal["low", "medium", "high"]
AgentRole = Literal["planner", "coder", "reviewer", "critic", "governance", "retriever"]


class ChronicleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Symbol(ChronicleModel):
    id: str
    name: str
    type: SymbolType
    file_path: str
    start_line: int
    end_line: int
    signature: str | None = None
    docstring: str | None = None
    calls: list[str] = Field(default_factory=list)
    imports: list[str] = Field(default_factory=list)
    content_hash: str = ""
    body: str = ""
    parent: str | None = None


class CommitChange(ChronicleModel):
    commit_hash: str
    message: str
    file_paths: list[str]
    symbols_changed: list[str]
    change_type: ChangeType
    risk_flags: list[str] = Field(default_factory=list)


class ProvenanceRecord(ChronicleModel):
    source_type: Literal["symbol", "commit"]
    identifier: str
    file_path: str
    start_line: int
    end_line: int
    reason: str
    score: float


class QueryPlan(ChronicleModel):
    intent: Intent
    keywords: list[str]
    candidate_symbols: list[str]
    candidate_files: list[str]
    needs_git_history: bool
    needs_runtime_context: bool = False
    needs_patch_context: bool = False


class LLMDecision(ChronicleModel):
    call_llm: bool
    reason: str
    model_class: ModelClass
    max_input_tokens: int
    max_output_tokens: int
    expected_value: str


class GuardrailCheck(ChronicleModel):
    safe_to_send: bool
    contains_secrets: bool
    redacted_text: str
    blocked_patterns: list[str] = Field(default_factory=list)
    redaction_count: int = 0


class SessionTurn(ChronicleModel):
    turn_id: str
    query: str
    intent: Intent
    token_budget: int
    estimated_tokens: int
    created_at: str
    selected_symbols: list[str] = Field(default_factory=list)
    selected_files: list[str] = Field(default_factory=list)
    excluded_symbols: list[str] = Field(default_factory=list)
    validation_confidence: float | None = None
    grounded: bool | None = None
    notes: list[str] = Field(default_factory=list)


class SessionMemory(ChronicleModel):
    session_id: str
    repo_path: str
    created_at: str
    updated_at: str
    turns: list[SessionTurn] = Field(default_factory=list)


class SessionMemoryHints(ChronicleModel):
    session_id: str
    prior_turn_count: int
    preferred_symbols: list[str] = Field(default_factory=list)
    preferred_files: list[str] = Field(default_factory=list)
    recent_queries: list[str] = Field(default_factory=list)
    validated_facts: list[str] = Field(default_factory=list)


class SessionMemorySummary(ChronicleModel):
    session_id: str
    prior_turn_count: int
    recalled_symbols: list[str] = Field(default_factory=list)
    recalled_files: list[str] = Field(default_factory=list)
    recent_queries: list[str] = Field(default_factory=list)
    validated_facts: list[str] = Field(default_factory=list)


class CallChainStep(ChronicleModel):
    symbol_id: str
    name: str
    file_path: str
    start_line: int


class CallChainReport(ChronicleModel):
    query: str
    max_depth: int
    entry_symbol: str | None = None
    chains: list[list[CallChainStep]] = Field(default_factory=list)
    mermaid: str = ""
    summary: str = ""


class PatchContextHints(ChronicleModel):
    changed_files: list[str] = Field(default_factory=list)
    changed_symbol_ids: list[str] = Field(default_factory=list)
    changed_symbol_names: list[str] = Field(default_factory=list)
    related_symbol_ids: list[str] = Field(default_factory=list)
    related_symbol_names: list[str] = Field(default_factory=list)
    related_test_files: list[str] = Field(default_factory=list)
    interface_files: list[str] = Field(default_factory=list)
    summary: str = ""


class LLMContextBrief(ChronicleModel):
    objective: str
    recommended_output: str
    focus_areas: list[str] = Field(default_factory=list)


class ContextPack(ChronicleModel):
    query: str
    token_budget: int
    selected_symbols: list[Symbol]
    selected_commits: list[CommitChange]
    compressed_context: str
    estimated_tokens: int
    provenance: list[ProvenanceRecord]
    confidence: float
    ranking_scores: dict[str, float] = Field(default_factory=dict)
    excluded_symbols: list[str] = Field(default_factory=list)
    llm_decision: LLMDecision | None = None
    session_id: str | None = None
    memory_summary: SessionMemorySummary | None = None
    call_chain: CallChainReport | None = None
    patch_context: PatchContextHints | None = None
    llm_brief: LLMContextBrief | None = None

    def to_markdown(self) -> str:
        lines = [
            f"# Context for: {self.query}",
            f"- Token budget: {self.token_budget}",
            f"- Estimated tokens: {self.estimated_tokens}",
            f"- Confidence: {self.confidence:.2f}",
        ]
        if self.session_id:
            lines.append(f"- Session: {self.session_id}")
        if self.memory_summary and self.memory_summary.prior_turn_count:
            lines.extend(
                [
                    "",
                    "## Session Memory",
                    f"- Prior turns: {self.memory_summary.prior_turn_count}",
                    f"- Recalled symbols: {', '.join(self.memory_summary.recalled_symbols) or 'none'}",
                    f"- Recalled files: {', '.join(self.memory_summary.recalled_files) or 'none'}",
                ]
            )
        if self.call_chain and self.call_chain.chains:
            lines.extend(
                [
                    "",
                    "## Call Chain",
                    self.call_chain.summary,
                ]
            )
        if self.patch_context and self.patch_context.summary:
            lines.extend(
                [
                    "",
                    "## Patch Context",
                    self.patch_context.summary,
                ]
            )
        if self.llm_brief:
            lines.extend(
                [
                    "",
                    "## LLM Brief",
                    f"- Objective: {self.llm_brief.objective}",
                    f"- Focus: {', '.join(self.llm_brief.focus_areas) or 'none'}",
                    f"- Output: {self.llm_brief.recommended_output}",
                ]
            )
        lines.extend(
            [
                "",
                "## Context",
                self.compressed_context.strip(),
                "",
                "## Provenance",
            ]
        )
        for record in self.provenance:
            lines.append(
                f"- {record.identifier} ({record.file_path}:{record.start_line}) — {record.reason} [{record.score:.2f}]"
            )
        return "\n".join(lines).strip()


class ValidationResult(ChronicleModel):
    valid: bool
    issues: list[str]
    grounded: bool
    confidence: float
    grounded_references: int = 0
    ungrounded_references: int = 0
    coverage_score: float = 0.0


class AgentPhaseState(ChronicleModel):
    role: AgentRole
    query: str
    token_budget: int
    context_pack: ContextPack
    created_at: str
    notes: list[str] = Field(default_factory=list)
    validation: ValidationResult | None = None
    llm_decision: LLMDecision | None = None


class AgentHandoffRecord(ChronicleModel):
    from_role: AgentRole
    to_role: AgentRole
    reason: str
    created_at: str


class MultiAgentContextBus(ChronicleModel):
    bus_id: str
    repo_path: str
    root_query: str
    created_at: str
    updated_at: str
    session_id: str | None = None
    phases: list[AgentPhaseState] = Field(default_factory=list)
    handoffs: list[AgentHandoffRecord] = Field(default_factory=list)


class EvaluationReport(ChronicleModel):
    baseline_tokens: int
    chronicle_tokens: int
    token_reduction_percent: float
    retrieval_hit_rate: float
    unused_context_percent: float
    answer_grounding_score: float
    estimated_cost_saved: float
    benchmark_confidence: BenchmarkConfidence
    recommendation: str


class IndexSnapshot(ChronicleModel):
    repo_path: str
    indexed_at: str
    symbols: list[Symbol]
    call_graph: dict[str, list[str]]
    dependency_graph: dict[str, list[str]]
    commit_changes: list[CommitChange]
    churn_by_file: dict[str, int]

    @classmethod
    def from_dict(cls, data: dict) -> "IndexSnapshot":
        return cls.model_validate(data)
