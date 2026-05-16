from __future__ import annotations

from ..core.models import CommitChange, ProvenanceRecord, Symbol


def symbol_provenance(symbol: Symbol, reason: str, score: float) -> ProvenanceRecord:
    return ProvenanceRecord(
        source_type="symbol",
        identifier=symbol.name,
        file_path=symbol.file_path,
        start_line=symbol.start_line,
        end_line=symbol.end_line,
        reason=reason,
        score=round(score, 3),
    )


def commit_provenance(commit: CommitChange, score: float) -> ProvenanceRecord:
    file_path = commit.file_paths[0] if commit.file_paths else ""
    return ProvenanceRecord(
        source_type="commit",
        identifier=commit.commit_hash,
        file_path=file_path,
        start_line=1,
        end_line=1,
        reason=commit.message,
        score=round(score, 3),
    )
