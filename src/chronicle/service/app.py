from __future__ import annotations

import os
from typing import Any

from ..api import Chronicle
from ..core.pydantic_compat import BaseModel, Field
from ..remote_repo import resolve_repo_path

try:
    from fastapi import FastAPI, Header, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError:  # pragma: no cover - exercised via create_app guard when dependency is absent.
    FastAPI = None  # type: ignore[assignment]
    Header = None  # type: ignore[assignment]
    HTTPException = RuntimeError  # type: ignore[assignment]
    HTMLResponse = None  # type: ignore[assignment]


class RepoRequest(BaseModel):
    repo: str = "."
    repo_url: str | None = None
    repos_dir: str | None = None
    branch: str | None = None
    index_dir: str | None = None


class QueryRequest(BaseModel):
    query: str
    repo: str = "."
    repo_url: str | None = None
    repos_dir: str | None = None
    branch: str | None = None
    index_dir: str | None = None
    token_budget: int | None = None
    session_id: str | None = None


class EvaluateRequest(BaseModel):
    query: str
    repo: str = "."
    repo_url: str | None = None
    repos_dir: str | None = None
    branch: str | None = None
    index_dir: str | None = None
    token_budget: int | None = None
    session_id: str | None = None
    baseline_token_budget: int | None = None


class CallChainRequest(BaseModel):
    query: str
    repo: str = "."
    repo_url: str | None = None
    repos_dir: str | None = None
    branch: str | None = None
    index_dir: str | None = None
    token_budget: int | None = None
    session_id: str | None = None
    max_depth: int = 4


def create_app() -> Any:
    if FastAPI is None:
        raise RuntimeError(
            "FastAPI dependencies are not installed. Use `pip install -e .[hosted]` to run the Chronicle service."
        )

    app = FastAPI(
        title="Chronicle API",
        version="0.1.0",
        summary="Hosted alpha API for grounded context planning on Python coding workflows.",
    )
    configured_api_key = os.getenv("CHRONICLE_API_KEY", "").strip()

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        auth_note = (
            "Protected mode is on. Enter the API key below."
            if configured_api_key
            else "Open mode is on. Add CHRONICLE_API_KEY when you want gated access."
        )
        api_key_note = "(required in protected mode)" if configured_api_key else "(optional in open mode)"
        template = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Chronicle</title>
    <style>
      :root {
        --bg: #07111c;
        --bg-soft: #0d1827;
        --surface: rgba(10, 19, 31, 0.88);
        --surface-strong: rgba(13, 24, 39, 0.96);
        --surface-muted: rgba(17, 31, 49, 0.88);
        --ink: #f6fbff;
        --muted: #8fa0b7;
        --line: rgba(173, 194, 222, 0.12);
        --line-strong: rgba(173, 194, 222, 0.22);
        --accent: #6ee7c8;
        --accent-deep: #2dc5a0;
        --accent-soft: rgba(110, 231, 200, 0.1);
        --accent-glow: rgba(110, 231, 200, 0.24);
        --success: #73e2a7;
        --warn: #f0c36c;
        --danger: #f08f9b;
        --shadow-lg: 0 28px 70px rgba(2, 7, 14, 0.52);
        --shadow-sm: 0 16px 34px rgba(2, 7, 14, 0.34);
        --radius-xl: 24px;
        --radius-lg: 18px;
        --radius-md: 14px;
        --space-1: 6px;
        --space-2: 10px;
        --space-3: 14px;
        --space-4: 18px;
        --space-5: 24px;
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        color: var(--ink);
        background:
          radial-gradient(circle at top left, rgba(110, 231, 200, 0.14), transparent 26%),
          radial-gradient(circle at 82% 0%, rgba(69, 130, 255, 0.14), transparent 28%),
          linear-gradient(180deg, rgba(255,255,255,0.03), transparent 24%),
          var(--bg);
        font-family: "Inter Tight", "Avenir Next", "Segoe UI", sans-serif;
      }
      body::before {
        content: "";
        position: fixed;
        inset: 0;
        pointer-events: none;
        background-image:
          linear-gradient(rgba(173, 194, 222, 0.035) 1px, transparent 1px),
          linear-gradient(90deg, rgba(173, 194, 222, 0.035) 1px, transparent 1px);
        background-size: 28px 28px;
        mask-image: linear-gradient(180deg, rgba(0,0,0,0.55), transparent 75%);
      }
      main {
        max-width: 1280px;
        margin: 0 auto;
        padding: 18px;
        position: relative;
      }
      .app-shell { display: grid; gap: var(--space-4); align-items: start; }
      .top-shell {
        display: grid;
        grid-template-columns: minmax(0, 1.1fr) minmax(360px, 0.9fr);
        gap: var(--space-4);
        align-items: start;
      }
      .panel {
        background: var(--surface);
        backdrop-filter: blur(10px);
        border: 1px solid var(--line);
        border-radius: var(--radius-xl);
        box-shadow: var(--shadow-lg);
        position: relative;
        overflow: hidden;
      }
      .panel::before {
        content: "";
        position: absolute;
        inset: 0 0 auto 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--accent-glow), transparent);
      }
      .panel-inner {
        padding: var(--space-4);
        position: relative;
        z-index: 1;
      }
      .hero-stack {
        display: grid;
        gap: var(--space-4);
      }
      .brand-badge {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        border-radius: 999px;
        background: rgba(110, 231, 200, 0.08);
        color: var(--accent-deep);
        border: 1px solid rgba(110, 231, 200, 0.16);
        font-size: 12px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        font-weight: 700;
        width: fit-content;
      }
      .brand-mark {
        width: 28px;
        height: 28px;
        border-radius: 10px;
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%);
        color: #07111c;
        display: grid;
        place-items: center;
        font-size: 12px;
        box-shadow: var(--shadow-sm);
      }
      h1 {
        font-family: "Söhne Breit", "Space Grotesk", "Inter Tight", sans-serif;
        font-size: clamp(34px, 4vw, 50px);
        line-height: 0.96;
        margin: 0;
        letter-spacing: -0.05em;
        text-wrap: balance;
        max-width: 15ch;
      }
      h2 {
        margin: 0;
        font-size: 18px;
        letter-spacing: -0.01em;
      }
      h3 {
        margin: 0;
        font-size: 15px;
        letter-spacing: -0.01em;
      }
      p {
        line-height: 1.55;
        color: var(--muted);
        margin: 0;
      }
      strong { color: var(--ink); }
      .hero-copy {
        font-size: 16px;
        max-width: 58ch;
      }
      .summary-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: var(--space-3);
      }
      .summary-tile {
        background: var(--surface-muted);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: var(--space-3);
        min-width: 0;
        transition: border-color 160ms ease, transform 160ms ease;
      }
      .summary-tile:hover {
        border-color: var(--line-strong);
        transform: translateY(-1px);
      }
      .summary-tile b {
        display: block;
        font-size: 16px;
        margin-bottom: 4px;
        color: var(--ink);
      }
      .summary-tile span {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
      }
      .section-stack {
        display: grid;
        gap: var(--space-3);
      }
      .workspace-shell {
        display: grid;
        gap: var(--space-4);
      }
      .button-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-3);
      }
      .button, button {
        appearance: none;
        border: 0;
        border-radius: var(--radius-md);
        padding: 11px 15px;
        font-size: 14px;
        cursor: pointer;
        text-decoration: none;
        transition: transform 160ms ease, box-shadow 160ms ease, background 160ms ease, border-color 160ms ease;
      }
      .button:hover, button:hover {
        transform: translateY(-1px);
      }
      .button.primary, button {
        background: linear-gradient(135deg, var(--accent) 0%, var(--accent-deep) 100%);
        color: #07111c;
        font-weight: 700;
        box-shadow: 0 14px 28px rgba(45, 197, 160, 0.2);
      }
      .button.secondary {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid var(--line);
        color: var(--ink);
      }
      .meta-grid {
        display: grid;
        gap: var(--space-3);
      }
      .info-card {
        background: var(--surface-strong);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: var(--space-3);
      }
      .meta-list {
        display: grid;
        gap: 0;
      }
      .meta-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-3);
        padding: 10px 0;
        border-top: 1px solid var(--line);
        min-width: 0;
      }
      .meta-row:first-child {
        border-top: 0;
        padding-top: 0;
      }
      .meta-row:last-child {
        padding-bottom: 0;
      }
      .meta-copy b {
        display: block;
        font-size: 14px;
        margin-bottom: 2px;
      }
      .meta-copy span {
        display: block;
        color: var(--muted);
        font-size: 13px;
        line-height: 1.45;
        overflow-wrap: anywhere;
      }
      .meta-value {
        color: var(--ink);
        font-weight: 600;
        font-size: 13px;
        white-space: nowrap;
        flex-shrink: 0;
      }
      .workspace-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 320px;
        gap: var(--space-4);
      }
      code, pre {
        background: rgba(255, 255, 255, 0.04);
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: 12px;
      }
      code { padding: 2px 6px; }
      pre { padding: 16px; overflow: auto; white-space: pre-wrap; font-size: 13px; line-height: 1.55; }
      a { color: var(--accent-deep); text-decoration: none; }
      label {
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 6px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
      }
      input, select, textarea {
        width: 100%;
        background: rgba(255, 255, 255, 0.04);
        color: var(--ink);
        border: 1px solid var(--line);
        border-radius: 14px;
        padding: 11px 13px;
        font-size: 14px;
        outline: none;
      }
      input::placeholder, textarea::placeholder {
        color: rgba(143, 160, 183, 0.82);
      }
      input:focus, select:focus, textarea:focus {
        border-color: rgba(110, 231, 200, 0.45);
        box-shadow: 0 0 0 4px rgba(110, 231, 200, 0.08);
      }
      textarea {
        min-height: 82px;
        max-height: 120px;
        resize: vertical;
      }
      .field-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) 144px 144px;
        gap: var(--space-3);
      }
      .input-stack { display: grid; gap: var(--space-3); }
      .helper-copy {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.5;
      }
      .subtle {
        font-size: 13px;
        color: var(--muted);
      }
      .result-shell {
        display: grid;
        gap: var(--space-3);
      }
      .result-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-3);
      }
      .empty-state {
        min-height: 260px;
        display: grid;
        place-items: center;
        background: rgba(255, 255, 255, 0.03);
        border: 1px dashed var(--line-strong);
        border-radius: var(--radius-lg);
        text-align: center;
        padding: var(--space-4);
      }
      .empty-state b {
        display: block;
        margin-bottom: 8px;
        font-size: 16px;
      }
      .loading-state {
        min-height: 260px;
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
      }
      .error-state {
        min-height: 220px;
        background: rgba(240, 143, 155, 0.06);
        border: 1px solid rgba(240, 143, 155, 0.18);
        border-radius: var(--radius-lg);
        padding: var(--space-4);
        display: grid;
        align-content: start;
        gap: 12px;
      }
      .error-state b {
        color: var(--danger);
        font-size: 16px;
      }
      .loading-bars {
        display: grid;
        gap: 10px;
      }
      .loading-bar {
        height: 12px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(15, 108, 99, 0.08), rgba(15, 108, 99, 0.18), rgba(15, 108, 99, 0.08));
        background-size: 200% 100%;
        animation: shimmer 1.6s linear infinite;
      }
      .loading-bar.short { width: 42%; }
      .loading-bar.medium { width: 68%; }
      .loading-bar.long { width: 100%; }
      .result-overview {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: var(--space-2);
      }
      .result-mini {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--line);
        border-radius: var(--radius-md);
        padding: 12px;
        min-width: 0;
      }
      .result-mini h4 {
        margin: 0 0 6px;
        font-size: 12px;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .result-mini p {
        margin: 0;
        font-size: 13px;
        color: var(--ink);
        line-height: 1.45;
        overflow-wrap: anywhere;
      }
      .kpi {
        display: grid;
        grid-template-columns: 1fr;
        gap: 6px;
      }
      .kpi b {
        font-size: 16px;
        letter-spacing: -0.02em;
      }
      .kpi span {
        font-size: 12px;
        color: var(--muted);
      }
      .divider {
        height: 1px;
        background: var(--line);
        margin: 10px 0;
      }
      .muted { color: var(--muted); }
      .small { font-size: 12px; }
      .result-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) minmax(0, 0.88fr);
        gap: var(--space-3);
      }
      .result-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid var(--line);
        border-radius: var(--radius-lg);
        padding: 14px;
        min-width: 0;
      }
      .result-card h3 {
        margin: 0 0 6px;
        font-size: 15px;
        color: var(--ink);
        font-family: "Söhne Breit", "Space Grotesk", "Inter Tight", sans-serif;
      }
      .result-card p, .result-card li {
        font-size: 13px;
        overflow-wrap: anywhere;
      }
      .inline-link {
        background: transparent;
        border: 0;
        padding: 0;
        color: var(--accent-deep);
        font-size: 12px;
        font-weight: 700;
        cursor: pointer;
        text-decoration: underline;
        text-underline-offset: 3px;
      }
      .inline-link:hover {
        transform: none;
        box-shadow: none;
        color: var(--accent);
      }
      .modal-backdrop {
        position: fixed;
        inset: 0;
        display: none;
        align-items: center;
        justify-content: center;
        padding: var(--space-4);
        background: rgba(3, 8, 15, 0.72);
        backdrop-filter: blur(10px);
        z-index: 60;
      }
      .modal-backdrop.is-open {
        display: flex;
      }
      .modal {
        width: min(980px, 100%);
        max-height: min(84vh, 900px);
        display: grid;
        grid-template-rows: auto 1fr;
        background: var(--surface-strong);
        border: 1px solid var(--line-strong);
        border-radius: 22px;
        box-shadow: var(--shadow-lg);
        overflow: hidden;
      }
      .modal-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: var(--space-3);
        padding: 18px 20px;
        border-bottom: 1px solid var(--line);
      }
      .modal-head h3 {
        margin: 0;
        font-size: 15px;
        letter-spacing: -0.02em;
      }
      .modal-close {
        width: 36px;
        height: 36px;
        border-radius: 10px;
        border: 1px solid var(--line);
        background: rgba(255, 255, 255, 0.04);
        color: var(--ink);
        font-size: 20px;
        line-height: 1;
        display: grid;
        place-items: center;
        cursor: pointer;
      }
      .modal-body {
        overflow: auto;
        padding: 20px;
      }
      .modal-body pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        color: var(--ink);
        font-size: 12px;
        line-height: 1.55;
      }
      .kv {
        display: grid;
        gap: 8px;
        margin-top: 10px;
      }
      .kv-row {
        display: grid;
        grid-template-columns: 92px minmax(0, 1fr);
        gap: 10px;
        align-items: start;
        min-width: 0;
      }
      .kv-row strong {
        font-size: 12px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        color: var(--muted);
      }
      .kv-row span {
        color: var(--ink);
        line-height: 1.45;
        overflow-wrap: anywhere;
      }
      .pill {
        display: inline-block;
        padding: 4px 9px;
        border-radius: 999px;
        font-size: 11px;
        font-weight: 700;
        margin-right: 6px;
        margin-bottom: 6px;
        background: rgba(110, 231, 200, 0.08);
        color: var(--ink);
        border: 1px solid rgba(110, 231, 200, 0.16);
        max-width: 100%;
        overflow-wrap: anywhere;
      }
      .pill.good {
        background: rgba(31, 122, 76, 0.1);
        border-color: rgba(31, 122, 76, 0.18);
        color: var(--success);
      }
      .pill.warn {
        background: rgba(183, 121, 31, 0.12);
        border-color: rgba(183, 121, 31, 0.2);
        color: var(--warn);
      }
      .pill.bad {
        background: rgba(176, 63, 69, 0.12);
        border-color: rgba(176, 63, 69, 0.18);
        color: #8f2f34;
      }
      .footer-note { margin-top: 10px; font-size: 13px; color: var(--muted); }
      details summary {
        cursor: pointer;
        color: var(--accent-deep);
        font-size: 13px;
        font-weight: 600;
      }
      .eyebrow {
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--accent);
        font-weight: 700;
      }
      details > pre {
        margin-top: 10px;
      }
      @keyframes shimmer {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
      }
      @media (max-width: 900px) {
        .top-shell,
        .summary-grid,
        .workspace-grid,
        .field-grid,
        .result-overview,
        .result-grid { grid-template-columns: 1fr; }
        .result-grid { grid-template-columns: 1fr; }
        .button-row {
          align-items: flex-start;
          flex-direction: column;
        }
        .meta-row,
        .result-header {
          align-items: flex-start;
          flex-direction: column;
        }
        h1 { font-size: 36px; max-width: none; }
      }
    </style>
  </head>
  <body>
    <main>
      <div class="app-shell">
        <div class="top-shell">
          <section class="hero-stack">
            <div class="panel">
              <div class="panel-inner section-stack">
              <div class="brand-badge">
                <span class="brand-mark">Ch</span>
                <span>Chronicle</span>
              </div>
              <span class="eyebrow">Context orchestration for coding workflows</span>
              <h1>Sharper context. Lower spend. Grounded output.</h1>
              <p class="hero-copy">Map the code that matters, compress it into a usable context pack, and decide whether an LLM call is worth the tokens before you ship.</p>
              <div class="summary-grid">
                <div class="summary-tile"><b>Python-first</b><span>Built for structured <code>.py</code> repos right now</span></div>
                <div class="summary-tile"><b>Grounded</b><span>Symbols, flow, provenance, and patch-aware hints</span></div>
                <div class="summary-tile"><b>Measured</b><span>Routing, budget checks, and token-savings evaluation</span></div>
              </div>
              <p class="subtle">Best for agent workflows, patch planning, architecture questions, and token-efficiency reviews on Python repos.</p>
              </div>
            </div>
          </section>

          <aside class="meta-grid">
            <div class="panel" id="result-model">
              <div class="panel-inner">
                <h2>Result model</h2>
                <div class="info-card" style="margin-top:14px;">
                  <div class="meta-list">
                    <div class="meta-row">
                      <div class="meta-copy">
                        <b>Signal</b>
                        <span>What code area matches the question and how confident that match is.</span>
                      </div>
                      <div class="meta-value">Fast read</div>
                    </div>
                    <div class="meta-row">
                      <div class="meta-copy">
                        <b>Route</b>
                        <span>Whether the current match is strong enough for an LLM call.</span>
                      </div>
                      <div class="meta-value">Decision</div>
                    </div>
                    <div class="meta-row">
                      <div class="meta-copy">
                        <b>Payload</b>
                        <span>What context would be sent, trimmed to the smallest useful slice.</span>
                      </div>
                      <div class="meta-value">Compression</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div class="panel">
              <div class="panel-inner">
                <div class="meta-list">
                  <div class="meta-row">
                    <div class="meta-copy">
                      <b>Doctor</b>
                      <span>Check support, retrieval strength, and likely code areas.</span>
                    </div>
                    <div class="meta-value">Repo fit</div>
                  </div>
                  <div class="meta-row">
                    <div class="meta-copy">
                      <b>Demo</b>
                      <span>Summarize route, context plan, and token impact.</span>
                    </div>
                    <div class="meta-value">Decision</div>
                  </div>
                  <div class="meta-row">
                    <div class="meta-copy">
                      <b>Context / Call chain / Evaluate</b>
                      <span>Inspect grounded context, flow, and estimated savings.</span>
                    </div>
                    <div class="meta-value">Core</div>
                  </div>
                </div>
              </div>
            </div>
          </aside>
        </div>

        <section class="panel" style="margin-top: var(--space-4);">
          <div class="panel-inner workspace-shell" id="try">
            <div class="section-stack">
              <h2>Workspace</h2>
              <p>Paste a repository URL, choose a view, and inspect the route before sending anything expensive to a model.</p>
            </div>
            <div class="field-grid">
              <div>
                <label for="repo_url">Repository URL</label>
                <input id="repo_url" value="https://github.com/pallets/flask.git" />
              </div>
              <div>
                <label for="token_budget">Token budget</label>
                <input id="token_budget" type="number" value="2500" />
              </div>
              <div>
                <label for="action">View</label>
                <select id="action">
                  <option value="demo">Demo</option>
                  <option value="doctor">Doctor</option>
                  <option value="context">Context</option>
                  <option value="evaluate">Evaluate</option>
                  <option value="call-chain">Call chain</option>
                </select>
              </div>
            </div>
            <div class="input-stack">
              <div>
                <label for="api_key">API key __API_KEY_NOTE__</label>
                <input id="api_key" type="password" placeholder="Paste CHRONICLE_API_KEY if required" />
              </div>
              <div>
                <label for="query">Question</label>
                <textarea id="query">Where is full_dispatch_request defined?</textarea>
              </div>
            </div>
            <div class="button-row">
              <button id="run_demo" type="button">Run demo</button>
              <span class="helper-copy">Signal, route, savings, and payload in one compact view.</span>
            </div>
            <div class="result-shell">
              <div class="result-header">
                <label for="result_box">Result</label>
                <span class="subtle">Compact by default. Details expand only when needed.</span>
              </div>
              <div id="result_box" class="empty-state">
                <div>
                  <b>Ready</b>
                  <p>Run a repo question to see grounded context, routing, and token impact.</p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </div>
      <div id="raw_payload_modal" class="modal-backdrop" aria-hidden="true">
        <div class="modal" role="dialog" aria-modal="true" aria-labelledby="raw_payload_title">
          <div class="modal-head">
            <h3 id="raw_payload_title">Raw payload</h3>
            <button id="raw_payload_close" class="modal-close" type="button" aria-label="Close modal">×</button>
          </div>
          <div class="modal-body">
            <pre id="raw_payload_content"></pre>
          </div>
        </div>
      </div>

      <script>
        const button = document.getElementById("run_demo");
        const resultBox = document.getElementById("result_box");
        const rawPayloadModal = document.getElementById("raw_payload_modal");
        const rawPayloadContent = document.getElementById("raw_payload_content");
        const rawPayloadTitle = document.getElementById("raw_payload_title");
        const rawPayloadClose = document.getElementById("raw_payload_close");
        let activeRawPayload = null;
        let activeRawTitle = "Raw payload";
        function openRawPayloadModal() {
          if (!activeRawPayload || !rawPayloadModal || !rawPayloadContent || !rawPayloadTitle) return;
          rawPayloadTitle.textContent = activeRawTitle;
          rawPayloadContent.innerHTML = renderJson(activeRawPayload);
          rawPayloadModal.classList.add("is-open");
          rawPayloadModal.setAttribute("aria-hidden", "false");
          document.body.style.overflow = "hidden";
        }
        function closeRawPayloadModal() {
          if (!rawPayloadModal) return;
          rawPayloadModal.classList.remove("is-open");
          rawPayloadModal.setAttribute("aria-hidden", "true");
          document.body.style.overflow = "";
        }
        if (rawPayloadClose) rawPayloadClose.addEventListener("click", closeRawPayloadModal);
        if (rawPayloadModal) {
          rawPayloadModal.addEventListener("click", event => {
            if (event.target === rawPayloadModal) closeRawPayloadModal();
          });
        }
        document.addEventListener("keydown", event => {
          if (event.key === "Escape") closeRawPayloadModal();
        });
        function renderEmptyState() {
          return `
            <div class="empty-state">
              <div>
                <b>Ready</b>
                <p>Run a repo question to see grounded context, routing, and token impact.</p>
              </div>
            </div>
          `;
        }
        function renderLoadingState(action) {
          return `
            <div class="loading-state">
              <div class="loading-bars">
                <div class="loading-bar short"></div>
                <div class="loading-bar long"></div>
                <div class="loading-bar medium"></div>
                <div class="loading-bar long"></div>
              </div>
              <p class="footer-note" style="margin-top:14px;">Running ${escapeHtml(action)} and building the compact result view.</p>
            </div>
          `;
        }
        function renderErrorState(message, payload = null) {
          return `
            <div class="error-state">
              <div>
                <b>Request blocked</b>
                <p>${escapeHtml(message)}</p>
              </div>
              ${payload ? `<details><summary>Response details</summary>${renderJson(payload)}</details>` : ""}
            </div>
          `;
        }
        function escapeHtml(value) {
          return String(value ?? "")
            .replaceAll("&", "&amp;")
            .replaceAll("<", "&lt;")
            .replaceAll(">", "&gt;");
        }
        function renderPills(items, klass = "") {
          if (!items || !items.length) return "<span class='muted'>None</span>";
          return items.map(item => `<span class="pill ${klass}">${escapeHtml(item)}</span>`).join("");
        }
        function renderJson(payload) {
          return `<pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>`;
        }
        function renderDemo(payload) {
          const readiness = payload.llm_readiness || {};
          const preview = readiness.payload_preview || {};
          const evaluation = payload.evaluation || {};
          const symbols = (preview.selected_symbols || []).map(symbol => `${symbol.name} (${symbol.file_path}:${symbol.start_line})`);
          const sendClass = readiness.send_to_llm ? "good" : "warn";
          const sendText = readiness.send_to_llm ? "Send" : "Hold";
          const contextSummary = preview.focus_summary || "No compact payload summary available.";
          const rawContext = preview.context_preview || "No context preview available.";
          return `
            <div class="result-overview">
              <div class="result-mini">
                <h4>Signal</h4>
                <div class="kpi">
                  <b>${escapeHtml(payload.human_summary || "No summary available.")}</b>
                  <span>${escapeHtml(payload.repo_insight || "")}</span>
                </div>
              </div>
              <div class="result-mini">
                <h4>Route</h4>
                <div class="kpi">
                  <b><span class="pill ${sendClass}">${escapeHtml(sendText)}</span></b>
                  <span>${escapeHtml(readiness.reason || "No recommendation available.")}</span>
                </div>
              </div>
              <div class="result-mini">
                <h4>Savings</h4>
                <div class="kpi">
                  <b>${escapeHtml(String(evaluation.token_reduction_percent ?? "?"))}%</b>
                  <span>${escapeHtml(String(evaluation.baseline_tokens ?? "?"))} → ${escapeHtml(String(evaluation.chronicle_tokens ?? "?"))} tokens, ${escapeHtml(String(evaluation.benchmark_confidence ?? "unknown"))} confidence</span>
                </div>
              </div>
            </div>
            <div class="result-grid">
              <div class="result-card">
                <h3>Summary</h3>
                <p>${escapeHtml(payload.human_summary || "No summary available.")}</p>
                <div class="divider"></div>
                <div class="kv">
                  <div class="kv-row">
                    <strong>Repo</strong>
                    <span>${escapeHtml(payload.repo_insight || "No repo-specific insight available yet.")}</span>
                  </div>
                  <div class="kv-row">
                    <strong>Symbols</strong>
                    <span>${renderPills(symbols)}</span>
                  </div>
                  <div class="kv-row">
                    <strong>Next</strong>
                    <span>${escapeHtml(readiness.recommended_next_step || "N/A")}</span>
                  </div>
                </div>
              </div>
              <div class="result-card">
                <h3>Payload</h3>
                <div class="kv">
                  <div class="kv-row">
                    <strong>Query</strong>
                    <span>${escapeHtml(preview.query || "")}</span>
                  </div>
                  <div class="kv-row">
                    <strong>Focus</strong>
                    <span>${escapeHtml(contextSummary.slice(0, 120))}${contextSummary.length > 120 ? ` <span class="muted">...[truncated]</span> <button type="button" class="inline-link" id="open_raw_payload_link">open modal</button>` : ""}</span>
                  </div>
                  <div class="kv-row">
                    <strong>Plan</strong>
                    <span>${escapeHtml(readiness.query_strategy || "N/A")}</span>
                  </div>
                  <div class="kv-row">
                    <strong>Context</strong>
                    <span>${escapeHtml(readiness.context_strategy || "N/A")}</span>
                  </div>
                </div>
                <details style="margin-top:10px;">
                  <summary>Context preview</summary>
                  <pre>${escapeHtml(rawContext)}</pre>
                </details>
                ${preview.prompt_preview ? `<details style="margin-top:10px;"><summary>Prompt preview</summary><pre>${escapeHtml(preview.prompt_preview)}</pre></details>` : ""}
              </div>
            </div>
          `;
        }
        function renderDoctor(payload) {
          const matches = (payload.top_matches || []).map(match => `${match.name} (${match.location})`);
          return `
            <div class="result-grid">
              <div class="result-card">
                <h3>Health</h3>
                <div class="result-overview" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
                  <div class="result-mini">
                    <h4>Status</h4>
                    <div class="kpi"><b>${escapeHtml(payload.health?.status || "unknown")}</b><span>${escapeHtml(payload.health?.message || "")}</span></div>
                  </div>
                  <div class="result-mini">
                    <h4>Match</h4>
                    <div class="kpi"><b>${escapeHtml(payload.query_diagnosis?.status || "unknown")}</b><span>${escapeHtml(payload.query_diagnosis?.message || "")}</span></div>
                  </div>
                  <div class="result-mini">
                    <h4>Index</h4>
                    <div class="kpi"><b>${escapeHtml(String(payload.symbol_count ?? "?"))} symbols</b><span>${escapeHtml(String(payload.python_file_count ?? "?"))} files</span></div>
                  </div>
                </div>
                <div class="divider"></div>
                <p class="small muted">${escapeHtml(payload.human_summary || "No summary available.")}</p>
              </div>
              <div class="result-card">
                <h3>Likely Areas</h3>
                <p>${renderPills(matches)}</p>
                <div class="divider"></div>
                <p class="small"><strong class="muted">Selected:</strong> ${escapeHtml((payload.selected_symbols || []).join(", ") || "None")}</p>
              </div>
            </div>
          `;
        }
        function renderContext(payload) {
          const symbols = (payload.selected_symbols || []).map(symbol => `${symbol.name} (${symbol.file_path}:${symbol.start_line})`);
          return `
            <div class="result-grid">
              <div class="result-card">
                <h3>Context</h3>
                <div class="result-overview" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
                  <div class="result-mini">
                    <h4>Confidence</h4>
                    <div class="kpi"><b>${escapeHtml(String(payload.confidence ?? "?"))}</b><span>${escapeHtml(String(payload.estimated_tokens ?? "?"))} tokens</span></div>
                  </div>
                  <div class="result-mini">
                    <h4>Route</h4>
                    <div class="kpi"><b>${payload.llm_decision?.call_llm ? "Send" : "Hold"}</b><span>${escapeHtml(payload.llm_decision?.reason || "No decision available.")}</span></div>
                  </div>
                  <div class="result-mini">
                    <h4>Coverage</h4>
                    <div class="kpi"><b>${escapeHtml(String((payload.selected_symbols || []).length))} symbols</b><span class="muted">compressed pack</span></div>
                  </div>
                </div>
                <div class="divider"></div>
                <p>${escapeHtml(payload.human_summary || "No summary available.")}</p>
                <p>${renderPills(symbols)}</p>
              </div>
              <div class="result-card">
                <h3>Details</h3>
                <p class="small muted">${escapeHtml(payload.llm_decision?.reason || "No decision available.")}</p>
                ${payload.compressed_context ? `<details style="margin-top:10px;"><summary>Context preview</summary><pre>${escapeHtml(String(payload.compressed_context).slice(0, 900) + (String(payload.compressed_context).length > 900 ? "\\n\\n...[truncated]" : ""))}</pre></details>` : ""}
              </div>
            </div>
          `;
        }
        function renderEvaluate(payload) {
          return `
            <div class="result-grid">
              <div class="result-card">
                <h3>Evaluation</h3>
                <div class="result-overview" style="grid-template-columns: repeat(3, minmax(0, 1fr));">
                  <div class="result-mini">
                    <h4>Savings</h4>
                    <div class="kpi"><b>${escapeHtml(String(payload.token_reduction_percent ?? "?"))}%</b><span>${escapeHtml(String(payload.baseline_tokens ?? "?"))} → ${escapeHtml(String(payload.chronicle_tokens ?? "?"))} tokens</span></div>
                  </div>
                  <div class="result-mini">
                    <h4>Confidence</h4>
                    <div class="kpi"><b>${escapeHtml(String(payload.benchmark_confidence ?? "unknown"))}</b><span>${escapeHtml(payload.recommendation || "")}</span></div>
                  </div>
                  <div class="result-mini">
                    <h4>Grounding</h4>
                    <div class="kpi"><b>${escapeHtml(String(payload.answer_grounding_score ?? "?"))}</b><span class="muted">estimator</span></div>
                  </div>
                </div>
                <div class="divider"></div>
                <p>${escapeHtml(payload.human_summary || "No summary available.")}</p>
              </div>
              <div class="result-card">
                <h3>Metrics</h3>
                <p>${renderPills([
                  `Baseline ${payload.baseline_tokens ?? "?"}`,
                  `Focused ${payload.chronicle_tokens ?? "?"}`,
                  `${payload.token_reduction_percent ?? "?"}% saved`,
                  `Grounding ${payload.answer_grounding_score ?? "?"}`
                ])}</p>
              </div>
            </div>
          `;
        }
        function renderCallChain(payload) {
          return `
            <div class="result-grid">
              <div class="result-card">
                <h3>Call Chain</h3>
                <p>${escapeHtml(payload.human_summary || "No summary available.")}</p>
                <pre>${escapeHtml(payload.summary || "No call chain available.")}</pre>
              </div>
              <div class="result-card">
                <h3>Entry</h3>
                <p>${escapeHtml(payload.entry_symbol || "No clear entry symbol.")}</p>
                <p>${renderPills((payload.selected_symbols || []).map(symbol => `${symbol.name}`))}</p>
              </div>
            </div>
          `;
        }
        button.addEventListener("click", async () => {
          const repoUrl = document.getElementById("repo_url").value.trim();
          const tokenBudget = Number(document.getElementById("token_budget").value || "2500");
          const query = document.getElementById("query").value.trim();
          const action = document.getElementById("action").value;
          const apiKey = document.getElementById("api_key").value.trim();
          resultBox.innerHTML = renderLoadingState(action);
          try {
            const response = await fetch("/" + action, {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
                ...(apiKey ? { "X-API-Key": apiKey } : {})
              },
              body: JSON.stringify({
                repo_url: repoUrl,
                query,
                token_budget: tokenBudget
              })
            });
            const payload = await response.json();
            if (!response.ok) {
              resultBox.innerHTML = renderErrorState(payload.detail || payload.message || "The request could not be completed.", payload);
              return;
            }
            if (action === "demo") {
              resultBox.innerHTML = renderDemo(payload);
              activeRawPayload = payload;
              activeRawTitle = "Raw payload";
              const openLink = document.getElementById("open_raw_payload_link");
              if (openLink) openLink.addEventListener("click", openRawPayloadModal);
            } else if (action === "doctor") {
              resultBox.innerHTML = renderDoctor(payload);
              activeRawPayload = payload;
              activeRawTitle = "Raw payload";
            } else if (action === "context") {
              resultBox.innerHTML = renderContext(payload);
              activeRawPayload = payload;
              activeRawTitle = "Raw payload";
            } else if (action === "evaluate") {
              resultBox.innerHTML = renderEvaluate(payload);
              activeRawPayload = payload;
              activeRawTitle = "Raw payload";
            } else if (action === "call-chain") {
              resultBox.innerHTML = renderCallChain(payload);
              activeRawPayload = payload;
              activeRawTitle = "Raw payload";
            } else {
              resultBox.innerHTML = renderJson(payload);
              activeRawPayload = payload;
              activeRawTitle = "Raw payload";
            }
          } catch (error) {
            resultBox.innerHTML = renderErrorState("Request failed: " + (error && error.message ? error.message : String(error)));
          }
        });
      </script>
    </main>
  </body>
</html>"""
        return template.replace("__AUTH_NOTE__", auth_note).replace("__API_KEY_NOTE__", api_key_note)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "chronicle-api"}

    @app.post("/index")
    def index_repo(request: RepoRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(configured_api_key, x_api_key)
        chronicle = _chronicle_from_request(request)
        snapshot = chronicle.index()
        return {
            "repo": str(chronicle.config.repo_path),
            "index_dir": str(chronicle.config.index_dir),
            "symbol_count": len(snapshot.symbols),
            "commit_change_count": len(snapshot.commit_changes),
            "call_graph_nodes": len(snapshot.call_graph),
            "dependency_graph_nodes": len(snapshot.dependency_graph),
        }

    @app.post("/doctor")
    def doctor(request: QueryRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(configured_api_key, x_api_key)
        chronicle = _chronicle_from_request(request)
        return chronicle.diagnose(
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id,
        )

    @app.post("/demo")
    def demo(request: QueryRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(configured_api_key, x_api_key)
        chronicle = _chronicle_from_request(request)
        return chronicle.demo(
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id,
        )

    @app.post("/context")
    def context(request: QueryRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(configured_api_key, x_api_key)
        chronicle = _chronicle_from_request(request)
        return chronicle.context(
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id,
        ).model_dump()

    @app.post("/evaluate")
    def evaluate(request: EvaluateRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(configured_api_key, x_api_key)
        chronicle = _chronicle_from_request(request)
        report = chronicle.evaluate(
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id,
        )
        return {
            "repo": str(chronicle.config.repo_path),
            "index_dir": str(chronicle.config.index_dir),
            "query": request.query,
            **report.model_dump(),
        }

    @app.post("/call-chain")
    def call_chain(request: CallChainRequest, x_api_key: str | None = Header(default=None)) -> dict[str, Any]:
        _require_api_key(configured_api_key, x_api_key)
        chronicle = _chronicle_from_request(request)
        return chronicle.call_chain(
            query=request.query,
            token_budget=request.token_budget,
            session_id=request.session_id,
            max_depth=request.max_depth,
        )

    return app


def _chronicle_from_request(request: RepoRequest) -> Chronicle:
    try:
        repo_path = resolve_repo_path(
            repo=request.repo,
            repo_url=request.repo_url,
            repos_dir=request.repos_dir,
            branch=request.branch,
        )
        return Chronicle(repo_path=repo_path, index_dir=request.index_dir)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _require_api_key(configured_api_key: str, provided_api_key: str | None) -> None:
    if not configured_api_key:
        return
    if provided_api_key == configured_api_key:
        return
    raise HTTPException(
        status_code=401,
        detail="Missing or invalid API key. Send `X-API-Key` for protected Chronicle endpoints.",
    )


if FastAPI is not None:
    app = create_app()
else:  # pragma: no cover - import-time fallback for environments without hosted extras.
    app = None


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - depends on optional hosted extra.
        raise RuntimeError(
            "Uvicorn is not installed. Use `pip install -e .[hosted]` before running `chronicle-api`."
        ) from exc
    if app is None:
        raise RuntimeError(
            "FastAPI dependencies are not installed. Use `pip install -e .[hosted]` before running `chronicle-api`."
        )
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("chronicle.service.app:app", host="0.0.0.0", port=port, reload=False)
