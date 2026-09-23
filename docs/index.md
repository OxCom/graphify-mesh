---
title: Documentation
---

<section class="docs-hero">
  <p class="docs-eyebrow">Graphify across repositories</p>
  <h1>One graph. Every codebase.</h1>
  <p class="docs-lede">
    Build and query a repo-attributed knowledge graph without losing the
    boundaries between projects.
  </p>
  <div class="docs-actions">
    <a class="docs-action docs-action-primary" href="setup.html">Start with setup</a>
    <a class="docs-action" href="architecture.html">Explore the architecture</a>
  </div>
</section>

## Documentation

<nav class="docs-grid" aria-label="Documentation sections">
  <a class="docs-card" href="setup.html">
    <span class="docs-card-label">Start here</span>
    <strong>Setup</strong>
    <span>Install graphify-mesh, register repositories, and run the first sync.</span>
  </a>
  <a class="docs-card" href="configuration.html">
    <span class="docs-card-label">Reference</span>
    <strong>Configuration</strong>
    <span>Look up environment variables, CLI flags, registry fields, and schemas.</span>
  </a>
  <a class="docs-card" href="architecture.html">
    <span class="docs-card-label">Concepts</span>
    <strong>Architecture</strong>
    <span>Understand the sync pipeline, graph boundaries, and query services.</span>
  </a>
  <a class="docs-card" href="mcp-server.html">
    <span class="docs-card-label">Tools</span>
    <strong>MCP server</strong>
    <span>Connect an agent and learn the available retrieval tools and transports.</span>
  </a>
  <a class="docs-card" href="keeping-sync-up-to-date.html">
    <span class="docs-card-label">Operations</span>
    <strong>Keep the mesh current</strong>
    <span>Schedule syncs, manage repository changes, and troubleshoot updates.</span>
  </a>
</nav>

## What graphify-mesh preserves

- Repository attribution on every result.
- A separate overlay for cross-project relationships.
- Current, immutable generations produced by rebuild-from-empty syncs.
- Current-repository search by default, with explicit cross-project widening.

[View the source on GitHub](https://github.com/OxCom/graphify-mesh)
