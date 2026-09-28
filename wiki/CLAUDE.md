# Wiki Schema — agent-telemetry-feedback-control

This wiki is a compiled knowledge layer for the project. It follows the LLM-wiki pattern:
raw sources → LLM-compiled pages → queryable index.

Open `wiki/` as an Obsidian vault to browse and edit pages visually.

## Structure

- `sources/` — raw inputs: traces, digests, transcripts, links. Never edit wiki pages directly from source.
- `wiki/index.md` — master catalog. One line per page: `[Title](path) — one-sentence summary`. Organized by category.
- `wiki/log.md` — append-only. Format: `## [YYYY-MM-DD] {operation} | {title}`
- `wiki/entities/` — one page per named entity (a component, a table, a hook event, a model tier)
- `wiki/topics/` — synthesized topic pages (KPI definitions, tuning history, failure patterns)

## Operations

### ingest [source]
1. Read the source file
2. Write or update a wiki page (entities/ or topics/ as appropriate)
3. Update index.md with a one-line entry
4. Append to log.md: `## [date] ingest | {title}`
5. Update any existing pages that reference the same entity/concept

### query [question]
1. Read index.md
2. Pull the relevant wiki pages
3. Synthesize an answer
4. If the answer is non-obvious, file it as a new wiki page and update index.md

### lint
Check for:
- Contradictions between pages
- Claims older than 90 days with no supporting source
- Orphan pages (in wiki/ but missing from index.md)
- Missing cross-references between related entities

## Page conventions

- Filename: kebab-case, e.g., `entities/turn-record.md`
- H1 = page title
- Frontmatter (YAML):
  ```yaml
  ---
  type: entity | topic
  updated: YYYY-MM-DD
  sources: [filename1, filename2]
  ---
  ```
- Cross-references: `[[entity-name]]` style
- Provisional claims (unverified): prefix with `?`
