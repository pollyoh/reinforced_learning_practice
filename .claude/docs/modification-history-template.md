# Modification History Standard

This document defines the mandatory format for tracking all modifications across project documentation.

---

## Format Specification

Every document must include a `## Modification History` section at the end, using this format:

```markdown
## Modification History

| Date | Time | Agent | Action | Details | Reason |
|------|------|-------|--------|---------|--------|
| YYYY-MM-DD | HH:MM | agent-name | verb | brief description | why this change was needed |
```

### Field Definitions

| Field | Format | Description |
|-------|--------|-------------|
| **Date** | `YYYY-MM-DD` | ISO 8601 date format |
| **Time** | `HH:MM` | 24-hour format (UTC preferred) |
| **Agent** | `agent-name` | Which agent made the change (e.g., `docs-specialist`, `backend-specialist`, `orchestrator`) |
| **Action** | verb | One of: `created`, `updated`, `refactored`, `fixed`, `deleted`, `moved` |
| **Details** | 1-2 lines | What was changed (be specific) |
| **Reason** | 1-2 lines | Why the change was necessary |

---

## Examples

### Example 1: Simple Update
```markdown
| 2026-01-30 | 09:15 | docs-specialist | updated | Added API endpoint documentation for /auth/login | New endpoint implemented by backend-specialist |
```

### Example 2: Multiple Changes
```markdown
| 2026-01-30 | 10:00 | backend-specialist | created | Initial API implementation for user service | Task #42: Implement user management |
| 2026-01-30 | 14:30 | qa-specialist | updated | Added test coverage for edge cases | Code review feedback |
| 2026-01-31 | 09:00 | docs-specialist | updated | Documented error codes and responses | Sync with implementation |
```

---

## Rules for Agents

### All Agents MUST:

1. **Add entry on every modification** - No silent changes
2. **Use accurate timestamps** - Reflect actual modification time
3. **Identify themselves** - Use exact agent name from `.claude/agents/`
4. **Be specific** - "Updated docs" is insufficient; "Added rate limiting section to API docs" is correct
5. **Explain why** - Context is mandatory, not optional

### All Agents MUST NOT:

1. Delete or modify previous history entries
2. Omit the reason field
3. Use vague descriptions ("various fixes", "minor updates")
4. Back-date entries

---

## Automated Summary Generation

Docs-specialist should generate periodic summaries using this format:

```markdown
## Change Summary: [Document Name]

**Period**: YYYY-MM-DD to YYYY-MM-DD
**Total Changes**: N

### By Agent
- backend-specialist: X changes
- frontend-specialist: Y changes
- docs-specialist: Z changes

### By Type
- created: A
- updated: B
- fixed: C

### Notable Changes
1. [Brief description of significant change #1]
2. [Brief description of significant change #2]
```

---

## Implementation Notes

- History section should always be the LAST section of any document
- Maximum 100 entries per document; archive older entries to `_archive/[doc-name]-history-YYYY.md`
- For code files, history is tracked via git; this standard applies to documentation and configuration files
