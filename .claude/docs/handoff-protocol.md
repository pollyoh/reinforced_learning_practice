# Handoff Protocol

This document defines how work is handed off between specialist agents.

---

## Handoff Types

| Type | From | To | Example |
|------|------|-----|---------|
| **Sequential** | Agent A | Agent B | DB migration → Backend implementation |
| **Parallel-Join** | Multiple agents | Single agent | Backend + Frontend → QA |
| **Review** | Implementer | Reviewer | Frontend → Design-reviewer |
| **Documentation** | Any agent | Docs-specialist | Implementation → Documentation |

---

## Standard Handoff Process

### Step 1: Completion Declaration

Completing agent marks task as done with summary:

```markdown
## Task Completion Summary

**Task**: [Task ID and title]
**Status**: COMPLETED
**Agent**: [agent-name]

### Deliverables
- [File/artifact 1]: [description]
- [File/artifact 2]: [description]

### Key Decisions Made
- [Decision 1]: [reasoning]
- [Decision 2]: [reasoning]

### Assumptions
- [Assumption 1]: [basis]
- [Assumption 2]: [basis]

### Open Questions
- [Question 1]: [context]

### Ready for: [next-agent-name]
```

### Step 2: Orchestrator Validation

Orchestrator checks:
- [ ] Deliverables exist and are complete
- [ ] No boundary violations
- [ ] Risk classification acceptable
- [ ] Quality meets standards

### Step 3: Context Transfer

Orchestrator creates new task with full context:

```markdown
## Task Assignment

**Task**: [New task title]
**Assigned To**: [next-agent-name]
**Depends On**: [previous task ID]

### Context from Previous Work
[Summary of what was done and decisions made]

### Your Objective
[What this agent needs to accomplish]

### Key Files to Review
- [File 1]: [why relevant]
- [File 2]: [why relevant]

### Constraints
- [Constraint 1]
- [Constraint 2]

### Success Criteria
- [ ] [Criterion 1]
- [ ] [Criterion 2]
```

### Step 4: Receiving Agent Confirmation

Receiving agent confirms:
- [ ] Context understood
- [ ] Dependencies available
- [ ] Ready to proceed

If issues: Report `[BLOCKED - HANDOFF ISSUE]`

---

## Common Handoff Patterns

### Pattern 1: Database → Backend

**Scenario**: Database-specialist creates schema; Backend-specialist implements API

```
database-specialist
        │
        ├── Creates migration file
        ├── Documents schema structure
        └── Notes any constraints
        │
        ▼
[HANDOFF: Schema Ready]
        │
        ▼
backend-specialist
        │
        ├── Reads migration file
        ├── Implements models/entities
        ├── Creates API endpoints
        └── Writes unit tests
```

**Handoff Package**:
- Migration file location
- Schema diagram (if complex)
- Relationship definitions
- Index strategy
- Known constraints

### Pattern 2: Backend → Frontend

**Scenario**: Backend creates API; Frontend consumes it

```
backend-specialist
        │
        ├── Implements API endpoint
        ├── Documents API contract
        └── Provides example responses
        │
        ▼
[HANDOFF: API Ready]
        │
        ▼
frontend-specialist
        │
        ├── Reviews API contract
        ├── Implements data fetching
        ├── Creates UI components
        └── Handles error states
```

**Handoff Package**:
- API endpoint documentation
- Request/response examples
- Authentication requirements
- Error code definitions
- Rate limiting info

### Pattern 3: Implementation → QA

**Scenario**: Feature implemented; QA validates

```
backend-specialist + frontend-specialist
        │
        ├── Complete feature implementation
        ├── Write unit tests (coverage report)
        └── Document test scenarios
        │
        ▼
[HANDOFF: Ready for QA]
        │
        ▼
qa-specialist
        │
        ├── Reviews implementation
        ├── Writes integration tests
        ├── Writes E2E tests
        └── Reports test results
```

**Handoff Package**:
- Feature description
- Unit test coverage report
- Suggested test scenarios
- Known edge cases
- Environment setup instructions

### Pattern 4: Frontend → Design Review

**Scenario**: UI implemented; Design-reviewer validates

```
frontend-specialist
        │
        ├── Implements UI component
        ├── Applies design tokens
        └── Adds accessibility attributes
        │
        ▼
[HANDOFF: Ready for Design Review]
        │
        ▼
design-reviewer
        │
        ├── Checks design token usage
        ├── Verifies accessibility
        ├── Reports compliance issues
        └── No code modifications
```

**Handoff Package**:
- Component locations
- Design spec reference
- Accessibility checklist
- Screenshots (if available)

### Pattern 5: Any → Documentation

**Scenario**: Work completed; Documentation updates needed

```
Any specialist
        │
        ├── Completes implementation
        ├── Notes documentation needs
        └── Provides technical details
        │
        ▼
[HANDOFF: Documentation Needed]
        │
        ▼
docs-specialist
        │
        ├── Reviews implementation
        ├── Updates API docs
        ├── Updates user guides
        └── Adds modification history
```

**Handoff Package**:
- What was implemented
- Technical details to document
- Target audience
- Existing docs to update
- New docs to create

---

## Handoff Failures

### Common Failure Modes

| Failure | Symptom | Resolution |
|---------|---------|------------|
| **Missing context** | Receiving agent asks basic questions | Re-do handoff with more detail |
| **Incomplete deliverable** | Required artifact missing | Sending agent completes work |
| **Assumption mismatch** | Output doesn't match expectation | Clarify requirements |
| **Dependency not ready** | Blocked on missing prerequisite | Wait or parallelize differently |

### Handling Handoff Failures

```
Receiving agent reports issue
        │
        ▼
Orchestrator diagnoses:
├── Missing context? → Gather and resend
├── Incomplete work? → Return to original agent
├── Unclear requirements? → Clarify with human
└── Technical blocker? → Problem-solve or escalate
        │
        ▼
Issue resolved, handoff retried
```

---

## Handoff Checklist

Before declaring handoff ready:

### Sending Agent
- [ ] All deliverables created
- [ ] Key decisions documented
- [ ] Assumptions listed
- [ ] Open questions noted
- [ ] No pending blockers

### Orchestrator
- [ ] Validation passed
- [ ] Risk classification acceptable
- [ ] Context package complete
- [ ] Receiving agent notified

### Receiving Agent
- [ ] Context reviewed
- [ ] Dependencies available
- [ ] Questions resolved
- [ ] Ready to proceed

---

## Modification History

| Date | Time | Agent | Action | Details | Reason |
|------|------|-------|--------|---------|--------|
| 2026-01-30 | 10:10 | orchestrator | created | Initial handoff protocol document | Define cross-agent handoffs per user request |
