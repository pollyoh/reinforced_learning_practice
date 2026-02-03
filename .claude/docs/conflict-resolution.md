# Conflict Resolution

This document defines how to handle conflicts between specialist agents.

---

## Types of Conflicts

| Conflict Type | Example | Resolution Path |
|---------------|---------|-----------------|
| **Technical Approach** | Different algorithms for same problem | Technical evaluation |
| **API Contract** | Backend/Frontend disagree on format | Contract negotiation |
| **Architecture** | Different structural approaches | Architecture review |
| **Priority** | Multiple agents need same resource | Orchestrator decides |
| **Scope** | Overlapping responsibilities | Boundary clarification |

---

## Conflict Detection

### Automatic Detection

Orchestrator detects conflicts when:
- Two agents modify same file
- Two agents propose incompatible changes
- Agent output contradicts another's assumptions
- API contracts don't match between producer/consumer

### Manual Reporting

Agents report conflicts using:
```markdown
[CONFLICT DETECTED]
- Type: [Technical | Contract | Architecture | Priority | Scope]
- With: [other-agent-name]
- Description: [what the conflict is]
- My position: [what I propose]
```

---

## Resolution Process

### Step 1: Conflict Identification

```
Conflict detected
      │
      ▼
Orchestrator gathers:
├── Agent A's proposal + reasoning
├── Agent B's proposal + reasoning
├── Shared context/requirements
└── Impact of each approach
```

### Step 2: Classification

| Classification | Criteria | Resolution |
|----------------|----------|------------|
| **Trivial** | Style/preference only | Orchestrator picks one |
| **Technical** | Measurable differences | Data-driven decision |
| **Architectural** | Design philosophy | Human decision |
| **Fundamental** | Incompatible requirements | Escalate to human |

### Step 3: Resolution by Type

#### Trivial Conflicts
- Orchestrator decides based on project conventions
- No human involvement needed
- Document decision for consistency

#### Technical Conflicts
```
Compare proposals on:
├── Performance (benchmarks if available)
├── Maintainability (complexity metrics)
├── Consistency (with existing code)
└── Extensibility (future requirements)
      │
      ▼
Choose approach with best overall score
      │
      ▼
Document reasoning for future reference
```

#### Architectural Conflicts
```
Present both options to human:
├── Option A: [Agent A's approach]
│   ├── Pros: [list]
│   └── Cons: [list]
├── Option B: [Agent B's approach]
│   ├── Pros: [list]
│   └── Cons: [list]
└── Orchestrator recommendation: [if any]
      │
      ▼
Human selects approach
      │
      ▼
Update architecture docs with decision
```

#### Fundamental Conflicts
```
Requirements are incompatible
      │
      ▼
[BLOCKED - REQUIREMENTS CLARIFICATION NEEDED]
      │
      ▼
Present to human:
├── Requirement A implies: [approach 1]
├── Requirement B implies: [approach 2]
└── These cannot both be satisfied because: [reason]
      │
      ▼
Human clarifies/prioritizes requirements
      │
      ▼
Restart planning with clarified requirements
```

---

## Common Conflict Scenarios

### Scenario: API Contract Mismatch

**Backend proposes**:
```json
{ "users": [{ "id": 1, "name": "Alice" }] }
```

**Frontend expects**:
```json
[{ "id": 1, "name": "Alice" }]
```

**Resolution**:
1. Check existing API conventions in codebase
2. If no convention, prefer simpler format (array)
3. If backend has strong reason (pagination, metadata), discuss
4. Document decision in API standards

### Scenario: Database Schema Dispute

**Approach A**: Normalized (3NF)
**Approach B**: Denormalized for performance

**Resolution**:
1. Identify query patterns
2. Estimate data volume
3. Benchmark if possible
4. Default to normalized unless performance is critical
5. Human decides if trade-offs unclear

### Scenario: Test Coverage Overlap

**Backend-specialist**: "I'll write unit tests for this service"
**QA-specialist**: "I'll write integration tests for this service"

**Resolution**:
1. This is NOT a conflict - different test types
2. Clarify ownership:
   - Backend: Unit tests (isolated, mocked)
   - QA: Integration tests (end-to-end)
3. Both proceed in their domains

---

## Test Ownership Clarification

| Test Type | Primary Owner | Secondary | Notes |
|-----------|---------------|-----------|-------|
| Unit tests (backend) | backend-specialist | - | Isolated, mocked dependencies |
| Unit tests (frontend) | frontend-specialist | - | Component tests, isolated |
| Integration tests | qa-specialist | backend, frontend | Cross-component |
| E2E tests | qa-specialist | - | Full user journeys |
| Performance tests | qa-specialist | database | Load, stress testing |
| Accessibility tests | qa-specialist | design-reviewer | WCAG compliance |

---

## Conflict Resolution Report

After resolving a conflict, document:

```markdown
## Conflict Resolution Report

**ID**: CONFLICT-YYYY-MM-DD-NNN
**Type**: [Technical | Contract | Architecture | Priority | Scope]
**Agents Involved**: [list]
**Date Resolved**: YYYY-MM-DD

### Conflict Description
[What the conflict was]

### Options Considered
1. [Option A description]
2. [Option B description]

### Resolution
[What was decided]

### Reasoning
[Why this resolution was chosen]

### Impact
[What changes resulted from this decision]

### Lessons Learned
[What to do differently in future]
```

---

## Preventing Conflicts

### Proactive Measures

1. **Clear domain boundaries** - See `role-boundary-matrix.md`
2. **Early contract definition** - API specs before implementation
3. **Architecture decisions** - Document early, reference often
4. **Regular sync** - Orchestrator reviews cross-domain work

### Warning Signs

Watch for these conflict precursors:
- "I assume the API will..." (assumption without verification)
- "This should work if..." (dependency on undefined behavior)
- "I'll just change this..." (modifying shared interface)

---

## Modification History

| Date | Time | Agent | Action | Details | Reason |
|------|------|-------|--------|---------|--------|
| 2026-01-30 | 10:05 | orchestrator | created | Initial conflict resolution document | Define conflict handling per user request |
