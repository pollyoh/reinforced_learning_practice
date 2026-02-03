# Role Boundary Matrix

Quick reference for what each agent can and cannot do.

---

## Permission Matrix

| Action | Orchestrator | Backend | Frontend | Database | QA | Docs | Design |
|--------|:------------:|:-------:|:--------:|:--------:|:--:|:----:|:------:|
| **Read any file** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Write backend code** | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Write frontend code** | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| **Write migrations** | ❌ | ❌ | ❌ | ✅ | ❌ | ❌ | ❌ |
| **Write test files** | ❌ | ✅* | ✅* | ❌ | ✅ | ❌ | ❌ |
| **Write documentation** | ❌ | ❌ | ❌ | ❌ | ❌ | ✅ | ❌ |
| **Execute Bash** | ❌ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ |
| **Create tasks** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Assign work** | ✅ | ❌ | ❌ | ❌ | ❌ | ❌ | ❌ |
| **Review code** | ❌ | ✅** | ✅** | ✅** | ✅ | ❌ | ✅*** |

**Legend**:
- ✅ = Allowed
- ❌ = Forbidden
- ✅* = Unit tests only (within domain)
- ✅** = Within domain only
- ✅*** = Read-only review, no approval authority

---

## Domain Ownership

| Domain | Primary Owner | Can Contribute | Cannot Touch |
|--------|--------------|----------------|--------------|
| API endpoints | backend-specialist | qa (tests) | frontend, database, docs |
| UI components | frontend-specialist | qa (tests), design (review) | backend, database |
| Database schema | database-specialist | None | All others |
| Migrations | database-specialist | None | All others |
| E2E tests | qa-specialist | None | All others |
| Integration tests | qa-specialist | backend, frontend (unit) | database, docs |
| Documentation | docs-specialist | All (provide info) | None |
| Design compliance | design-reviewer | frontend (implements) | None |
| Task coordination | orchestrator | None | All others |

---

## Forbidden Actions (All Agents)

| Action | Why Forbidden |
|--------|---------------|
| `rm -rf` / recursive delete | Destructive, irreversible |
| `git push --force` | Destroys history |
| `DROP TABLE` / `TRUNCATE` | Data loss |
| Production DB access | Safety risk |
| Deploy to production | Requires human approval |
| Modify credentials | Security risk |
| Modify IAM/permissions | Security risk |
| Self-modify guardrails | Safety violation |

---

## Cross-Role Handoff Rules

| From | To | Handoff Mechanism |
|------|-----|-------------------|
| database-specialist | backend-specialist | Migration file created → Backend implements using schema |
| backend-specialist | frontend-specialist | API contract defined → Frontend consumes API |
| frontend-specialist | qa-specialist | Component implemented → QA writes E2E tests |
| Any specialist | docs-specialist | Implementation done → Docs updates documentation |
| frontend-specialist | design-reviewer | UI implemented → Design reviews compliance |

---

## Escalation Paths

| Issue | Escalate To | Action |
|-------|-------------|--------|
| Unclear requirements | orchestrator → human | Block until clarified |
| Cross-domain change needed | orchestrator | Delegate to correct specialist |
| Conflicting specialist outputs | orchestrator → human | Present both, human decides |
| Safety rule violation | orchestrator → human | Block, report violation |
| Scope creep detected | orchestrator → human | Pause, revise plan |

---

## Quick Decision Guide

### "Can I do X?"

```
1. Is X in my domain?
   NO  → Delegate to correct specialist
   YES → Continue

2. Is X destructive or irreversible?
   YES → Propose only, require human approval
   NO  → Continue

3. Does X require cross-domain coordination?
   YES → Request orchestrator mediation
   NO  → Continue

4. Am I confident X is safe?
   NO  → Stop, ask for clarification
   YES → Proceed with two-step execution
```

---

## Modification History

| Date | Time | Agent | Action | Details | Reason |
|------|------|-------|--------|---------|--------|
| 2026-01-30 | 09:35 | orchestrator | created | Initial role boundary matrix | Consolidate scattered permission info |
