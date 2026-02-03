# Deployment Workflow

This document defines the path from development to production.

---

## Deployment Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    DEVELOPMENT                               │
│  • Agent writes code                                         │
│  • Unit tests pass                                           │
│  • Code review complete                                      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    INTEGRATION                               │
│  • Merge to main branch                                      │
│  • Integration tests run                                     │
│  • E2E tests run                                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGING                                   │
│  • Deploy to staging environment                             │
│  • Smoke tests                                               │
│  • QA verification                                           │
│  • Design review (if UI changes)                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                [HUMAN APPROVAL REQUIRED]                     │
│  • Review deployment checklist                               │
│  • Verify rollback plan                                      │
│  • Confirm go/no-go                                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION                                │
│  • Deploy to production                                      │
│  • Health checks                                             │
│  • Monitor for issues                                        │
│  • Rollback if needed                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Agent Restrictions

**CRITICAL: Agents CANNOT deploy to production directly.**

| Agent | Development | Staging | Production |
|-------|:-----------:|:-------:|:----------:|
| backend-specialist | ✅ Write code | ❌ Cannot deploy | ❌ Forbidden |
| frontend-specialist | ✅ Write code | ❌ Cannot deploy | ❌ Forbidden |
| database-specialist | ✅ Create migrations | ❌ Cannot run | ❌ Forbidden |
| qa-specialist | ✅ Write tests | ✅ Run tests | ❌ Forbidden |
| docs-specialist | ✅ Write docs | N/A | N/A |
| orchestrator | ✅ Coordinate | ✅ Request deploy | ❌ Cannot execute |

### Who Can Deploy?

**Only humans can authorize and execute production deployments.**

Agents can:
- Prepare deployment packages
- Create deployment checklists
- Request deployment approval
- Monitor post-deployment

Agents cannot:
- Execute deployment commands
- Access production systems
- Modify production databases
- Roll back production changes

---

## Deployment Checklist

### Pre-Deployment (Agent Prepared)

```markdown
## Deployment Checklist

**Version**: [version number]
**Date**: YYYY-MM-DD
**Prepared By**: [orchestrator]

### Changes Included
- [ ] [Change 1]: [description]
- [ ] [Change 2]: [description]

### Tests Passed
- [ ] Unit tests: [X/Y passed]
- [ ] Integration tests: [X/Y passed]
- [ ] E2E tests: [X/Y passed]

### Database Changes
- [ ] No migrations
- [ ] Migration: [file name]
  - [ ] Tested rollback
  - [ ] Backup verified

### Dependencies
- [ ] No new dependencies
- [ ] New dependency: [name]
  - [ ] Security scanned
  - [ ] License verified

### Breaking Changes
- [ ] None
- [ ] Breaking: [description]
  - [ ] Migration guide prepared
  - [ ] Clients notified

### Rollback Plan
- [ ] Rollback command: [command]
- [ ] Database rollback: [migration name]
- [ ] Estimated rollback time: [minutes]

### Monitoring
- [ ] Alerts configured
- [ ] Dashboards updated
- [ ] On-call notified
```

### Deployment Request (Agent Submits)

```markdown
## Production Deployment Request

**Status**: AWAITING HUMAN APPROVAL
**Urgency**: [Routine | Urgent | Emergency]
**Requested By**: [orchestrator]
**Checklist**: [link to checklist]

### Summary
[Brief description of what's being deployed]

### Risk Assessment
- **Risk Level**: [Low | Medium | High]
- **Impact if failure**: [description]
- **Mitigation**: [steps]

### Deployment Window
- **Preferred**: [date/time]
- **Alternative**: [date/time]
- **Blackout periods**: [list any]

### Required Actions
1. Review checklist
2. Approve deployment
3. Execute deployment
4. Verify health checks
5. Monitor for [X] minutes

---
**ACTION REQUIRED**: Human must approve and execute deployment
```

---

## Environment Definitions

### Development
- **Purpose**: Active development, experimentation
- **Data**: Test/mock data only
- **Access**: All agents (within role)
- **Deployment**: Automatic on commit

### Staging
- **Purpose**: Pre-production validation
- **Data**: Anonymized production-like data
- **Access**: Read for all agents, deploy for QA
- **Deployment**: Automatic on merge to main

### Production
- **Purpose**: Live user-facing system
- **Data**: Real user data
- **Access**: NO agent access
- **Deployment**: Human-only, manual approval

---

## Rollback Procedures

### Automatic Rollback Triggers
- Health check failures (>3 consecutive)
- Error rate spike (>5% increase)
- Latency degradation (>2x baseline)
- Memory/CPU exhaustion

### Manual Rollback Process

```
Issue detected in production
        │
        ▼
[HUMAN DECISION: ROLLBACK?]
        │
        ├── YES: Execute rollback
        │         │
        │         ├── Revert application code
        │         ├── Rollback migrations (if safe)
        │         └── Verify system health
        │
        └── NO: Hotfix path
                  │
                  ├── Diagnose issue
                  ├── Develop fix
                  └── Emergency deployment
```

### Post-Rollback Actions
1. Document what happened
2. Analyze root cause
3. Create fix task
4. Update deployment checklist
5. Review process for gaps

---

## Emergency Procedures

### Severity Levels

| Level | Definition | Response Time |
|-------|------------|---------------|
| **P1** | Complete outage | Immediate |
| **P2** | Major feature broken | < 1 hour |
| **P3** | Minor feature affected | < 4 hours |
| **P4** | Cosmetic/non-urgent | Next business day |

### Emergency Deployment

For P1/P2 issues:
1. Human declares emergency
2. Skip staging (with explicit approval)
3. Deploy directly to production
4. Monitor closely
5. Full review after resolution

**Even emergencies require human execution.**

---

## Deployment History

Maintain deployment history:

```markdown
| Date | Version | Type | Status | Rollback | Notes |
|------|---------|------|--------|----------|-------|
| YYYY-MM-DD | vX.Y.Z | Feature | Success | N/A | [brief] |
| YYYY-MM-DD | vX.Y.W | Hotfix | Success | N/A | [brief] |
| YYYY-MM-DD | vX.Y.V | Feature | Rolled back | Yes | [reason] |
```

---

## Modification History

| Date | Time | Agent | Action | Details | Reason |
|------|------|-------|--------|---------|--------|
| 2026-01-30 | 10:15 | orchestrator | created | Initial deployment workflow document | Define production path per user request |
