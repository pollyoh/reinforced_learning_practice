# Approval Process

This document defines when and how approvals are required in the multi-agent workflow.

---

## Approval Levels

| Level | Authority | When Required |
|-------|-----------|---------------|
| **Auto-Approve** | Agent self | Read-only operations, analysis, proposals |
| **Orchestrator Review** | Orchestrator | Task completion, output validation |
| **Human Required** | User | Production changes, risky operations, conflicts |

---

## Approval Matrix by Action Type

| Action | Risk Level | Approval Required |
|--------|------------|-------------------|
| Read any file | SAFE | Auto-approve |
| Analyze code | SAFE | Auto-approve |
| Propose changes | SAFE | Auto-approve |
| Create new file (dev) | LOW | Orchestrator review |
| Edit existing file | MEDIUM | Orchestrator review |
| Delete file | HIGH | Human required |
| Database migration | HIGH | Human required |
| API contract change | HIGH | Human required |
| Production deployment | CRITICAL | Human required + verification |

---

## Approval Workflow

### Standard Change (Low/Medium Risk)

```
Agent proposes change
        │
        ▼
Orchestrator validates:
├── Boundary check (validate-agent-boundaries)
├── Risk assessment (risk-classifier)
└── Output quality review
        │
        ▼
Classification?
├── SAFE → Orchestrator approves, proceeds
├── REVIEW_REQUIRED → Flag for human
└── DANGEROUS → Block, escalate
```

### High-Risk Change

```
Agent proposes change
        │
        ▼
Orchestrator detects high-risk:
├── Database schema change
├── API breaking change
├── Security-related change
└── Cross-system impact
        │
        ▼
[BLOCKED - HUMAN APPROVAL REQUIRED]
        │
        ▼
Present to human:
├── What: Exact change proposed
├── Why: Reason for change
├── Risk: Potential impacts
├── Rollback: Recovery plan
        │
        ▼
Human decision:
├── APPROVE → Proceed with change
├── MODIFY → Agent revises proposal
└── REJECT → Change cancelled
```

---

## Approval Authorities

### Orchestrator Can Approve:
- New test files
- Documentation updates
- Code style fixes
- Bug fixes in non-critical paths
- Internal refactoring (same behavior)

### Orchestrator Cannot Approve (Human Required):
- Production deployments
- Database schema changes
- API contract changes
- Security configuration changes
- Credential/secret modifications
- Cross-system integration changes
- Rollback operations
- Emergency fixes

---

## Approval Request Format

When requesting human approval, use this format:

```markdown
## Approval Request

**Type**: [Schema Change | API Change | Deployment | Security | Other]
**Risk Level**: [HIGH | CRITICAL]
**Requested By**: [agent-name]
**Task**: [Task ID and description]

### Proposed Change
[Detailed description of what will change]

### Justification
[Why this change is needed]

### Impact Assessment
- **Affected Systems**: [List]
- **Affected Users**: [Description]
- **Downtime Expected**: [Yes/No, duration]

### Risk Analysis
- **What could go wrong**: [List potential issues]
- **Mitigation**: [How risks are addressed]

### Rollback Plan
[Steps to undo if needed]

### Verification Steps
[How to confirm change was successful]

---
**ACTION REQUIRED**: Please respond with APPROVE, MODIFY, or REJECT
```

---

## Timeout and Escalation

| Approval Type | Timeout | Escalation |
|---------------|---------|------------|
| Orchestrator review | None (immediate) | N/A |
| Human review (normal) | 24 hours | Reminder sent |
| Human review (urgent) | 4 hours | Escalation to backup |
| Human review (critical) | 1 hour | Emergency contact |

### If No Response:
- Normal changes: Remain blocked, send reminder
- Critical changes: Escalate to emergency contact
- Never auto-approve without human response

---

## Post-Approval Actions

After approval is granted:

1. **Log the approval**
   - Who approved
   - When approved
   - What was approved
   - Any conditions

2. **Execute with monitoring**
   - Track execution progress
   - Capture any errors
   - Log completion

3. **Verify outcome**
   - Run verification steps
   - Confirm expected behavior
   - Report success/failure

4. **Document in history**
   - Add modification history entry
   - Include approval reference

---

## Revoking Approval

An approval can be revoked if:
- Requirements changed after approval
- New risks discovered
- Execution not started within timeout
- Approver requests revocation

**Process**:
1. Mark approval as REVOKED
2. Notify agent/orchestrator
3. Stop any pending execution
4. Document reason for revocation

---

## Modification History

| Date | Time | Agent | Action | Details | Reason |
|------|------|-------|--------|---------|--------|
| 2026-01-30 | 10:00 | orchestrator | created | Initial approval process document | Define approval workflows per user request |
