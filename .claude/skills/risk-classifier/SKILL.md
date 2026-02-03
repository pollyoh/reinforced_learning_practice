---
description: Classify agent outputs by operational risk level
user-invocable: true
---

# Risk Classifier Skill

Classify all agent outputs according to operational risk level.

## Purpose

This skill evaluates proposed changes and outputs to determine their risk level, ensuring dangerous operations are blocked before execution.

## Risk Levels

| Level | Description | Action |
|-------|-------------|--------|
| `SAFE` | Low risk, reversible, non-production | May proceed |
| `REVIEW_REQUIRED` | Moderate risk, needs human review | Pause for approval |
| `DANGEROUS` | High risk, destructive, or irreversible | **BLOCK EXECUTION** |

**Outputs classified as DANGEROUS must NEVER proceed to execution.**

## Usage

```
/risk-classifier <action_description>
```

## Examples

```
/risk-classifier "Add new utility function to helpers.ts"
/risk-classifier "Modify user authentication flow"
/risk-classifier "Drop unused database table"
/risk-classifier "Deploy to production environment"
```

## Classification Criteria

### SAFE Criteria
- Read-only operations
- New file creation (non-config)
- Test file modifications
- Documentation updates
- Local development changes
- Easily reversible edits
- No external system impact

### REVIEW_REQUIRED Criteria
- Modifying existing business logic
- API contract changes
- Database schema modifications (non-destructive)
- Configuration changes
- Authentication/authorization changes
- External service integrations
- Changes affecting multiple components
- Performance-sensitive modifications

### DANGEROUS Criteria (Auto-Block)
- `DROP`, `TRUNCATE`, `DELETE` without WHERE
- `rm -rf`, `wipefs`, `mkfs`, `dd` commands
- Force push to protected branches
- Production database access
- Credential/secret modifications
- Production deployments
- IAM/permission changes
- Irreversible data transformations
- Service account modifications
- API key rotation/revocation

## Risk Assessment Matrix

| Factor | Low Risk | Medium Risk | High Risk |
|--------|----------|-------------|-----------|
| Reversibility | Easy rollback | Partial rollback | Irreversible |
| Scope | Single file | Multiple files | System-wide |
| Data Impact | No data change | Data modification | Data deletion |
| Environment | Development | Staging | Production |
| External Impact | None | Internal services | External users |

## Instructions

When this skill is invoked:

1. **Parse the action description**
2. **Evaluate against each risk factor**:
   - Destructiveness (can it destroy data?)
   - Irreversibility (can it be undone?)
   - Production impact (does it affect live systems?)
   - Scope (how much is affected?)
   - Data sensitivity (PII, credentials, etc.)

3. **Apply classification rules**:
   - If ANY dangerous criteria match → `DANGEROUS`
   - If ANY review criteria match → `REVIEW_REQUIRED`
   - Otherwise → `SAFE`

4. **Output format**:
```
## Risk Classification Result

**Action**: <action_description>
**Classification**: <SAFE|REVIEW_REQUIRED|DANGEROUS>
**Confidence**: <HIGH|MEDIUM|LOW>

### Risk Factors Identified
- [FACTOR] Description and severity

### Danger Signals
- <list any dangerous patterns detected>

### Recommendation
- <PROCEED|REVIEW|BLOCK>

### Mitigation (if applicable)
- <suggested safeguards or alternatives>
```

5. **For DANGEROUS classification**:
   - Output `[DANGEROUS - EXECUTION BLOCKED]`
   - Provide explicit reason
   - Suggest safer alternatives if possible
   - Require human override to proceed

## Automatic Danger Patterns

The following patterns trigger automatic `DANGEROUS` classification:

```
# Database
/DROP\s+(TABLE|DATABASE|INDEX)/i
/TRUNCATE\s+TABLE/i
/DELETE\s+FROM\s+\w+\s*$/i  # DELETE without WHERE

# Shell
/rm\s+-rf?\s+\//
/rm\s+-rf?\s+\*/
/wipefs|mkfs|dd\s+if=/

# Git
/push\s+.*--force/
/push\s+-f/
/reset\s+--hard/

# Production
/production|prod\s+(deploy|database|db)/i
/--env[=\s]+prod/

# Credentials
/(password|secret|key|token)\s*[=:]/i
/rotate.*key|revoke.*token/i
```

## Integration

This skill should be invoked:
- Before any task execution begins
- After risk-relevant outputs are generated
- As part of the orchestrator's review workflow

## Fail-Safe Behavior

- Unknown patterns → Default to `REVIEW_REQUIRED`
- Classification uncertainty → Escalate to human
- Never auto-approve when in doubt
