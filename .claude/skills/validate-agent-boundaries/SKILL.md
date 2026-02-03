---
description: Validate that agent outputs adhere to role boundaries and guardrails
user-invocable: true
---

# Validate Agent Boundaries Skill

Verify that each subagent's output strictly adheres to its assigned role, scope, and guardrails.

## Purpose

This skill detects and flags:
- Cross-role behavior (agent acting outside its domain)
- Unauthorized actions (forbidden operations for the role)
- Violations of execution or permission constraints
- Guardrail bypasses or self-modification attempts

**Any detected violation must block further execution and be escalated to the orchestrator.**

## Usage

```
/validate-agent-boundaries <agent_name> <output_description>
```

## Examples

```
/validate-agent-boundaries backend-specialist "Modified database schema in migration file"
/validate-agent-boundaries frontend-specialist "Changed API endpoint response format"
/validate-agent-boundaries qa-specialist "Fixed bug in application code instead of writing test"
```

## Validation Rules

### Role Boundary Matrix

| Agent | Allowed Domains | Forbidden Domains |
|-------|-----------------|-------------------|
| orchestrator | Task coordination, planning | Code, execution, approvals |
| backend-specialist | API, services, server logic | Database schema, frontend |
| frontend-specialist | UI, components, client state | Backend API, database |
| database-specialist | Schema, migrations, queries | Application code |
| qa-specialist | Tests only | Application code, schemas |
| docs-specialist | Documentation only | Any code changes |
| design-reviewer | Read-only review | Any modifications |

### Violation Detection Criteria

1. **Cross-Role Violation**
   - Agent modified files outside its domain
   - Agent made decisions reserved for another role
   - Agent bypassed orchestrator for cross-domain work

2. **Unauthorized Action**
   - Destructive commands proposed or executed
   - Production-affecting changes without approval
   - Guardrail circumvention attempts

3. **Permission Violation**
   - Used tools not in agent's allowed list
   - Accessed restricted resources
   - Modified configuration or permissions

## Instructions

When this skill is invoked:

1. **Identify the agent role** from the agent name
2. **Load the agent's constraints** from `.claude/agents/<agent>.md`
3. **Analyze the output** against:
   - Allowed actions list
   - Forbidden actions list
   - Role boundaries
   - Tool permissions

4. **Classification**:
   - `VALID` - Output is within role boundaries
   - `BOUNDARY_VIOLATION` - Output crosses into another role's domain
   - `UNAUTHORIZED_ACTION` - Output includes forbidden operations
   - `PERMISSION_VIOLATION` - Output uses disallowed capabilities

5. **Response Format**:
```
## Boundary Validation Result

**Agent**: <agent_name>
**Classification**: <VALID|BOUNDARY_VIOLATION|UNAUTHORIZED_ACTION|PERMISSION_VIOLATION>

### Analysis
<detailed explanation>

### Violations Found
- [VIOLATION TYPE] Description of violation

### Recommendation
<PROCEED|BLOCK|ESCALATE>

### Required Actions
- <action items if violations found>
```

6. **If violations detected**:
   - Output `[BOUNDARY VIOLATION DETECTED]`
   - Block further execution
   - Escalate to orchestrator with full details
   - Do NOT allow the output to proceed

## Integration

This skill should be invoked:
- By the orchestrator after receiving any specialist output
- Before any task is marked as complete
- As part of the standard review workflow

## Fail-Safe Behavior

If unable to determine validity:
- Default to `REVIEW_REQUIRED`
- Do not auto-approve
- Escalate for human review
