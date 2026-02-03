# Agent-Skill-Rule Interaction Framework

This document defines how agents, skills, and rules interact within the multi-agent system.

---

## 1. HIERARCHY OF AUTHORITY

```
┌─────────────────────────────────────────────────────────────┐
│                    HUMAN (Final Authority)                   │
├─────────────────────────────────────────────────────────────┤
│              GLOBAL RULES (CLAUDE.md, settings.json)         │
├─────────────────────────────────────────────────────────────┤
│                    SAFETY SKILLS                             │
│        (validate-agent-boundaries, risk-classifier)          │
├─────────────────────────────────────────────────────────────┤
│                    ORCHESTRATOR                              │
├─────────────────────────────────────────────────────────────┤
│              COORDINATION SKILLS                             │
│                  (coordinate-work)                           │
├─────────────────────────────────────────────────────────────┤
│                 SPECIALIST AGENTS                            │
│   (backend, frontend, database, qa, docs, design-reviewer)   │
└─────────────────────────────────────────────────────────────┘
```

### Authority Levels

| Level | Entity | Can Override |
|-------|--------|--------------|
| 1 (Highest) | Human | Everything |
| 2 | Global Rules | Agent decisions |
| 3 | Safety Skills | Agent outputs |
| 4 | Orchestrator | Specialist assignments |
| 5 | Coordination Skills | Task sequencing |
| 6 (Lowest) | Specialist Agents | Own domain only |

---

## 2. RULE PRECEDENCE

When rules conflict, apply in this order:

### 2.1 Absolute Rules (Never Override)
```
settings.local.json deny patterns > Agent MUST NOT rules > Role-specific forbidden
```

**Example**: If `settings.local.json` denies `Bash(rm -rf *)`, this blocks ALL agents regardless of their tool permissions.

### 2.2 Safety Rules
```
ABSOLUTE SAFETY RULES > UNIVERSAL SAFETY RULES > OPERATIONAL SAFETY PATTERNS
```

### 2.3 Workflow Rules
```
Human instruction > Orchestrator delegation > Skill invocation > Agent default behavior
```

---

## 3. AGENT-TO-AGENT INTERACTION RULES

### 3.1 Direct Communication: PROHIBITED

Agents CANNOT:
- Send messages directly to other agents
- Read another agent's state or context
- Invoke another agent directly

### 3.2 Indirect Communication: VIA TASK SYSTEM

All inter-agent communication flows through the task system:

```
Agent A                    Task System                    Agent B
   │                           │                             │
   ├── TaskCreate ────────────►│                             │
   │                           │◄──────── TaskGet ───────────┤
   │                           │                             │
   │                           │◄──────── TaskUpdate ────────┤
   ├── TaskList ──────────────►│                             │
   │                           │                             │
```

### 3.3 Handoff Protocol

When Agent A completes work that Agent B needs:

1. **Agent A**: Mark task as `completed` with summary in description
2. **Orchestrator**: Validates output (boundary check, risk classification)
3. **Orchestrator**: Creates new task for Agent B with context from Agent A
4. **Agent B**: Retrieves task, sees full context

**Example**: Database-specialist creates migration → Backend-specialist implements API

```
1. DB creates: Task "Create user migration" → completed
2. Orchestrator validates: No boundary violations
3. Orchestrator creates: Task "Implement user API" for backend
   Description includes: "Migration file: migrations/001_users.sql"
4. Backend retrieves task, reads migration file, implements API
```

### 3.4 Conflict Detection

When two agents produce conflicting outputs:

1. **Orchestrator detects** conflict during review
2. **Orchestrator creates** comparison task: "Resolve conflict: API format"
3. **Orchestrator presents** both options to human
4. **Human decides** which approach wins
5. **Losing agent's output** is archived (not deleted)

---

## 4. SKILL-TO-AGENT INTERACTION RULES

### 4.1 Skill Invocation

Skills can be invoked by:
- **Human**: Via `/skill-name` command
- **Orchestrator**: Via skill description in task

Skills CANNOT be invoked by:
- Specialist agents (they don't have Skill tool)

### 4.2 Skill Output Handling

| Skill | Output | Effect on Agents |
|-------|--------|------------------|
| `validate-agent-boundaries` | VALID | Agent proceeds |
| `validate-agent-boundaries` | VIOLATION | Agent output rejected, task reassigned |
| `risk-classifier` | SAFE | Agent proceeds |
| `risk-classifier` | REVIEW_REQUIRED | Agent pauses, human reviews |
| `risk-classifier` | DANGEROUS | Agent blocked, human must override |

### 4.3 Skill-Agent Dependency Chain

```
Agent Output → validate-agent-boundaries → risk-classifier → Orchestrator Review → Human Approval
```

**Blocking Points**:
- If `validate-agent-boundaries` fails → Output rejected immediately
- If `risk-classifier` returns DANGEROUS → Output blocked, no further processing
- If Orchestrator finds issues → Output flagged for human review

---

## 5. RULE-TO-AGENT INTERACTION

### 5.1 Rule Sources

| Source | Scope | Override Level |
|--------|-------|----------------|
| `settings.local.json` | All agents | Highest (system-enforced) |
| `CLAUDE.md` | All agents | High (inherited context) |
| `agents/*.md` | Specific agent | Medium (role-specific) |
| Task description | Current task | Low (task-specific) |

### 5.2 Rule Application Order

When an agent receives a task:

1. **Load global rules** from `CLAUDE.md` and `settings.local.json`
2. **Load agent-specific rules** from `agents/<agent>.md`
3. **Parse task description** for task-specific constraints
4. **Merge rules** with precedence: global > agent > task
5. **Resolve conflicts** by choosing higher-precedence rule

### 5.3 Rule Inheritance

```
CLAUDE.md (Project Context)
    │
    ├── Inherited by ALL agents
    │
    ▼
agents/<agent>.md (Agent Definition)
    │
    ├── Extends CLAUDE.md rules
    ├── Adds role-specific rules
    │
    ▼
Task Description (Runtime)
    │
    ├── May add task-specific constraints
    ├── Cannot override safety rules
    │
    ▼
Agent Execution (Constrained)
```

---

## 6. CROSS-CUTTING CONCERNS

### 6.1 Logging Requirements

All interactions must be logged:

| Event | Logger | Format |
|-------|--------|--------|
| Task creation | Orchestrator | Task ID, assignee, description |
| Task completion | Agent | Task ID, outcome, files changed |
| Boundary violation | validate-agent-boundaries | Agent, violation type, blocked action |
| Risk classification | risk-classifier | Action, classification, reason |
| Human decision | System | Decision, context, timestamp |

### 6.2 Error Propagation

```
Agent Error → Orchestrator → Human (if needed)
     │
     ├── Recoverable: Orchestrator retries with more context
     │
     └── Non-recoverable: Orchestrator escalates to human
```

### 6.3 State Management

| Entity | State Storage | Persistence |
|--------|---------------|-------------|
| Agents | None (stateless) | Per-invocation |
| Tasks | Task system | Session |
| Rules | File system | Permanent |
| Skills | None (stateless) | Per-invocation |

---

## 7. INTERACTION DIAGRAMS

### 7.1 Normal Workflow

```
Human ──► Orchestrator ──► Specialist Agent
              │                    │
              │                    ├── Read code
              │                    ├── Propose change
              │                    └── Submit output
              │                           │
              ◄───────────────────────────┘
              │
              ├── validate-agent-boundaries
              ├── risk-classifier
              │
              ▼
         [Review]
              │
              ├── SAFE: Task complete
              └── RISKY: Human approval needed
```

### 7.2 Conflict Resolution

```
Specialist A ──► Output A ──┐
                            │
                            ├──► Orchestrator detects conflict
                            │           │
Specialist B ──► Output B ──┘           ▼
                                   [Present both to Human]
                                        │
                                        ▼
                                   Human decides
                                        │
                              ┌─────────┴─────────┐
                              ▼                   ▼
                         Choose A            Choose B
                              │                   │
                              └─────────┬─────────┘
                                        │
                                        ▼
                               Winning output proceeds
```

### 7.3 Error Recovery

```
Agent ──► Task ──► Error
                     │
                     ▼
            Orchestrator receives error
                     │
          ┌──────────┼──────────┐
          ▼          ▼          ▼
       Retry      Escalate    Block
     (1st fail)  (2nd fail)  (3rd fail)
          │          │          │
          ▼          ▼          ▼
    More context  Human help  Stop workflow
```

---

## 8. ENFORCEMENT MECHANISMS

### 8.1 Pre-Execution Checks

Before any agent executes:
- [ ] Tools match agent's allowed list
- [ ] Action doesn't match deny patterns
- [ ] Task is within role boundaries

### 8.2 Post-Execution Validation

After any agent output:
- [ ] validate-agent-boundaries passes
- [ ] risk-classifier returns acceptable level
- [ ] Output matches task requirements

### 8.3 Continuous Monitoring

During execution:
- [ ] No attempts to access forbidden resources
- [ ] No attempts to modify other agents' files
- [ ] No attempts to bypass approval workflow

---

## Modification History

| Date | Time | Agent | Action | Details | Reason |
|------|------|-------|--------|---------|--------|
| 2026-01-30 | 09:30 | orchestrator | created | Initial interaction rules framework | User requested agent-skill-rule interaction documentation |
