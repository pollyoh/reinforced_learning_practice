---
description: Coordinate multi-agent workflow for complex tasks
user-invocable: true
---

# Coordinate Work Skill

Invoke this skill to coordinate a multi-agent workflow for implementing features, fixing bugs, or other complex tasks.

## Usage

```
/coordinate-work <description of work to be done>
```

## Examples

```
/coordinate-work Implement user authentication with login and signup
/coordinate-work Fix the checkout flow bug where cart items disappear
/coordinate-work Add dark mode support across the application
```

## Workflow

When this skill is invoked:

1. **Invoke the Orchestrator** - Use the `orchestrator` subagent to analyze the request
2. **Task Planning** - The orchestrator will break down the work into tasks
3. **Delegation** - Tasks are assigned to appropriate specialists
4. **Coordination** - Progress is tracked through the task system
5. **Completion** - Results are summarized and reported

## Instructions

To coordinate the requested work:

1. Use the `orchestrator` subagent with the following prompt:

```
Coordinate the following work request:

<user_request>
$ARGUMENTS
</user_request>

Steps to follow:
1. Analyze the request and identify required work
2. Use TaskList to check for any existing related tasks
3. Create tasks using TaskCreate for each work item
4. Assign tasks to appropriate specialists:
   - backend-specialist: API, server logic
   - frontend-specialist: UI components, client logic
   - database-specialist: Schema, migrations, queries
   - qa-specialist: Testing
   - docs-specialist: Documentation
   - design-reviewer: Design compliance review

5. Provide a summary of the plan to the user
6. Track progress and handle any blockers

Remember to follow the Error & Recovery Rules:
- Retry failed tasks once with more detail
- Escalate after retry fails
- Block on ambiguity with clear questions
- Pause on major scope changes
```

2. Monitor the orchestrator's progress and relay results to the user

## Notes

- The orchestrator uses the task system (TaskCreate, TaskList, etc.) as a shared work queue
- Subagents cannot spawn other subagents directly
- Each specialist starts fresh - provide full context in task descriptions
- Use this skill for complex, multi-step work that benefits from coordination
