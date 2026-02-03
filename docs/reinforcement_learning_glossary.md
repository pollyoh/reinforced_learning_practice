# Reinforcement Learning Glossary for LLM Training

A beginner-friendly guide to understanding reinforcement learning concepts, specifically focused on how RL is used to train and improve Large Language Model (LLM) agents.

---

## Table of Contents

1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Core Terminology](#core-terminology)
3. [RL Algorithms for LLM Training](#rl-algorithms-for-llm-training)
4. [Putting It All Together](#putting-it-all-together)

---

## Introduction to Reinforcement Learning

### What is Reinforcement Learning?

Reinforcement Learning (RL) is a type of machine learning where an **agent** learns to make decisions by interacting with an **environment** and receiving **feedback** (rewards or penalties) based on its actions.

Think of it like training a dog:
- The dog (agent) tries different behaviors
- You (environment) give treats (positive rewards) for good behavior
- The dog learns which behaviors lead to treats

For LLMs, RL helps the model learn to generate responses that humans find helpful, harmless, and honest.

### Why Use RL for LLM Training?

LLMs are initially trained on massive text datasets (pre-training), but this alone doesn't teach them:
- What humans actually prefer
- How to be helpful without being harmful
- How to follow instructions precisely

RL bridges this gap by:
1. Letting the model generate responses
2. Getting feedback on those responses (from humans or reward models)
3. Updating the model to produce better responses

### How RL Differs from Other Learning Types

```
+------------------+------------------------+---------------------------+
|   Learning Type  |      Training Signal   |         Example           |
+------------------+------------------------+---------------------------+
| Supervised       | Correct answers        | "The capital of France    |
|                  | provided directly      | is Paris" (given answer)  |
+------------------+------------------------+---------------------------+
| Unsupervised     | No labels, finds       | Clustering similar        |
|                  | patterns in data       | documents together        |
+------------------+------------------------+---------------------------+
| Reinforcement    | Rewards/penalties      | "That response was        |
|                  | after actions          | helpful" (+1 reward)      |
+------------------+------------------------+---------------------------+
```

**Key difference**: In RL, the model doesn't get told the "right" answer. It explores, tries things, and learns from the consequences.

---

## Core Terminology

### Agent

**Definition**: The learner and decision-maker in the RL system.

**In LLM context**: The language model itself. It observes a prompt (state) and generates a response (action).

```
+------------------+
|      AGENT       |
|   (The LLM)      |
|                  |
| Observes state   |
| Takes actions    |
| Learns from      |
| rewards          |
+------------------+
```

**Example**: ChatGPT is an agent. When you ask it a question, it decides what to say based on what it has learned.

---

### Environment

**Definition**: Everything the agent interacts with. It receives actions from the agent and returns new states and rewards.

**In LLM context**: The environment includes:
- The user providing prompts
- The reward model evaluating responses
- The conversation context

```
     +----------+          action           +---------------+
     |          | -----------------------> |               |
     |  AGENT   |                          |  ENVIRONMENT  |
     |  (LLM)   | <----------------------- |  (User +      |
     |          |    state + reward        |   Reward      |
     +----------+                          |   Model)      |
                                           +---------------+
```

**Example**: When you chat with an LLM, you are part of its environment. Your follow-up questions and feedback shape what it learns.

---

### State

**Definition**: The current situation or observation that the agent uses to make decisions.

**In LLM context**: The state is the input the model receives, including:
- The current prompt/question
- Conversation history
- System instructions

**Example**:
```
State (what the LLM sees):
- System: "You are a helpful assistant"
- User: "What is machine learning?"
- Previous response: "Machine learning is..."
- User: "Can you explain it more simply?"  <-- Current state
```

---

### Action

**Definition**: A choice the agent makes in response to a state.

**In LLM context**: The action is the response the model generates. This includes:
- The specific words chosen
- The length of response
- The tone and style

**Example**:
```
State: "Explain quantum computing to a 10-year-old"

Possible Actions:
- Action A: "Quantum computing uses qubits that can be 0 and 1
            simultaneously, like a coin spinning in the air..."
- Action B: "Quantum computing is a type of computation that
            harnesses quantum mechanical phenomena..."
```

Action A might receive a higher reward for being more appropriate for the audience.

---

### Reward

**Definition**: A numerical signal that tells the agent how good or bad its action was.

**In LLM context**: Rewards come from:
- Human evaluators rating responses
- Reward models trained on human preferences
- Automated metrics (helpfulness, harmlessness, etc.)

```
     Response Quality          Reward
     -----------------         ------
     Helpful and accurate      +1.0
     Partially helpful         +0.3
     Unhelpful                 -0.5
     Harmful/toxic             -2.0
```

**Example**:
```
Prompt: "How do I make a cake?"

Response A: "Preheat oven to 350F. Mix flour, sugar..."  --> Reward: +0.9
Response B: "I don't know."                              --> Reward: -0.5
Response C: "Cakes are unhealthy, you shouldn't eat..."  --> Reward: -0.3
```

---

### Policy (π)

**Definition**: The strategy or rule that determines which action to take in each state. This is the core of what the agent learns.

**In LLM context**: The policy is essentially the model's weights/parameters that determine how it generates responses. When we "train" an LLM with RL, we're updating its policy.

```
+-------------+                    +------------+
|   STATE     |  ---> Policy ---> |   ACTION   |
| (prompt)    |       (π)         | (response) |
+-------------+                    +------------+

Policy = "Given this input, what should I output?"
```

**Why this is KEY**: The entire goal of RL training is to find a good policy. A good policy:
- Consistently produces helpful responses
- Avoids harmful outputs
- Matches human preferences

**Types of policies**:
- **Deterministic**: Same state always produces same action
- **Stochastic**: Same state can produce different actions (with probabilities)

LLMs use stochastic policies -- that's why you get slightly different responses to the same prompt.

---

### Value Function

**Definition**: A prediction of how much total reward the agent expects to receive from a state (or state-action pair) going forward.

**In LLM context**: The value function estimates "if we start from this conversation point, how good will the final outcome be?"

```
State: User asks a complex coding question

High Value State:
- Clear question
- Sufficient context
- Expected reward: High (can give helpful answer)

Low Value State:
- Ambiguous question
- Missing information
- Expected reward: Low (likely to fail)
```

**Two types**:
1. **State Value V(s)**: Expected reward starting from state s
2. **Action Value Q(s,a)**: Expected reward after taking action a in state s

---

### Episode

**Definition**: A complete sequence of interactions from start to finish.

**In LLM context**: An episode could be:
- A single prompt-response pair
- An entire conversation
- A task completion attempt

```
Episode Example (Single Turn):
+--------+     +-----------+     +--------+     +----+
| Prompt | --> | LLM thinks| --> |Response| --> |End |
+--------+     +-----------+     +--------+     +----+
   t=0             t=1              t=2          Done

Episode Example (Multi-Turn Conversation):
+----+   +----+   +----+   +----+   +----+   +-----+
| Q1 |-->| A1 |-->| Q2 |-->| A2 |-->| Q3 |-->| END |
+----+   +----+   +----+   +----+   +----+   +-----+
 t=0      t=1      t=2      t=3      t=4      Done
```

---

### Trajectory

**Definition**: The specific sequence of states, actions, and rewards that occurred during an episode.

**In LLM context**: A trajectory records exactly what happened:

```
Trajectory = [(s0, a0, r0), (s1, a1, r1), (s2, a2, r2), ...]

Example:
t=0: State="Hi" --> Action="Hello! How can I help?" --> Reward=+0.8
t=1: State="What's 2+2?" --> Action="4" --> Reward=+1.0
t=2: State="Thanks!" --> Action="You're welcome!" --> Reward=+0.9

Total trajectory reward = 0.8 + 1.0 + 0.9 = 2.7
```

Trajectories are collected during training to update the policy.

---

### Discount Factor (γ)

**Definition**: A number between 0 and 1 that determines how much the agent cares about future rewards versus immediate rewards.

**Symbol**: γ (gamma)

```
γ = 0.0  --> Only care about immediate reward
γ = 0.5  --> Future rewards worth half as much
γ = 0.99 --> Future rewards almost as valuable as immediate
γ = 1.0  --> Future rewards equally valuable (can be unstable)
```

**In LLM context**: Helps the model think ahead:
- Should I give a quick answer now? (immediate reward)
- Or ask a clarifying question for a better answer later? (future reward)

**Example calculation**:
```
Immediate reward: 1.0
Reward in 1 step: 2.0
Reward in 2 steps: 3.0

With γ = 0.9:
Total value = 1.0 + (0.9 * 2.0) + (0.9² * 3.0)
            = 1.0 + 1.8 + 2.43
            = 5.23
```

---

### Exploration vs Exploitation

**Definition**: The fundamental trade-off between:
- **Exploration**: Trying new, uncertain actions to discover better options
- **Exploitation**: Using known good actions to maximize reward

**In LLM context**:

```
+------------------+------------------------------------------+
|   Exploration    |   "Let me try a creative new approach   |
|                  |    to answering this question..."        |
+------------------+------------------------------------------+
|   Exploitation   |   "I'll use the response style that     |
|                  |    has worked well before..."            |
+------------------+------------------------------------------+
```

**The dilemma**:
- Too much exploration: Wastes time on bad responses
- Too much exploitation: Never discovers better approaches

**How LLMs handle this**: Temperature parameter controls randomness
- Low temperature (0.1): More exploitation, predictable outputs
- High temperature (1.0+): More exploration, creative/random outputs

---

## RL Algorithms for LLM Training

### Policy Gradient Methods

**What it is**: A family of algorithms that directly optimize the policy by computing gradients (directions of improvement) based on rewards.

**Core idea**:
1. Generate responses using current policy
2. Calculate rewards for each response
3. Increase probability of high-reward actions
4. Decrease probability of low-reward actions

```
Simple Policy Gradient Update:

Before training:
  "What's 2+2?"  -->  P("4")=0.3, P("5")=0.2, P("22")=0.2 ...

After reward (+1 for "4"):
  "What's 2+2?"  -->  P("4")=0.5, P("5")=0.1, P("22")=0.1 ...
                           ^
                      Increased!
```

**Strengths**: Simple, works directly on the policy
**Weaknesses**: High variance, can be unstable

---

### PPO (Proximal Policy Optimization)

**What it is**: A popular policy gradient method that prevents the policy from changing too much in a single update.

**Why "Proximal"**: It keeps updates close (proximal) to the current policy, making training more stable.

```
Standard Policy Gradient:
  Old Policy --[big jump]--> New Policy (might be bad!)

PPO:
  Old Policy --[small step]--> New Policy (safer)

  If update is too big --> Clip it back
```

**Key mechanism - Clipping**:
```
ratio = P_new(action) / P_old(action)

If ratio > 1.2 --> Clip to 1.2 (don't increase too much)
If ratio < 0.8 --> Clip to 0.8 (don't decrease too much)
```

**Why PPO is popular for LLMs**:
- Stable training (less likely to break the model)
- Works well with large models
- Balances learning speed and safety

---

### RLHF (Reinforcement Learning from Human Feedback)

**What it is**: A technique that uses human preferences to train a reward model, which then guides RL training.

**The problem it solves**: How do we define "good" responses mathematically? Humans know a good response when they see one, but can't easily write rules for it.

**The RLHF Pipeline**:

```
Step 1: Collect Human Preferences
+----------------+     +----------------+
| Response A     |     | Response B     |
| "The answer    | vs  | "42"           |
|  is 42..."     |     |                |
+----------------+     +----------------+
         ^
    Human picks A as better


Step 2: Train Reward Model
+------------------+     +--------+
| (Prompt,Response)| --> | Reward |
|                  |     | Model  | --> Score: 0.85
+------------------+     +--------+


Step 3: RL Training with Reward Model
+------+     +-----------+     +--------+     +--------+
| LLM  | --> | Response  | --> | Reward | --> | Update |
|      |     |           |     | Model  |     | Policy |
+------+     +-----------+     +--------+     +--------+
                                  |
                           Score: 0.85
```

**Complete RLHF Workflow**:

```
1. Pre-trained LLM (knows language, but not preferences)
              |
              v
2. Supervised Fine-Tuning (SFT) on high-quality examples
              |
              v
3. Reward Model training (learns human preferences)
              |
              v
4. PPO training (optimizes LLM using reward model)
              |
              v
5. Final aligned LLM (helpful, harmless, honest)
```

**Why RLHF matters**: It's how models like ChatGPT learned to be helpful and avoid harmful outputs.

---

### DPO (Direct Preference Optimization)

**What it is**: A simpler alternative to RLHF that skips the reward model entirely.

**The insight**: Instead of training a separate reward model, directly update the LLM using preference pairs.

**RLHF vs DPO**:

```
RLHF (Complex):
Preferences --> Reward Model --> PPO --> Updated LLM
     ^               ^           ^
     |               |           |
   3 steps    Can be unstable   Complicated

DPO (Simpler):
Preferences --> Direct Optimization --> Updated LLM
     ^                  ^
     |                  |
   1 step         More stable
```

**How DPO works**:

Given pairs of (preferred response, rejected response):
1. Increase probability of preferred response
2. Decrease probability of rejected response
3. Use a clever loss function that implicitly learns the reward

```
Training pair:
  Prompt: "Explain gravity"
  Preferred: "Gravity is a force that attracts objects..."
  Rejected: "idk lol"

DPO updates:
  P("Gravity is a force...") --> Increase
  P("idk lol") --> Decrease
```

**Advantages of DPO**:
- Simpler implementation
- No need to train a separate reward model
- More stable training
- Faster to train

**Current trend**: Many newer models use DPO or DPO variants instead of full RLHF.

---

## Putting It All Together

### The Complete RL Loop for LLMs

```
+------------------------------------------------------------------+
|                        RL TRAINING LOOP                           |
+------------------------------------------------------------------+
|                                                                   |
|  1. SAMPLE                                                        |
|     +-------+       +--------+       +----------+                 |
|     | Prompt| ----> |  LLM   | ----> | Response |                 |
|     +-------+       | (Agent)|       | (Action) |                 |
|                     +--------+                                    |
|                                                                   |
|  2. EVALUATE                                                      |
|     +----------+       +--------+       +--------+                |
|     | Response | ----> | Reward | ----> | Score  |                |
|     +----------+       | Model  |       | (r=0.8)|                |
|                        +--------+                                 |
|                                                                   |
|  3. UPDATE                                                        |
|     +-------+       +-------+       +---------+                   |
|     | Score | ----> |  PPO  | ----> | Updated |                   |
|     |       |       |       |       | Policy  |                   |
|     +-------+       +-------+       +---------+                   |
|                                                                   |
|  4. REPEAT                                                        |
|     Loop back to step 1 with updated policy                       |
|                                                                   |
+------------------------------------------------------------------+
```

### Key Relationships

```
+----------------+          +------------------+
|    POLICY      |  guides  |     ACTIONS      |
| (LLM weights)  | -------> | (responses)      |
+----------------+          +------------------+
       ^                            |
       |                            |
   updated by                   evaluated by
       |                            |
       |                            v
+----------------+          +------------------+
|   RL ALGORITHM |  uses    |     REWARDS      |
|   (PPO/DPO)    | <------- | (human feedback) |
+----------------+          +------------------+
```

### Summary Table

| Concept | Simple Definition | LLM Example |
|---------|-------------------|-------------|
| Agent | The learner | The LLM model |
| Environment | What agent interacts with | User + reward model |
| State | Current situation | Prompt + context |
| Action | What agent does | Generated response |
| Reward | Feedback signal | Helpfulness score |
| Policy | Strategy for acting | Model weights |
| Value | Expected future reward | Conversation quality prediction |
| Episode | Complete interaction | Full conversation |
| Trajectory | Sequence of events | All turns + rewards |
| Discount | Future reward importance | How much to plan ahead |
| Exploration | Trying new things | Creative responses |
| Exploitation | Using known good | Proven response patterns |

---

## Further Reading

To deepen your understanding:

1. **Sutton & Barto - Reinforcement Learning: An Introduction** (free online)
   - The classic RL textbook

2. **InstructGPT Paper** (OpenAI, 2022)
   - First major paper on RLHF for LLMs

3. **DPO Paper** - "Direct Preference Optimization" (2023)
   - The simpler alternative to RLHF

4. **Anthropic's Constitutional AI Paper**
   - How to align AI using principles

---

*Last updated: 2026-02-03*
*Created for developers new to reinforcement learning in the context of LLM training*
