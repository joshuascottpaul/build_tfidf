# Contributing

This project values reliability, deterministic installs, and clear UX. Use the prompts and process below to keep changes consistent and legendary.

## Prompt Playbook

### 1) Mission and standards
```
Review issue. Do a dive into goals and mission. Review options and make legendary recommendations and plan with to do list.
```

### 2) SDD control
```
Make SDD and save as build_tfidf_index_sdd.txt.
Update SDD with reasoning, upgrade policy, and any config changes.
Move the to do section to SDD Decisions Locked section.
Convert scope into a detailed execution checklist with file by file changes and add to SDD.
```

### 3) Requirements and constraints
```
Must use python venv.
No em dash. No emoji. Respond in pilot talk. Keep it brief, factual, calm, and confident.
```

### 4) Implementation order
```
Review the list, optimize the order for implementation, apply the order, and proceed.
```

### 5) Quality and evaluation
```
Tighten evaluation target to 85 to 90 percent top 5 once you have baseline performance.
Make it legendary. Update SDD and move on.
```

### 6) Packaging and release
```
Files should live in build_tfidf and be pushed to GitHub, a Homebrew tap created, and a GitHub workflow action made to update the tap.
Requirements.txt must be part of brew install, tests must be in pytest, and GitHub CI must be set up.
```

### 7) Troubleshooting and tests
```
Run pytest now.
Lets troubleshoot test.
Lets troubleshoot network issue.
```

### 8) Dependency decisions
```
Pin versions for Homebrew stability. Document pins in README and how to address in the future.
```

### 9) Homebrew strategy
```
Review Homebrew options and make legendary recommendations and plan with to do list.
Document reasons and issues, then document plans to implement phase 2 in the future with reasoning.
```

### 10) Decision forcing
```
Provide numbered options with pros and cons and a recommendation, then wait for a number.
```

## Legendary prompt set (copy and paste)

### 1) Legendary upgrade
```
Make it legendary. Optimize for reliability, clarity, and zero-surprise defaults.
List tradeoffs and the smallest safe changes first.
```

### 2) Full review pass
```
Do a full pass: README + code + tests. Fix correctness first, then UX/docs.
Run tests after changes and summarize risks.
```

### 3) Test + docs discipline
```
Rebuild tests for updated scope/spec, run them, then update README and cheatsheet.
```

### 4) Release + Homebrew cycle
```
Complete the full release cycle:
1) run tests
2) commit + push
3) create release
4) update tap + SHA
5) brew upgrade locally
6) verify with brew info
```

### 5) Homebrew vendoring - future plan
```
Move to Homebrew vendored resources. Generate resource blocks, update formula,
and verify brew install. Call out optional deps explicitly.
```

### 6) Documentation audit
```
Review README for accuracy, missing info, and drift from actual behavior.
Update to match install paths and runtime requirements.
```

## Scope guardrails
- CLI only. No web UI or services.
- Keep install paths and runtime guidance accurate for macOS and Homebrew.
- Use Lets in prompts for consistency.

## Canonical process

### A) Plan
1) Define the goal and decision points.
2) Choose the minimal safe option.
3) Outline the change list and test scope.

### B) Implement
1) Update code and tests.
2) Update README/cheatsheet.
3) Run tests and capture results.

### C) Release + distribution
1) Bump version.
2) Commit + push.
3) Create release.
4) Update Homebrew tap + SHA.
5) `brew upgrade build-tfidf`.
6) Verify with `brew info build-tfidf`.

### D) Post-release verification
1) Smoke test CLI.
2) Confirm install guidance still matches behavior.
