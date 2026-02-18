# AI Assistant Guidelines for Interactive-Control-System

Purpose
-------
This document guides AI code assistants (and human reviewers) when reading, editing, or extending the Interactive-Control-System codebase. Follow these rules to keep changes consistent, minimal, and easy for the project owner to review.

Core principles
---------------
- Keep changes small and focused: prefer a single, minimal behavioral change per patch.
- Preserve program flow: do not reorder logic unless it improves clarity and you document why.
- Match existing naming and style: follow current conventions (snake_case for functions and variables, PascalCase for classes, UPPER_SNAKE for constants). Mirror the project's indentation, spacing, and docstring style.
- Avoid introducing unnecessary variables: reuse existing names when clear, and don't add extra temporary variables unless they improve readability.
- Minimize new dependencies: prefer using the standard library or existing helpers in the repo. If a new dependency is required, justify it and add it to `requirements.txt` only after approval.

Code edits and structure
------------------------
- Prefer small helper functions over very long functions, but avoid over-fragmentation—every function should have a clear, single responsibility.
- Keep control/flow logic explicit: avoid deeply nested inline expressions or lambda-heavy constructs that obscure the main flow.
- Use clear names reflecting intent: choose descriptive names rather than abbreviations, unless the codebase already uses consistent short names for particular concepts.
- Add docstrings for new public functions and classes. Keep docstrings concise and focused on behavior and side-effects.
- If adding type hints, follow the existing level of annotation in the file. Do not add fulltyping across the codebase in a single PR—incremental additions are preferred.

Testing and verification
------------------------
- Run existing tests locally after making changes. The project uses the `tests/` directory—run tests with `pytest`.
- Typical commands:

```
pip install -r requirements.txt
pytest -q
```

- If your change affects behavior, add or update tests under `tests/` to cover the change.

Style and formatting
--------------------
- Follow PEP 8 where it aligns with the project. Match the repository's existing formatting choices for consistency.
- Do not reformat entire files—only format the regions you change unless the project maintainer asked for a broad style pass.

Committing and PR guidance
--------------------------
- Keep commits small and focused; write concise commit messages describing why the change was made.
- In PR descriptions, summarize the problem, the change, and any test steps to verify behavior.

Communication and questions
---------------------------
- If uncertain about the intended behavior, ask a clarifying question instead of guessing.
- When suggesting larger refactors, present the benefits and a small plan, and request approval before implementing.

Contact points
--------------
If in doubt, open a short PR and request feedback rather than making large speculative changes.

Thank you for keeping contributions clear, minimal, and consistent with the project owner's expectations.
