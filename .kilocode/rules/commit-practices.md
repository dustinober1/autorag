## Brief overview
This rule defines the git commit workflow for this project. Committing at the file level ensures granular, traceable changes and simplifies code review and rollback.

## Commit granularity
- Commit each file individually rather than batching multiple file changes into a single commit
- Each commit should represent a single logical change to one file
- When modifying multiple unrelated files, create separate commits for each

## Commit messages
- Include the issue number in the commit message when applicable
- Use descriptive messages that explain what changed and why
- Reference the specific file being committed in the message

## When to commit
- Commit immediately after completing a single file's changes rather than waiting
- Push can be deferred until the end of a work session, but commits should be frequent and atomic

## Example workflow
- Modify `src/autokg_rag/cli.py` → commit immediately with message like "Add validate command to CLI (#123)"
- Modify `tests/test_cli.py` → commit separately with message like "Add tests for validate command"
- Push all commits at the end of the session
</parameter>
</new_rule>
</minimax:tool_call>
