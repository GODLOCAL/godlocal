#!/bin/bash
# GodLocal post-commit hook
# Appends commit summary to tasks/lessons.md for AI session memory
# Install: cp .claude/hooks/post-commit.sh .git/hooks/post-commit && chmod +x .git/hooks/post-commit

LESSONS_FILE="tasks/lessons.md"
DATE=$(date +%Y-%m-%d)
COMMIT_MSG=$(git log -1 --pretty=%B | head -1)
FILES_CHANGED=$(git diff --name-only HEAD~1 HEAD 2>/dev/null | head -10 | tr '\n' ', ')

# Only append if lessons file exists
if [ -f "$LESSONS_FILE" ]; then
  cat >> "$LESSONS_FILE" << EOF

## [$DATE] — git commit
- Message: $COMMIT_MSG
- Files: $FILES_CHANGED
EOF
  echo "📝 lessons.md updated with commit context"
fi
