# Security Guidelines

## Overview

This document outlines security best practices for developing and deploying the OmniCellAgent project. **Never commit credentials, API keys, or sensitive information to the repository.**

## Sensitive Files

The following types of files are **automatically ignored** by git and should NEVER be committed:

- `.env` and `.env.*` files
- `configs/*.env` files  
- Private keys (`.pem`, `.key`, `.crt`, `.p12`, `.pfx`)
- Any files containing credentials or secrets

## Environment Variables Setup

### 1. Root .env File

Copy the template and fill in your credentials:

```bash
cp .env.example .env
```

Then edit `.env` with your actual API keys:

```bash
# OpenAI Configuration
OPENAI_API_KEY="sk-..." 

# Google/Gemini Configuration
GEMINI_API_KEY="AIza..."
GOOGLE_API_KEY="AIza..."
GOOGLE_SEARCH_ENGINE_ID="..."

# HuggingFace Configuration  
HF_TOKEN="hf_..."
```

### 2. Database Configuration

Copy the template and set up your database credentials:

```bash
cp configs/db.env.example configs/db.env
```

Then edit `configs/db.env`:

```bash
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your-secure-password
OPENAI_API_KEY=sk-...
```

## Required API Keys

To run this project, you'll need:

### 1. OpenAI API Key
- Get it from: https://platform.openai.com/api-keys
- Used for LLM operations
- Set in: `OPENAI_API_KEY`

### 2. Google/Gemini API Keys  
- Get them from: https://ai.google.dev/
- Used for search and AI operations
- Set in: `GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GOOGLE_SEARCH_ENGINE_ID`

### 3. HuggingFace Token
- Get it from: https://huggingface.co/settings/tokens
- Used for model access and downloads
- Set in: `HF_TOKEN`

### 4. Neo4j Database
- Default username: `neo4j`
- Set a strong password during Neo4j setup
- Set in: `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`

## Development Workflow

1. **Always use `.env` files locally** - Never hardcode credentials
2. **Check .gitignore before committing** - Run `git status` to verify no `.env` files are staged
3. **Use environment variables in code** - Access keys via `os.getenv("KEY_NAME")`
4. **Rotate keys regularly** - If a key is accidentally exposed, rotate it immediately
5. **Review commits before pushing** - Use `git diff` to check for credentials

## Pre-commit Hook (Optional)

To prevent accidental credential commits, add a pre-commit hook:

```bash
#!/bin/bash
# .git/hooks/pre-commit

if git diff --cached | grep -E "sk-proj|AIza|hf_"; then
    echo "ERROR: Potential API key detected in staged changes!"
    echo "Please remove credentials before committing"
    exit 1
fi
```

## If Credentials Are Accidentally Committed

### Immediate Actions:

1. **Rotate the exposed credential immediately** - regenerate API keys in their respective services
2. **Remove from git history** - Use `git filter-repo` to clean history
3. **Force push** - `git push --force-with-lease` (only if you own the repo)

### Example Cleanup:

```bash
# Install git-filter-repo if not already installed
pip install git-filter-repo

# Remove a file from all commits
git filter-repo --path configs/db.env --invert-paths --force

# Force push to remote (only for owned repos)
git push origin --force-with-lease
```

## Deployment Security

### Production Environment Variables

Use secure environment variable management:

- **Docker**: Pass via `--env-file` or environment variables
- **Cloud Platforms**: Use managed secrets (AWS Secrets Manager, Google Secret Manager, etc.)
- **CI/CD**: Use platform-specific secret storage (GitHub Secrets, GitLab CI/CD Variables, etc.)

### Example: Docker Deployment

```bash
# Create secure .env file (not in repo)
docker run --env-file /path/to/.env myapp

# Or pass individual variables
docker run -e OPENAI_API_KEY="$OPENAI_API_KEY" myapp
```

## Code Review Checklist

Before submitting pull requests, verify:

- [ ] No `.env` files are staged for commit
- [ ] No hardcoded API keys in code
- [ ] All secrets use environment variables
- [ ] `.gitignore` properly covers sensitive files
- [ ] No debug logs containing credentials
- [ ] Configuration examples use placeholders

## Reporting Security Issues

If you discover a security vulnerability:

1. **Do not open a public issue** - Send details privately to the maintainers
2. **Include affected versions** - Specify which versions are vulnerable
3. **Provide reproduction steps** - Help us understand the issue
4. **Allow time for fixes** - Give maintainers time to respond before public disclosure

## Additional Resources

- [OWASP: Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)
- [GitHub: Managing sensitive data](https://docs.github.com/en/code-security/secret-scanning/about-secret-scanning)
- [Git: Removing sensitive data](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/removing-sensitive-data-from-a-repository)

---

**Last Updated**: January 2026

**Repository**: bioRAG

**Branch**: omic-neo4j
