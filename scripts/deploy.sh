#!/bin/bash
# PipelineCore Deployment Script
# This script automates the process of pushing, creating PR, and releasing

set -e  # Exit on error

echo "🚀 PipelineCore Deployment Script"
echo "=================================="
echo ""

# Step 1: Push to remote
echo "📤 Step 1: Pushing to remote repository..."
git push -u origin feature/enhance-project-configuration
echo "✅ Push completed!"
echo ""

# Step 2: Create Pull Request
echo "🔀 Step 2: Creating Pull Request..."
gh pr create --title "feat: enhance project configuration and add i18n documentation" \
  --body "## 📝 Summary

This PR enhances the project configuration and adds comprehensive internationalized documentation.

## ✨ Changes

### 1. Enhanced Project Configuration (pyproject.toml)
- ✅ Add project metadata (classifiers, keywords, URLs)
- ✅ Add core dependencies (tensorflow >=2.10.0, pynvml >=11.0.0)
- ✅ Add dependency groups (dev, test, linting, docs)
- ✅ Configure development tools (ruff, mypy, pytest, coverage)
- ✅ Change build backend to hatchling

### 2. Internationalized Documentation
- ✅ Root README.md - Brief overview with bilingual links
- ✅ docs/README.md - Complete English documentation
- ✅ docs/README.zh-TW.md - Complete Traditional Chinese documentation
- ✅ docs/INSTALLATION_GUIDE.md - Bilingual installation guide
- ✅ CLAUDE.md - Claude Code integration guide

### 3. Package Structure Improvements
- ✅ Add __version__ attribute
- ✅ Proper package imports
- ✅ Export all necessary classes

## 📊 Statistics
10 files changed, 1,418 insertions(+), 1 deletion(-)

🤖 Generated with [Claude Code](https://claude.com/claude-code)" \
  --base main \
  --head feature/enhance-project-configuration

PR_URL=$(gh pr view --json url -q .url)
echo "✅ Pull Request created: $PR_URL"
echo ""

# Wait for user to merge PR
echo "⏸️  Please merge the PR in GitHub, then press Enter to continue..."
read -r

# Step 3: Create Release
echo "🏷️  Step 3: Creating Release..."

# Switch to main and pull latest
git checkout main
git pull origin main

# Create and push tag
git tag -a v0.1.0 -m "Release v0.1.0 - Initial release"
git push origin v0.1.0

# Create GitHub release
gh release create v0.1.0 \
  --title "v0.1.0 - Initial Release" \
  --notes "## 🎉 PipelineCore v0.1.0

### First Official Release

A Python framework for building ML/inference pipelines with GPU resource management.

### Features
- Template Method Pattern pipeline framework
- GPU resource management with TensorFlow
- Type-safe generics
- Comprehensive bilingual documentation

### Installation
\`\`\`bash
uv pip install git+https://github.com/ggggggg13289AI/pipelinecore.git@v0.1.0
\`\`\`

🤖 Generated with [Claude Code](https://claude.com/claude-code)"

echo "✅ Release v0.1.0 created!"
echo ""
echo "🎉 Deployment completed successfully!"
echo "=================================="
