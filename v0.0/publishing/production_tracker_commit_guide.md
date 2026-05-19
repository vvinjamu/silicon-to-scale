# Production v1.0 Tracker Commit Guide

This folder contains the production tracker for **AI/ML Infrastructure from Silicon to Scale**.

## Files to commit

- `publishing/book_production_tracker.xlsx`
- `publishing/production_tracker_commit_guide.md`

## Recommended branch

`Production v1.0`

## Immediate tasks

1. Create a `/publishing/` folder in the repo.
2. Copy `book_production_tracker.xlsx` into `/publishing/`.
3. Copy this guide into `/publishing/`.
4. Commit with:

```bash
git add publishing/book_production_tracker.xlsx publishing/production_tracker_commit_guide.md
git commit -m "Add production v1.0 book tracker"
git push origin Production-v1.0
```

## Tracker tabs

- Dashboard
- Chapter Audit
- Diagram Inventory
- Print Checklist
- Web Checklist
- Technical Validation
- Marketing Assets
- Release Plan
- Source Map

## Operating rule

Do not mark a chapter as `Done` until it passes editorial, diagram, web, print, and technical validation gates.

Source site: https://vvinjamu.github.io/silicon-to-scale/
Repository: https://github.com/vvinjamu/silicon-to-scale
