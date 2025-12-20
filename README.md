# Ai-proctor-Proctovision
This project contains the final year project

## Git Workflow - Pushing Your Local Folder to the Repository

### First Time Setup
If you haven't cloned the repository yet, start here:

```bash
# Clone the repository
git clone https://github.com/avyay5311/Ai-proctor-Proctovision.git
cd Ai-proctor-Proctovision
```

### Pushing Changes to the Repository

Follow these steps to push your local changes:

1. **Check the status of your files**
```bash
git status
```

2. **Add files to staging area**
```bash
# Add all files
git add .

# Or add specific files
git add <filename>
```

3. **Commit your changes**
```bash
git commit -m "Your descriptive commit message"
```

4. **Push to the repository**
```bash
# Push to main branch
git push origin main

# Or push to current branch
git push
```

### Common Git Commands

- **View commit history**: `git log`
- **View changes**: `git diff`
- **Create a new branch**: `git checkout -b branch-name`
- **Switch branches**: `git checkout branch-name`
- **Pull latest changes**: `git pull`
- **View remote repository**: `git remote -v`

### Complete Workflow Example

```bash
# 1. Make changes to your files
# 2. Check what changed
git status

# 3. Add all changes
git add .

# 4. Commit with a message
git commit -m "Add new feature"

# 5. Push to repository
git push origin main
```

### Troubleshooting

If you encounter issues:
- **Authentication error**: Make sure you have proper access rights to the repository
- **Merge conflicts**: Pull the latest changes first with `git pull`, resolve conflicts, then push
- **Behind remote**: Run `git pull` before pushing to sync with remote changes
