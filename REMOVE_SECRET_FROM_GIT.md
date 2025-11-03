# How to Remove Secret from Git History

## Issue
GitHub detected a Hugging Face token in commit `1b91c7d8352b4d744de9be0b12fadd67f4ab3346`

## Steps to Fix

### Option 1: Rewrite Git History (Recommended)

```bash
# Install git-filter-repo if not installed
pip install git-filter-repo

# Remove the secret from all commits
git filter-repo --path kaggle_dataset/hf.py --invert-paths

# OR if git-filter-repo is not available, use BFG:
# java -jar bfg.jar --delete-files kaggle_dataset/hf.py
```

### Option 2: Use git filter-branch (Built-in)

```bash
# Remove the file from entire history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch kaggle_dataset/hf.py" \
  --prune-empty --tag-name-filter cat -- --all

# Force push (WARNING: This rewrites history)
git push origin feature-slm --force
```

### Option 3: Allow Secret (Not Recommended)

If you must keep the token, you can temporarily allow it:
- Visit: https://github.com/VanshShah2004/test1/security/secret-scanning/unblock-secret/34xEamB2812Jj2Y8sFnMz5FwSMb
- But this is NOT recommended - always revoke exposed tokens!

## After Fixing

1. ✅ Revoke the token on Hugging Face (already done above)
2. ✅ Remove token from current file (done - now uses environment variable)
3. ✅ Remove from git history (use one of the options above)
4. ✅ Commit the fixed file:
   ```bash
   git add kaggle_dataset/hf.py
   git commit -m "Remove hardcoded token, use environment variable instead"
   git push origin feature-slm
   ```

## Prevention

✅ Token is now stored as environment variable
✅ File is in .gitignore
✅ Never commit secrets to git!

