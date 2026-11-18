# Uploading to GitHub

Suggested repository name: `failure-aware-environmental-forecasting`.

Suggested description: “Code and reproducibility materials for environmental forecasting under missing, faulty, and shifted sensor measurements.”

## Web upload

1. Create a repository in your GitHub account. Choose visibility according to your submission plans. This package already includes a README.
2. Extract the repository ZIP. Upload the contents of its `failure-aware-environmental-forecasting` folder, keeping the nested folders. The README should appear at the repository root.
3. Use **Add file → Upload files**. If the browser asks for smaller batches, upload folders separately. The Git CLI is an alternative for uploading the entire tree.
4. Open **Releases → Draft a new release**. Create tag `v1.0.0` when you are ready to freeze the first public research release.
5. Attach the original `primna-tom-res.zip`. Paste the release notes from `release/RELEASE_NOTES.md`, review, and publish.
6. Add the resulting release link to the README. Confirm the author list and license before adding publication-ready citation metadata.

Raw datasets are downloaded from UCI by the code. Their sources and filenames are documented in `data/README.md`; they are not silently bundled as project-owned data.

## Git CLI alternative

Run the following inside the extracted repository folder after creating an empty GitHub repository. Replace the account/repository URL with your actual URL.

```bash
git init
git add .
git commit -m "Add forecasting code and reproducibility materials"
git branch -M main
git remote add origin https://github.com/YOUR-ACCOUNT/failure-aware-environmental-forecasting.git
git push -u origin main
```

Keep the large output ZIP outside the repository folder; upload it through Releases. This package has not created or published a GitHub repository.
