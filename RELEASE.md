
# Releasing and publishing to PyPI

This document describes how to cut a new GitHub release and publish packages to PyPI.

> **Prerequisites**
> - Write permissions on the GitHub repo (required to draft/publish releases).
> - Access to the release workflows (GitHub Actions) that build and publish to PyPI.

---

## Versioning convention
Before publishing a new release, ensure that the **version number** has been updated in `pyproject.toml`. 
The convention followed for package versioning is: 
- `<package>@v<major>.<minor>.<patch>` (e.g., `evo-objects@v1.2.3`).

---

## Tagging & drafting the GitHub release

1. Go to the repo **Releases** tab and click **“Draft a new release”**.
2. **Create or choose a new tag** to publish using the convention: `<package>@v<major>.<minor>.<patch>`.

    - **Create a new tag** if the version you’re releasing does **not** already exist as a Git tag in the repository.  
    Example: You bumped `pyproject.toml` to `v2025.12.17` and haven’t pushed that tag yet. GitHub will create the tag on the commit you specify (usually `main`).

    - **Choose an existing tag** if you’ve already created and pushed the tag locally (e.g., `git tag v2025.12.17 && git push origin v2025.12.17`).  
    In this case, just select it in the release UI.

    > **Tip:** The release workflow uses the tag to build and publish artifacts, so make sure the tag matches the version in `pyproject.toml`.
3. **Select the previous tag**:
   - Pick the previous tag **for the specific package** being released (not just any repo tag).
4. Click **“Generate release notes”** and review:
   - Where appropriate, remove change details unrelated to the package you’re releasing.
   - If notes look wrong, it usually means the **previous tag selection** was incorrect. You can re-select the correct tag and regenerate to fix.
7. Click **“Publish release”**.

---

## CI/CD: build & publish

Publishing the release triggers the release workflow, which:
- **Builds the assets** and publishes them to **GitHub Releases** and **PyPI**.
- **Does not** bump version numbers automatically (these were updated manually earlier).

After publishing:
- The **new version appears on PyPI** almost immediately.
- The **GitHub README on the PyPI page** can take some time to refresh due to caching.

---

## Troubleshooting

- **Incorrect release notes**: Recheck the *previous tag* selection and regenerate notes.
- **PyPI publish failed**: Verify `pyproject.toml` versions, tags, and that CI permissions are valid.
- **Notes too noisy**: Manually curate the generated notes to only include changes for the package being released.

---

## References

- Repository structure and maintainer docs: see `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`.  
- PyPI (metapackage and sub-packages) for visibility of published versions.
