# Releasing sparc-pc to PyPI

The distribution name on PyPI is **`sparc-pc`**; users import it as **`sparc`**
(`pip install sparc-pc` → `import sparc`).

## One-time PyPI setup

### 1. Create accounts

- [pypi.org](https://pypi.org/account/register/)
- [test.pypi.org](https://test.pypi.org/account/register/) (for dry runs)

### 2. Claim the project name

Register **`sparc-pc`** on PyPI before the first release. The bare name `sparc`
is already taken by an unrelated project.

### 3. Configure trusted publishing (recommended)

Trusted publishing uses GitHub OIDC — no long-lived API tokens in repository
secrets.

On **PyPI** (repeat on **TestPyPI** for staging):

1. Open **Account settings → Publishing**.
2. Add a new pending publisher:
   - **PyPI project name:** `sparc-pc`
   - **Owner:** `aciotinga`
   - **Repository name:** `sparc`
   - **Workflow name:** `release.yml`
   - **Environment name:** `pypi` (for production) or `testpypi` (optional)

On **GitHub**:

1. Go to **Settings → Environments**.
2. Create environment **`pypi`**.
3. Optionally enable **Required reviewers** so production uploads need approval.
4. (Optional) Create **`testpypi`** for TestPyPI-only dry runs.

The release workflow (`.github/workflows/release.yml`) triggers on tags matching
`v*` (e.g. `v0.5.2`).

## Local validation before tagging

```bash
pip install -e ".[dev]"
python -m build
```

In a fresh virtual environment:

```bash
pip install dist/sparc_pc-*.whl
python -c "import sparc; from sparc import CategoricalInputNode; print(sparc.__version__)"
```

Source install (requires a C++17 compiler):

```bash
pip install dist/sparc_pc-*.tar.gz
```

Run the test suite against the built wheel:

```bash
pip install "dist/sparc_pc-*.whl[dev]"
pytest -q
```

## Dry run on TestPyPI (optional)

```bash
python -m build
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ sparc-pc
```

Use `--extra-index-url https://pypi.org/simple/` so pip can resolve `numpy`
and other dependencies not mirrored on TestPyPI.

## Cutting a release

1. Bump `version` in `pyproject.toml` (canonical version).
2. Commit, push, and tag:

   ```bash
   git tag v0.5.2
   git push origin v0.5.2
   ```

3. GitHub Actions builds sdist + wheels (Linux, Windows, macOS) and publishes
   to PyPI via trusted publishing.

4. Verify:

   ```bash
   pip install sparc-pc
   python -c "import sparc; print(sparc.__version__)"
   ```
