Great question — you're absolutely right to focus on **workflow** when using `uv` and `pyproject.toml`, especially because `uv` doesn't (yet) have a built-in way to automatically add dependencies to the `.toml` like Poetry does.

But don't worry — there's a clean, efficient way to work with this.

---

## ✅ Best Workflow with `pyproject.toml` + `uv`

Here’s a **practical, modern workflow** that balances speed, reproducibility, and ease of use.

---

### 🧱 Step 1: Initialize Your Project (if not done)

If you're starting fresh, create a `pyproject.toml` manually or with a tool:

```bash
uv init  # (if this feature is available, or create manually)
```

Or manually:

```toml
# pyproject.toml
[project]
name = "my-project"
version = "0.1.0"
dependencies = []

[build-system]
requires = ["uv"]
build-backend = "uv.backend"
```

---

### 🚀 Step 2: Add Dependencies (manually or with helper command)

Since `uv` **doesn’t modify** `pyproject.toml` when installing with `uv pip install`, you **must manually declare dependencies**.

#### ✅ Recommended: Edit `pyproject.toml` directly

```toml
[project]
dependencies = [
    "requests>=2.0",
    "numpy>=1.25",
]
```

Then run:

```bash
uv install
```

This resolves dependencies, installs them, and creates a `uv.lock` file. ✅

#### ⚠️ Alternative (Not Recommended for Production): Use `uv pip install`

```bash
uv pip install requests
```

This installs immediately, but:

* It won’t persist in `pyproject.toml`
* Not ideal for teams or reproducibility

You'd then need to **manually add** `requests` to `[project.dependencies]`.

---

### 🔄 Step 3: Update Dependencies

Want to upgrade or change a version?

* Update it in `pyproject.toml`
* Then run:

```bash
uv install
```

---

### 🔒 Step 4: Lock & Share

Once dependencies are set, your `uv.lock` ensures **reproducible installs** across machines:

```bash
uv install --locked
```

This installs only what’s listed in the lockfile, ensuring **no surprises** for teammates or CI/CD.

---

### ✅ Optional Tools to Help

Until `uv` gets auto-dependency tracking, consider:

* `uv sync` (syncs current lockfile state — still needs you to declare deps in `.toml`)
* A simple alias or script that lets you "add" a dependency and update the `.toml` and install at once.

Example script (`adddep.sh`):

```bash
#!/bin/bash
dep=$1
echo "\"$dep\"," >> pyproject.toml
uv install
```

(Not perfect, but saves time.)

---

## 🧪 Bonus: Dev Dependencies?

You can add them like this:

```toml
[project.optional-dependencies]
dev = ["pytest", "black", "ruff"]
```

Then install with:

```bash
uv install --extras dev
```

---

## 🔁 Summary: Best Practices

| Step               | Action                                             |
| ------------------ | -------------------------------------------------- |
| 📦 Add deps        | Edit `pyproject.toml` manually                     |
| 🚀 Install         | Run `uv install`                                   |
| 🔐 Lockfile        | Commit `uv.lock` to version control                |
| ⚙️ Dev tools       | Use `[project.optional-dependencies]`              |
| 💡 One-off install | Use `uv pip install`, then add to `.toml` manually |

---

Let me know if you want a `pyproject.toml` template for your use case — dev, web, data science, etc.
