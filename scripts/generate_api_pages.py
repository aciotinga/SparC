# pylint: disable=all
# type: ignore

"""Generate the code API documentation pages and navigation."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

root = Path(__file__).parent.parent
src = root / "sparc"

# Cython extension modules (no .py source on disk).
EXTENSION_MODULES = (
    "sparc.nodes",
    "sparc.eval",
    "sparc.grad",
    "sparc.metrics",
    "sparc._graph",
    "sparc.queries._engine",
    "sparc.queries.cw",
    "sparc.queries.gcw",
    "sparc.queries.expectation",
    "sparc.queries.esd",
    "sparc.solvers.transport",
    "sparc.solvers.assignment",
    "sparc.solvers.northwest",
)

seen = set()


def _add_module(ident: str, doc_path: Path) -> None:
    if ident in seen:
        return
    seen.add(ident)
    parts = tuple(ident.split("."))
    nav[parts] = doc_path.as_posix()
    full_doc_path = Path("api", doc_path)
    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        fd.write(f"::: {ident}")


for path in sorted(src.rglob("*.py")):
    module_path = path.relative_to(root).with_suffix("")
    doc_path = path.relative_to(root).with_suffix(".md")

    parts = tuple(module_path.parts)

    if parts[-1] == "__init__":
        parts = parts[:-1]
    elif parts[-1] == "__main__":
        continue

    ident = ".".join(parts)
    _add_module(ident, doc_path)
    mkdocs_gen_files.set_edit_path(Path("api", doc_path), path.relative_to(root))

for ident in EXTENSION_MODULES:
    parts = tuple(ident.split("."))
    doc_path = Path(*parts).with_suffix(".md")
    _add_module(ident, doc_path)

with mkdocs_gen_files.open("api/overview.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
