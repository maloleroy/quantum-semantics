## Commands

Use `uv add <package>` to add runtime packages. Use `uv add --dev <package>` for development-only tools and dependencies.

```bash
uv run ruff format .
uv run ruff check .
uv run pyright .
uv run pytest -q
```
For long-running commands, if you are a Codex agent, use the $codex-long-jobs skill.

On the DGX, intensive computation must run primarily on GPU, never as a CPU-only
workload. Before substantial experiments, verify CUDA-enabled dependencies and
that model tensors, forward passes and gradients actually execute on GPU. A Slurm
GPU allocation alone is insufficient. Keep CPU work to setup, light tests, data
handling and bounded CPU comparison benchmarks explicitly requested by the user.
Use the CUDA dependency extra and pass device="cuda" in custom experiment code;
the CLI defaults to CUDA inside dgx.sbatch and CPU on the laptop.
Default to `prod10` with one `nvidia_a100_1g.10gb` MIG slice. Use `prod80`
with one `nvidia_a100-sxm4-80gb` only when measured memory requirements or a
documented compute/time requirement justify it. Override both `--partition` and
`--gres` together; do not change an experiment's batch/update budget just to fit.

## Structure and research

- Keep project code directly in `src/`, tests in `tests/`, source datasets in `data/`, general documentation in `docs/`, compact results in `docs/experiments/`, and all generated experiment artifacts and tools in `output/`.
- Read `docs/research.md` and `docs/experiments/findings.md` before model changes.
  Keep the scientific tests light. No retired job/sweep infrastructure is needed.
- Sampling success, argmax accuracy and NLL are distinct. Preserve train-only vocabulary/statistics, sentence-group partitions, the UNK policy, and a declared validation selection criterion. Lock weights before scoring held-out test.
- Use CSV for tables/histories, NPZ for large arrays/weights, and only small JSON metadata. Do not put array dumps or epoch histories in JSON.
- Type-hint every function and keep strict Pyright enabled for source and tests.
  Fix type errors rather than weakening the configuration.

## Git and production

- Format all files before committing.
- Make atomic commits: commit titles should tell the story of your work process.
- Commit titles start with a capitalized verb, e.g. "Add", "Merge", "Update", "Fix"... Any meaningful verb is allowed.
- When you are done and don't think your work requires further review before committing, partiularly for a simple task, commit without asking for permission.

## Code style

- Never use `getattr`, `hasattr`, `__dict__`, `dir`, or `try/except` (except `KeyboardInterrupt` while showing plots or awaiting text input).
- Keep imports out of `if` blocks. Imports at the top of functions are fine when used only there. Do not use `from __future__ import annotations`.
- Annotate variables whose types cannot be inferred by linting. Prefer dataclasses and functions to OOP.
