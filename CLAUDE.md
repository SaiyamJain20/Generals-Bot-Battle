# Bot-Battle: instructions for Claude

Code Bot hackathon bot (generals.bot competition ruleset). The plan is in
`~/.claude/plans/so-i-have-given-luminous-adleman.md`, and unanswered items are in `OPEN_QUESTIONS.md`.

**Submission constraints.**
- One stdlib-only Python 3.12 file, ≤ 1 MiB, exposing `act(observation)`.
- 150 ms per move.
- A timeout or exception forfeits the game.

**Ground truth.**
- The pinned engine is in `vendor/generals-bots` (commit 13db8f69).
- `sim/engine.py` is our exact replica of it, checked by `tests/test_parity.py`.
- Local env: `.venv312/bin/python` with `PYTHONPATH=vendor/generals-bots:.`

---

## Ada cluster: hard rules

The account belongs to someone else (`<cluster_user>`). We are guests.

### Access
- `ssh -o BatchMode=yes <cluster_user>@<login_node>`. Key-based login already works.
- Never ask for, print or store a password.

### 1. Never touch anything we did not create
Do not modify, move, overwrite or delete any existing file, folder, job or setting. That includes:
- **Home** (`/home2/$USER`): `hackathon/`, `nlp-hc/`, `submission/`, `submission.tar.gz`, `miniconda3/`, and every dotfile and dot-dir (`.bashrc`, `.bash_profile`, `.zshrc`, `.ssh/`, `.conda/`, `.config/`, `.cache/`, `.local/`, `.nv/`, `.triton/`, `.humming/`, `.gitconfig`, …).
- **Scratch**: the existing contents of `/scratch/$USER/` on any node. They predate us.
- **Other storage**: `/share1` (their quota), and other users' directories.
- **Jobs**: never `scancel` a job we didn't submit.

### 2. No global or shared installs
- No `pip install --user`.
- No `conda create`, `conda install` or `conda init`. Do not use or modify `~/miniconda3` at all.
- No edits to shell rc files, and no `module` default changes.

### 3. Keep tool caches off home
Set these in every job and command:
`PIP_CACHE_DIR`, `XDG_CACHE_HOME`, `UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `JAX_COMPILATION_CACHE_DIR`, `MPLCONFIGDIR`, `TMPDIR`.
Point them all inside our own scratch directory.

### 4. Before any `rm`
- Use only the explicit absolute paths listed in `ada_manifest.txt`.
- Never use globs or unset variables (`rm -rf $DIR/` with `DIR` empty is catastrophic).
- `ls` the exact target first and confirm it is ours.

### 5. Log everything we create
Every path and job ID we create on Ada goes into the local `ada_manifest.txt`. Cleanup removes exactly those, and nothing else.

---

## Where our things live (verified 2026-10-02)

### `/scratch` (the ~7-day auto-purged storage)
- It exists **only on compute nodes**, not on the login node `<login_node>`.
- It is **node-local**: `/dev/sdb1`, 1.8 TB, about 1.4 TB free on gnode002.
- **All our venvs, data, logs and results go in `/scratch/$USER/botbattle-saiyam/`** (create it with `mkdir -p` inside jobs).
- Because the disk is node-local, every job must:
  1. stage code in at start;
  2. copy summaries out before it ends.

### `/home2/$USER/botbattle-saiyam/` (minimal staging dir, shared NFS)
- Needed because the login node has no `/scratch`.
- Holds only rsynced source code, `sbatch` scripts and small result summaries (JSON/CSV).
- **Target < 200 MB and < 2,000 files.** Home has 25 GB free, and it is their quota.
- Delete it when the hackathon ends.

### Running jobs
- **Login node:** no compute (it has tight per-process memory limits). Use `sbatch`/`srun` only.
- **Verified working flags:** `srun -A research --qos=low -p u22 -n 1 -c <N> --mem=<M> --time=<T>`.
- **Limits** (`research`/`low`): 10 CPUs, 1 GPU, 32 GB, 5 jobs, 4-day wall time.
- **Not yet tested:** the account `irel` with QoS `normal` (listed with no limits).
- **Partitions:**
  - `u22`: 40-core nodes, 2080 Ti GPUs. **Allowed.**
  - `u22-cpu`: only account `devalab`. **Not allowed** for us.
- **Python:** compute nodes run Ubuntu 22.04. For Python 3.12, install standalone `uv` into scratch and create the venv there, never in home.

### Moving code
`rsync -az --exclude .venv312 --exclude data --exclude vendor ./ <cluster_user>@<login_node>:~/botbattle-saiyam/`, then each job copies it to `/scratch/$USER/botbattle-saiyam/` on its node.
