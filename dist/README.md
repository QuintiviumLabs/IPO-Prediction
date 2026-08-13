# dist/

`ipo-model-update.zip` — a packaged copy of every file added or changed since
commit `f19aef0` (the run-book commit), for machines where files are moved by
hand rather than by `git pull`.

Download it from GitHub's file view (**Download raw file**), unzip over the
repo root keeping the folder structure, and follow `MANIFEST.txt` inside.

**This is a snapshot, not a moving target.** It reflects the repo at the
commit where it was added and does not update itself. If the branch has moved
on since, prefer `git pull`, or ask for a fresh package. To rebuild it from
any checkout:

```bash
git diff --name-only f19aef0 HEAD > /tmp/changed.txt
mkdir -p /tmp/pkg && while read -r f; do
  [ -f "$f" ] && mkdir -p "/tmp/pkg/$(dirname "$f")" && cp "$f" "/tmp/pkg/$f"
done < /tmp/changed.txt
(cd /tmp/pkg && zip -r ../ipo-model-update.zip .)
```
