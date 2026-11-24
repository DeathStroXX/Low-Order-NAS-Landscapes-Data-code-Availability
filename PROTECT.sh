#!/bin/sh
# Write-protect the irreplaceable parts of this archive.
#
# Run this once after unpacking the archive. Archive formats do not carry file attributes, so
# an extracted copy is never protected until you run it.
#
#     sh PROTECT.sh            lock
#     sh PROTECT.sh unlock     unlock, when you really do mean to regenerate something
#     sh PROTECT.sh status     show what is locked
#
# What it protects, and why:
#
#   logs/ large_input_data/  the raw output of the 34 search runs this study reads (the 34
#                         pod logs are in large_input_data/pod_logs/, kept out of Overleaf).
#                         Nothing regenerates these without GPU-weeks. Every analysis script
#                         reads them. `sh PROTECT.sh status` prints the current size.
#   data/                 the per-architecture record and the NAS-Bench-201 table.
#                         Five are regenerable from logs/; nb201_all.pickle is not
#                         (redistributed by NASLib, and fetching it needs network access).
#                         The NAS-Bench-Suite-Zero proxy files are not shipped; if you fetch
#                         them into data/ they are protected along with everything else.
#   analysis/             the scripts, the 90 extracted LLM operator rewrites, and the
#                         computed JSON behind every number in the paper. The JSON is
#                         regenerable, but only by rerunning the pipeline -- hours of CPU,
#                         and s31, s43 and s62 dominate it.
#   tables/ figures/      every generated table and figure the two documents \input.
#   main.tex supplementary.tex bibmain.bib README.md Makefile
#
# How: the immutable attribute (chattr +i), which stops even root from removing or
# overwriting a file -- an ordinary rm fails with "Operation not permitted". Where the
# filesystem does not support it the script falls back to chmod a-w, which stops an
# unprivileged user but not root; it says which of the two it used.
#
# Unlock before rerunning any analysis script, since they rewrite analysis/results/ and
# tables/:
#
#     sh PROTECT.sh unlock && python3 analysis/s36_regret.py && sh PROTECT.sh

set -e
cd "$(dirname "$0")"

DIRS="logs large_input_data data analysis tables figures"
FILES="main.tex supplementary.tex bibmain.bib README.md Makefile"

have_chattr() {
    command -v chattr >/dev/null 2>&1 || return 1
    : > .protect_probe 2>/dev/null || return 1
    if chattr +i .protect_probe 2>/dev/null; then
        chattr -i .protect_probe 2>/dev/null
        rm -f .protect_probe
        return 0
    fi
    rm -f .protect_probe
    return 1
}

targets() {
    for d in $DIRS; do
        [ -d "$d" ] && find "$d" -type f ! -name '*.pyc'
    done
    for f in $FILES; do
        [ -f "$f" ] && echo "$f"
    done
}

case "$1" in
unlock)
    if have_chattr; then
        targets | xargs -r chattr -i 2>/dev/null || true
    fi
    targets | xargs -r chmod u+w
    echo "unlocked. Re-run 'sh PROTECT.sh' when you are finished."
    ;;
status)
    n=$(targets | wc -l | tr -d ' ')
    if have_chattr; then
        locked=$(targets | xargs -r lsattr 2>/dev/null | grep -c '^....i' || true)
        echo "$locked of $n files immutable"
    else
        locked=$(targets | xargs -r find -maxdepth 0 ! -writable 2>/dev/null | wc -l | tr -d ' ')
        echo "$locked of $n files read-only (chattr unavailable on this filesystem)"
    fi
    for d in $DIRS; do
        if [ -d "$d" ]; then
            printf '  %-20s %5s files %7s\n' "$d" \
                "$(find "$d" -type f | wc -l | tr -d ' ')" "$(du -sh "$d" | cut -f1)"
        else
            printf '  %-20s MISSING -- restore it before running the analysis\n' "$d"
        fi
    done
    ;;
*)
    missing=""
    for d in $DIRS; do
        [ -d "$d" ] || missing="$missing $d"
    done
    if have_chattr; then
        targets | xargs -r chattr +i
        how="immutable (chattr +i): even root cannot remove or overwrite them"
    else
        targets | xargs -r chmod a-w
        how="read-only (chmod a-w): chattr is unavailable here, so root can still remove them"
    fi
    echo "locked $(targets | wc -l | tr -d ' ') files -- $how"
    [ -n "$missing" ] && echo "MISSING:$missing -- restore before running the analysis"
    echo "Unlock with: sh PROTECT.sh unlock"
    ;;
esac
