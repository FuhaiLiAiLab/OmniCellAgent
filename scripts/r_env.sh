#!/usr/bin/env bash
# Set up R for the agent. The omics workflow calls a bare `Rscript` (tools/omic_tools/subprocess_r.py),
# so R must be on PATH in the shell that runs the agent. R comes from the cluster module rather than a
# fixed spack path, because RIS rebuilds put R under a new hash directory. The script then checks that
# every package the agent's R scripts load is installed in R_LIBS.
#
# Usage:
#   source scripts/r_env.sh                      set PATH and R_LIBS in the current shell
#   scripts/r_env.sh <command> [args...]         run one command with R set up, e.g.
#       scripts/r_env.sh /storage3/.../miniconda/omnicellagent/bin/python agent/langgraph_agent.py
#   scripts/r_env.sh                             check only; the settings end with the script
#
# Settings (environment variables; defaults in brackets):
#   R_MODULE   module that provides Rscript      [r/4.5.2]
#   R_LIBS     R package library                 [/storage1/fs1/fuhai.li/Active/di.huang/cache/R]
#
# No `set -e`/`exit` at top level: a sourced script would apply them to (or close) the caller's shell.

# Package names loaded by an R script through library(...) or requireNamespace(...)
_r_env_packages() {
    grep -oE '(library|requireNamespace)\("?[A-Za-z0-9.]+' "$1" | sed -E 's/.*\("?//' | sort -u | tr '\n' ' '
}

_r_env_setup() {
    local project_root module required optional missing pkg missing_required="" missing_optional=""
    project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    module="${R_MODULE:-r/4.5.2}"

    # `module` is a shell function; a batch job or non-login shell may not have it yet
    if ! type module >/dev/null 2>&1 && [[ -f /etc/profile.d/modules.sh ]]; then
        source /etc/profile.d/modules.sh
    fi
    if ! type module >/dev/null 2>&1; then
        echo "R: the 'module' command is not available on $(hostname)" >&2
        return 1
    fi
    module load "$module" >/dev/null
    if ! command -v Rscript >/dev/null 2>&1; then
        echo "R: Rscript not on PATH after 'module load $module'; see 'module avail r'" >&2
        return 1
    fi
    export R_LIBS="${R_LIBS:-/storage1/fs1/fuhai.li/Active/di.huang/cache/R}"

    # Required: the omics workflow's R script. Optional: the network figure, which falls back to
    # matplotlib without its packages (agent/network_renderer.py).
    required=$(_r_env_packages "$project_root/tools/omic_tools/run_casestudy.R")
    optional=$(_r_env_packages "$project_root/agent/network_renderer.R")
    # shellcheck disable=SC2086  # word splitting passes one package per argument
    missing=$(Rscript --vanilla -e \
        'p <- commandArgs(TRUE); cat(p[!vapply(p, requireNamespace, logical(1), quietly = TRUE)])' \
        $(printf '%s\n' $required $optional | sort -u) 2>/dev/null) || {
        echo "R: Rscript failed to start ($(command -v Rscript))" >&2
        return 1
    }
    for pkg in $missing; do
        if [[ " $required " == *" $pkg "* ]]; then
            missing_required+=" $pkg"
        else
            missing_optional+=" $pkg"
        fi
    done

    echo "R: $(Rscript --version 2>&1 | grep -oE 'version [0-9.]+') from module $module; R_LIBS=$R_LIBS"
    [[ -n "$missing_optional" ]] &&
        echo "R: network-figure packages missing:$missing_optional (the figure falls back to matplotlib)"
    if [[ -n "$missing_required" ]]; then
        echo "R: packages required by the omics workflow are missing:$missing_required" >&2
        return 1
    fi
    echo "R: all $(wc -w <<<"$required") packages the omics workflow loads are installed"
}

if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
    _r_env_setup
    return
fi

_r_env_setup || exit 1
if (( $# )); then
    exec "$@"
fi
echo "R: check passed, but these settings end with this script. Use 'source scripts/r_env.sh'" \
     "or 'scripts/r_env.sh <command>'."
