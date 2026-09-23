"""One selection rule for every volcano, independent of disease or cell type."""
from pathlib import Path
import subprocess

import pytest

RUNNER = Path(__file__).resolve().parents[1] / "run_casestudy.R"


@pytest.mark.parametrize("up,down,want_up,want_down", [
    (22,5,10,5), (227,355,10,5), (5,26,5,10), (20,2,13,2),
    (0,30,0,15), (30,0,15,0), (2,3,2,3), (0,0,0,0),
])
def test_limits_labels_and_fills_unused_direction_quota(up,down,want_up,want_down):
    script = r'''
    source(commandArgs(TRUE)[1])
    args <- as.integer(commandArgs(TRUE)[-1])
    n_up <- args[1]; n_down <- args[2]
    up <- data.frame(Gene=sprintf("U%d",seq_len(n_up)), status=rep("Up",n_up), adj.P.Val=seq_len(n_up)/1e6)
    down <- data.frame(Gene=sprintf("D%d",seq_len(n_down)), status=rep("Down",n_down), adj.P.Val=seq_len(n_down)/1e6)
    # These gray genes outrank every colored gene but must not be labeled.
    gray <- data.frame(Gene=c("GRAY1","GRAY2"),status="NS",adj.P.Val=c(0,1e-100))
    input <- rbind(up,down,gray)
    original <- input
    got <- select_volcano_labels(input)
    stopifnot(nrow(got)==sum(args[3:4]),sum(got$status=="Up")==args[3],
              sum(got$status=="Down")==args[4],nrow(got)<=15,
              !any(got$status=="NS"),!anyDuplicated(got$Gene),identical(input,original))
    if(args[3]>0) stopifnot(setequal(got$Gene[got$status=="Up"],paste0("U",seq_len(args[3]))))
    if(args[4]>0) stopifnot(setequal(got$Gene[got$status=="Down"],paste0("D",seq_len(args[4]))))
    '''
    r=subprocess.run(["Rscript","-e",script,str(RUNNER),str(up),str(down),str(want_up),str(want_down)],
                     capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr


def test_ties_are_reproducible_and_custom_limit_is_respected():
    script=r'''
    source(commandArgs(TRUE)[1])
    input <- data.frame(Gene=c("Z","B","A","Y","D","C"),
                        status=c("Up","Up","Up","Down","Down","Down"), adj.P.Val=.001)
    got <- select_volcano_labels(input,max_labels=3,up_quota=2)
    stopifnot(identical(got$Gene,c("A","B","C")),
              identical(got$Gene,select_volcano_labels(input[6:1,],max_labels=3,up_quota=2)$Gene))
    stopifnot(inherits(try(select_volcano_labels(input,max_labels=-1),silent=TRUE),"try-error"))
    '''
    r=subprocess.run(["Rscript","-e",script,str(RUNNER)],capture_output=True,text=True,timeout=30)
    assert r.returncode==0,r.stdout+r.stderr
