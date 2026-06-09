Hard-negative MiniLM adapter workflow.
NQ order: apply_nq_mine.sh, apply_nq_train.sh, apply_nq_eval.sh, then apply_nq_aggregate.sh after evals finish.
Large order after NQ validation: apply_large_mine.sh, apply_large_train.sh, apply_large_eval.sh, then apply_large_aggregate.sh.
Mining config: ms=1,1,1, alphas=2,2,2, K2=1000, Kfinal=100.
Do not apply large jobs until NQ confirms the cache/train/eval path.
