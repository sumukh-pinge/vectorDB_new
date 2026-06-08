MiniLM adapter/no-adapter retrieval jobs.
Order: apply_prep.sh, apply_eval.sh, apply_aggregate.sh.
Eval jobs wait on INDEX_READY and adapter-on jobs also wait on ADAPTER_READY.
Do not apply until code is pushed to the GIT_REF used by the YAMLs.
