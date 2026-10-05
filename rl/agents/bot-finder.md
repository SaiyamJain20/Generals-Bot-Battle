# bot-finder report (2026-10-02)

## Wrappers added (LOCAL TESTING ONLY)
- bots/opp/ext_ksolmann.py : KSolmann/generals-bot-training, frozen opponents/3m_castle_28k (3.0M-param
  plain history transformer, competition ruleset incl. castles). Weights exported with the repo's
  scripts/export_checkpoint.py (equinox installed in /tmp only), safetensors copied to
  vendor/ext/KSolmann_generals-bot-training/agents/current_standalone/averagejoe_model.safetensors,
  runtime vendor/ext/_pylib/safetensors. The repo's standalone runtime only supports the conv-factorized
  net, so I wrote main_plain.py (same dir) = plain-transformer forward on the repo's own obs pipeline.
  1 thread, ~90 ms first move. STRONG: beats hunter 10-0; vs tune_base12g we lose 8-16 (n=24, 33% for ours).
- bots/opp/ext_ronit_graph.py : RonitNath/generals-bots GraphSearchAgent via vendor/ext/RonitNath_generals-bots/stdio_shim.py
  (old ruleset, never builds). Beats hunter 9W 1D 0L; loses 0-10 to tune_base12g. Mid-weak. Edit AGENT in the file
  for siblings (material, surround, scout, backdoor, defense, turtle, punish, swarm, sniper, greedy_city). Untested besides graph.
  jax idle thread pool (~53 threads, idle) - XLA flags set to single-thread compute.
