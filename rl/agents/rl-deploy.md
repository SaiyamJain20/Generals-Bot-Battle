# rl-deploy report
Files: rl/pyfeat.py, rl/pyinfer.py, rl/pack_student.py, rl/build_pystudent.py (assembles), rl/bots/pystudent.py (12x1, 29.8 KB),
tests: rl/test_pyfeat.py, test_pyinfer.py, test_pystudent.py, bench/prof_pyinfer.py, sweep_acc.sh.
Feature parity: 21110 positions / 20 games: planes max diff 0, scalars 1.2e-7 (float32 rounding), legal sets identical.
Technique: each feature map = one Python int of 506 32-bit lanes; conv tap = bit shift; out channel = sum(map(mul, int_w, shifted)).
ReLU/rescale/pad-mask done lane-parallel with shifts/ands (lanes offset by 2^30). Naive per-cell sum(map(mul)) est. 80-250 ms.
Params SA=8 act frac bits, stem shift 14, hidden 10; fp16 weights. Accuracy (argmax vs torch): e 99.86%, f32 99.76%, int8 per-row 97.0% (12x1) / 93% (8x2) -> use fp16.
Timing (1 core, loaded laptop, feat+infer ms): 12x1 p50 11 p99 13 (best-of-3 12.3); 8x2 p50 8-9 p99 11 (best-of-3); 16x1(random weights) p50 14-16 p99 21 (best-of-3).
pystudent.py act p50 11.6 p99 20.7 max 61.7 (loaded; spikes are contention); import 5-15 ms. Same-move rate vs torch bot 6502/6513 = 99.83%.
