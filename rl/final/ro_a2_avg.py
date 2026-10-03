"""
Code Bot entry. participant: [PARTICIPANT_ID]   bot: [BOT_NAME]

Strategy (stdlib-only heuristic, every rule simulated exactly):
  * Memory + exact opponent accounting.
    - opp_army / opp_land deltas reveal every enemy castle build and its exact price.
    - New type-5 cells locate those castles.
    - The price pins the enemy general to a Manhattan ring around the castle.
  * Enemy-general belief.
    - Spawn rules (walking distance >= 17, |room7 - ours| <= 5) narrow the candidate cells.
    - Fog sightings and castle-price rings prune them further.
    - A small learned spawn prior (logistic regression on generator samples) ranks what is left.
  * Opening: a simulated opening plan (about 24 land at turn 50), then fast expansion to about 50 land at turn 100.
  * Army cycles.
    - Greedy value-per-move gather trees feed the general's stack.
    - Half of it is launched at the most likely general, castles, or nearby threats, routed stealthily.
  * Defence.
    - The garrison scales with visible, tracked (approaching) and hidden enemy armies.
    - Defensive gathers are budgeted by the arrival time of the binding threat.
    - An adjacent stack chase-kills an attacker.
    - An exact one-turn resolver (chasing > reinforcing > smaller army) vetoes moves that lose the general.
  * Castle economy, enemy-castle sniping, and a deathtouch fortress/strike after turn 800.
  * A context modulator (56 weights, a linear map of 8 game-state features into 7 option multipliers)
    adapts the option weights to the opponent and the game phase at run time.
  * No network, no files, no threads, no subprocesses, no external randomness.
    Only the observation dict is read; no AI service is called at game time.

Embedded constants (all inline in this file, produced during the event by the participant's own scripts):
  * PARAMS (including the m_* modulator weights).
    - Numeric weights and thresholds tuned by CMA-ES / evolution strategies in self-play on the pinned engine.
    - Sparring opponents (used locally only): older versions of this bot, hand-written heuristics,
      public competition bots, and small networks behaviour-cloned from public generals.bot Marathon replays.
    - None of those bots or networks is included in or loaded by this file.
  * PRIOR_W / PRIOR_B: a 10-weight logistic spawn prior, trained on maps from the pinned engine's own generator.

Sources / attribution:
  * Rules, move order and map generator re-implemented from strakam/generals-bots@13db8f69 (MIT), the pinned engine.
  * Ideas only, no code copied:
    - EklipZ generals-bot (gather pruning, back-tracing);
    - relh Sentinel and juraj bots (chase-kill interceptor);
    - Straka & Schmid, arXiv 2507.06825;
    - statistics from the public generals.bot Marathon replays.

AI assistance: developed with Claude Code (Anthropic, Claude models) as a coding, experiment-running and
research assistant. Material parts of the code were written by it, including the tuning and training scripts
that produced the constants above. The participant directed the work, reviewed the submitted code, and can
explain and reproduce it. Standard library only.
"""
import gc
import math
import time
from collections import deque
from heapq import heappush, heappop

PARAMS = {
    # opening
    "open_div": 2,            # launch when (a-1) >= (49 - t - transit) // open_div
    "open_end": 50,
    "open_plan_s": 0.8,
    # garrison / defense
    "garrison_min": 2,
    "garrison_frac_hidden": 0.5,    # fraction of largest possible hidden stack kept home
    "garrison_cap_frac": 0.5,      # never keep more than this fraction of our army at home
    "threat_margin": 2,
    "stealth_w": 0.5,
    "x_intercept": 1,
    "x_neutral_cost": 2.0,
    "x_big_route": 0,
    "x_bonus_graded": 0,
    "x_no_pass": 0,
    "x_kill_stealth": 0,       # >0: kill runs longer than this use the stealth path (<= kill_stealth_extra longer)
    "kill_stealth_extra": 4,
    "x_phase_enemy": 0,        # 1: enemy-tile captures x pe_hi when turn%50 >= pe_hi_phi, x pe_lo when < pe_lo_phi
    "pe_hi_phi": 35,
    "pe_hi": 2.0,
    "pe_lo_phi": 15,
    "pe_lo": 0.3,
    "x_spread": 0.0,           # >0: after early_expand_until, small-stack neutral captures at turn%50 < spread_phi get early_expand_bonus x this
    "spread_phi": 8,
    "spread_max_a": 3,
    "x_early_enemy": 0.0,      # >0: enemy-tile captures also get early_expand_bonus (x this) before early_expand_until
    "snipe_range": 12,
    "snipe_margin": 5,
    "stealth_last": 3,
    "lead_w": 0.0,
    "fog_model": 0.0,
    "fog_tracks": 0,
    "gather_ratio_adj": 0.3,
    "threat_vis_range": 10,
    "threat_decay": 1.0,
    # option weights
    "w_garrison": 6.0,
    "w_garrison_urgent": 9.0,
    "hidden_stack_frac": 0.5,
    "track_min": 8,
    "track_frac": 0.12,
    "track_ttl": 30,
    "track_threat_dist": 12,
    "track_max_adv": 8,
    "closing_min": 2,
    "threat_far": 24,
    "far_decay_mult": 0.5,
    "min_def_budget": 4,
    "w_build": 7.0,
    "w_cycle": 1.2,
    "w_launch": 3.2,
    "w_scout": 0.9,
    "attack_root_front": 0,
    "scout_start": 60,
    "scout_min": 3,
    "scout_max_frac": 0.15,
    "regather_budget": 8,
    # captures
    "v_neutral": 1.0,
    "v_enemy": 2.2,
    "v_kill": 0.08,
    "v_ecastle": 8.0,
    "v_near_home": 2.0,
    "bonus_mult": 2.0,
    "bonus_window": 12,
    "bonus_lead": 0,
    "toward_w": 0.3,
    "home_r": 5,
    "v_home": 1.5,
    "w_home_fill": 2.4,
    "home_fill_start": 50,
    "small_min": 3,
    "small_frac": 0.04,
    "gen_move_pen": 0.6,
    # castles
    "castle_start": 90,
    "castle_stop": 720,
    "castle_every": 45,
    "castle_horizon": 900,
    "castle_reserve": 1,
    "castle_safe_dist": 4,
    "castle_val_min": 10.0,
    "castle_move_w": 1.0,
    "castle_price_w": 1.0,
    "castle_safety_w": 0.5,
    "castle_gather_budget": 30,
    "castle_home_n": 2,
    "castle_home_maxd": 3,
    "castle_home_w": 6.0,
    "castle_g": 0,
    "castle_g_walk": 8,
    "castle_g_walk_w": 1.5,
    "castle_stack_price": 0,
    "kill_front": 0,
    "kill_gather_budget": 18,
    "mod_on": 1,
    "m_launch_0": 0.0,
    "m_launch_1": 0.0,
    "m_launch_2": 0.0,
    "m_launch_3": 0.0,
    "m_launch_4": 0.0,
    "m_launch_5": 0.0,
    "m_launch_6": 0.0,
    "m_launch_7": 0.0,
    "m_cycle_0": 0.0,
    "m_cycle_1": 0.0,
    "m_cycle_2": 0.0,
    "m_cycle_3": 0.0,
    "m_cycle_4": 0.0,
    "m_cycle_5": 0.0,
    "m_cycle_6": 0.0,
    "m_cycle_7": 0.0,
    "m_capture_0": 0.0,
    "m_capture_1": 0.0,
    "m_capture_2": 0.0,
    "m_capture_3": 0.0,
    "m_capture_4": 0.0,
    "m_capture_5": 0.0,
    "m_capture_6": 0.0,
    "m_capture_7": 0.0,
    "m_garrison_0": 0.0,
    "m_garrison_1": 0.0,
    "m_garrison_2": 0.0,
    "m_garrison_3": 0.0,
    "m_garrison_4": 0.0,
    "m_garrison_5": 0.0,
    "m_garrison_6": 0.0,
    "m_garrison_7": 0.0,
    "m_scout_0": 0.0,
    "m_scout_1": 0.0,
    "m_scout_2": 0.0,
    "m_scout_3": 0.0,
    "m_scout_4": 0.0,
    "m_scout_5": 0.0,
    "m_scout_6": 0.0,
    "m_scout_7": 0.0,
    "m_home_0": 0.0,
    "m_home_1": 0.0,
    "m_home_2": 0.0,
    "m_home_3": 0.0,
    "m_home_4": 0.0,
    "m_home_5": 0.0,
    "m_home_6": 0.0,
    "m_home_7": 0.0,
    "m_kill_0": 0.0,
    "m_kill_1": 0.0,
    "m_kill_2": 0.0,
    "m_kill_3": 0.0,
    "m_kill_4": 0.0,
    "m_kill_5": 0.0,
    "m_kill_6": 0.0,
    "m_kill_7": 0.0,
    "attack_ratio": 0.0,
    "castle_front_w": 0.0,
    "castle_keep": 0,          # 1: castles only give half to gathers/captures, none when a visible enemy is near
    "castle_guard_r": 4,
    "ring_r": 2,
    "ring_w": 0.0,
    "early_expand_until": 100,
    "early_expand_bonus": 6.0,
    # army cycle
    "gather_budget": 14,
    "gather_default_budget": 40,
    "launch_max": 40,
    "min_stack": 6,
    "feed_min": 6,
    "kill_margin": 2,
    "x_sweep": 1,               # kill check collects own army along the path (Sentinel V8 collection)
    "sweep_cands": 10,
    "x_stage": 0,               # hold + regather a strike near the known general until it can kill
    "stage_d": 8,
    "stage_max": 3,
    "stage_budget": 10,
    "intercept_dist": 4,
    "belief_enemy_w": 0.7,
    "belief_explore_w": 1.0,
    "belief_prior_w": 3.0,
    "attack_min_army": 60,
    # time
    "soft_budget_ms": 45,
    "first_budget_ms": 3000,
    # endgame
    "fortress_turn": 740,
    "x_dt_guard": 1,           # deathtouch-era chase / block defence of the general (dt_guard)
    "dt_guard_turn": 790,
    "dt_stage_turn": 760,
    "aggro_turn": 1000,
    "expand_toward_w": 0.15,
}

P_SNIPE_REMAINDER = 6  # typical army left on a freshly built enemy castle

# learned spawn prior (logistic regression on generator samples; see learn/train_prior.py)
PRIOR_W = [-0.2466, 0.1374, 0.0267, 0.6231, 1.4779, 0.2212, -0.1133, 0.0064, -0.049, 13.246]
PRIOR_B = -5.1142


PARAMS.update({'open_div': 2, 'open_end': 50, 'open_plan_s': 1.5, 'garrison_min': 2, 'garrison_frac_hidden': 0.4328, 'garrison_cap_frac': 0.6529, 'threat_margin': 2, 'threat_vis_range': 8, 'threat_decay': 0.9542, 'w_garrison': 4.3648, 'w_garrison_urgent': 9.0, 'hidden_stack_frac': 0.4606, 'track_min': 8, 'track_frac': 0.12, 'track_ttl': 30, 'track_threat_dist': 5, 'w_build': 9.8385, 'w_cycle': 2.4314, 'w_launch': 4.7703, 'w_scout': 1.0867, 'attack_root_front': 0, 'scout_start': 60, 'scout_min': 3, 'scout_max_frac': 0.3596, 'regather_budget': 8, 'v_neutral': 0.3017, 'v_enemy': 2.4779, 'v_kill': 0.1549, 'v_ecastle': 8.0, 'v_near_home': 2.1846, 'bonus_mult': 1.9644, 'bonus_window': 10, 'bonus_lead': 0, 'toward_w': 0.3, 'home_r': 6, 'v_home': 1.5, 'w_home_fill': 3.0349, 'home_fill_start': 50, 'small_min': 3, 'small_frac': 0.1288, 'gen_move_pen': 0.6, 'castle_start': 244, 'castle_stop': 434, 'castle_every': 49, 'castle_horizon': 900, 'castle_reserve': 1, 'castle_safe_dist': 5, 'castle_val_min': 10.0, 'castle_move_w': 1.0, 'castle_price_w': 0.9628, 'castle_safety_w': 0.0266, 'castle_gather_budget': 9, 'castle_home_n': 2, 'castle_home_maxd': 3, 'castle_home_w': 6.0, 'gather_budget': 21, 'gather_default_budget': 40, 'launch_max': 69, 'min_stack': 4, 'feed_min': 4, 'kill_margin': 2, 'intercept_dist': 1, 'belief_enemy_w': 0.8921, 'belief_explore_w': 1.0, 'belief_prior_w': 3.0, 'attack_min_army': 50, 'soft_budget_ms': 45, 'first_budget_ms': 3000, 'fortress_turn': 696, 'dt_stage_turn': 760, 'aggro_turn': 1000, 'expand_toward_w': 0.15})

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
PASS = [1, 0, 0, 0, 0]
INF = 10 ** 9
DEBUG = False  # arena sets True to surface exceptions


def _now():
    return time.perf_counter()


# ======================================================================
# RO (Residual Option) policy hook. RO_W None => exactly the heuristic.
# z_i = alpha*s_i + b[lab] + u[lab].phi + v.psi_i + Uh[lab].tanh(W1 phi + b1)
# ======================================================================
import random as _random

RO_W = {"alpha":1.581565,"alpha0":2.0,"u":[[0.071967,0.462711,0.64706,-0.547197,-0.590877,0.309825,-0.43433,0.201128,-0.25656,0.110386,0.034571,-0.086977,0.2306,-0.75974,0.024285,0.037967,-0.246007,-0.045191,-0.218161,0.864288,0.23997,-0.158968,-0.437143,0.205734],[-0.097224,0.208363,0.131348,0.334452,0.279786,0.058971,-0.139309,-0.006631,0.478156,-0.514221,-0.064307,0.374311,-0.125738,-0.083048,-0.053215,-0.232984,0.639176,-0.239201,0.242342,0.062486,0.145075,0.330384,0.360495,-0.032649],[-0.405687,0.730438,0.603371,0.456644,0.395539,-0.317113,-0.194431,-0.246774,-0.111517,-0.318189,-0.478342,0.329064,-0.62118,-0.232504,-0.067655,-0.304406,0.263864,-0.536244,0.606889,0.24689,-0.384019,-0.666377,0.058524,-0.381487],[0.073682,-0.151275,0.025793,0.409353,0.425436,-0.007854,0.425117,0.557045,0.133416,0.066244,0.429767,-0.162039,-0.183909,-0.30924,-0.203236,-0.483354,-0.261625,0.394247,-0.241058,-0.783285,0.116014,-0.000507,0.18936,-0.079336],[-0.127128,0.465637,0.350651,-0.225757,-0.292796,0.162491,-0.387695,-0.146776,0.073882,-0.01636,0.15917,0.012613,-0.203657,0.627254,-0.080489,0.243933,-0.381794,0.030478,0.030262,-0.277649,-0.165482,-0.529926,-0.473802,0.025041],[-0.21498,-0.676399,-0.628104,0.437129,0.555828,-0.395437,0.702375,0.183368,-0.185714,0.3395,-0.59355,-0.334835,-0.366287,0.42864,0.42864,0.271024,-0.29515,-0.471867,0.371937,-0.515592,-0.267827,0.111887,-0.213791,-0.499098],[0.146689,-0.285005,-0.645785,0.228911,0.258016,-0.228112,0.13401,-0.371942,0.171778,-0.191495,-0.068096,0.208712,0.293253,0.0,-0.01496,0.159954,0.421706,0.059612,-0.017874,-0.393328,-0.057725,0.179966,0.423293,0.081653],[-0.077605,-0.312663,-0.164152,-0.002737,0.037949,0.064511,0.168878,-0.18174,-0.215091,0.225762,-0.059316,-0.110803,-0.223651,0.0,-0.00551,-0.34069,0.283671,0.208733,0.186495,-0.270533,-0.296485,0.146405,-0.10126,-0.07309],[-0.163238,0.401372,1.207956,-0.039486,-0.110651,-0.112956,0.564958,0.023034,-0.475271,0.126089,0.192074,-0.810351,-0.093596,0.069027,0.069027,0.694203,-0.063055,-0.013715,-0.175593,0.525456,0.341484,-0.072183,-0.144374,0.200776]],"v":[0.134396,-0.228008,-0.171024,0.061853,0.291858,-0.32344,0.857542,0.078585],"b":[-0.278003,0.145075,0.43899,0.162039,-0.043525,0.428639,-0.01496,0.005511,-0.069027],"W1":[[0.508784,-1.02806,-0.598311,0.133986,0.648815,-0.549832,0.897714,-0.180241,0.500657,-0.010893,0.521708,-0.041075,0.113515,-0.020731,-0.203512,-0.289713,0.017915,0.256114,0.090149,-0.911091,-0.129586,0.408129,1.527826,0.163691],[0.028162,-0.277629,-0.065421,0.175148,0.33877,-0.007294,0.774016,-0.434918,-0.911252,0.273014,-0.292666,-0.234904,-0.710111,0.367605,0.182273,-0.005899,-0.637319,0.101553,0.109594,-0.005107,-0.460337,0.173514,-0.193793,0.157208],[-0.017853,-0.01683,-0.371809,0.168965,0.025416,-0.218606,0.620788,-0.451745,-0.194033,0.177708,-0.144913,0.109419,-0.552508,0.639137,0.154252,0.646997,0.254313,0.113063,0.490832,-0.072408,0.188916,0.072352,-0.454281,-0.280033],[-0.082001,0.785189,0.667911,-0.741008,-0.47459,0.439866,-0.91297,0.032719,0.284735,-0.517926,0.350233,0.14821,0.023906,-0.047371,-0.069565,-0.369436,-0.050236,0.482949,0.033556,0.310039,-0.027976,-0.596773,0.034311,0.173747],[0.047677,-0.133035,-0.318489,0.603332,0.394914,0.173878,1.220337,0.188632,-0.203021,0.566273,0.114925,-0.252378,-0.075747,0.458545,-0.183843,0.637708,0.088736,-0.048669,0.399445,-0.606071,-0.117499,0.195634,0.502614,0.022475],[-0.012339,0.61178,0.648581,0.311075,-0.563479,0.170255,1.023221,-0.092175,-0.600691,0.616493,-0.080321,-0.370406,-0.620311,0.234668,0.176067,0.172731,-0.365515,-0.761474,0.526737,-0.019498,-0.21811,-0.108113,-0.357743,0.025293],[0.123605,-0.644842,-0.77254,0.614531,0.620627,0.115961,0.501898,-0.69513,0.372296,0.381974,0.106944,0.040329,-0.190683,-0.121151,0.020163,-0.211293,0.291716,0.007968,0.16581,-0.399195,0.149754,0.561371,0.681959,-0.092738],[0.484521,-1.241049,-1.017631,0.605705,1.048926,-0.433595,1.045393,0.189283,0.241125,0.586183,-0.087403,-0.285447,-0.136669,0.188653,0.1841,0.860369,0.153541,-0.280638,0.333641,-0.253544,-0.153812,0.336269,0.312875,-0.093878],[0.010665,1.210942,1.032599,-0.535752,-0.216139,0.350595,-0.896918,0.158321,0.15443,-0.46581,-0.003077,0.660115,-0.541908,-0.1131,-0.244958,-0.675914,-0.426552,0.43484,-0.277236,-0.215839,0.223084,-0.758197,-0.715778,0.11927],[0.106243,0.147459,0.772242,-0.270828,0.01766,0.22738,0.142396,0.103344,-0.3108,0.699183,-0.015353,-0.224881,-0.291895,-0.060465,-0.234195,-0.575809,-0.101461,0.919948,-0.274158,0.197511,-0.388147,-0.240417,-0.666974,0.252843],[-0.027455,-0.842306,-0.720711,-0.632563,-0.399479,0.28454,-0.030652,-0.095146,0.032345,0.08298,-0.203219,-0.392728,-0.75245,1.012117,0.573508,0.045889,0.478985,-0.007811,-0.152953,-0.055297,0.190674,-0.112736,0.145496,-0.01543],[0.046113,0.311409,0.078118,-0.074479,-0.069681,-0.208262,0.628028,0.443788,-0.16534,0.524727,-0.44667,-0.345641,-0.46847,0.522914,0.183228,0.385206,-0.599527,-0.09053,0.184315,0.388479,-0.061168,-0.405057,-0.107058,-0.146522],[-0.424698,0.146057,0.326123,-0.145335,-0.059262,-0.118037,-1.101669,0.133235,0.49791,-0.193385,-0.016763,0.256136,0.456758,-0.425776,-0.068345,-0.734361,0.208625,0.222791,-0.320201,-0.122871,0.321367,-0.66667,0.338523,0.120119],[0.350495,0.666096,1.083865,-0.016868,-0.153914,0.066411,-1.024102,-0.156696,0.470365,-0.564433,0.336708,0.159477,0.461248,0.24787,-0.183905,-0.147761,-0.10221,0.228014,0.102416,0.704036,-0.119243,-0.368039,0.32406,0.067634],[-0.555779,-0.517334,-0.150336,-0.720032,-0.34619,-0.001363,0.235875,0.123177,-0.378044,0.204036,-0.466964,-0.002291,0.073249,-0.506312,0.362808,0.305723,-0.398402,-0.062558,-0.182253,-0.264028,-0.257212,-0.477976,-0.104421,-0.110233],[-0.273008,-0.785056,-1.140069,0.127047,0.132968,-0.076278,0.135473,-0.485876,-0.13554,0.218442,-0.442783,-0.543527,-0.016939,0.0741,0.541709,-0.301524,-0.222965,0.289404,-0.481649,-0.422522,0.007998,0.162671,-0.223436,-0.25569]],"b1":[0.006314,0.312191,0.148752,-0.133063,0.122349,0.038391,0.171448,0.165278,0.071601,-0.108382,-0.182201,0.071437,-0.044431,-0.114248,-0.110122,0.056907],"Uh":[[-0.306811,-0.18326,-0.148238,0.24391,-0.357058,-0.054622,-0.191192,-0.323432,0.223337,-0.080438,0.06245,-0.097843,0.111571,0.102608,0.097736,-0.133503],[-0.120719,-0.332634,-0.069734,-0.034846,-0.006877,-0.133447,0.205841,0.080624,-0.066826,-0.290394,0.011156,-0.415905,0.21241,0.239177,-0.441946,-0.165706],[0.03262,0.346087,0.049323,0.005991,0.204509,0.033724,0.352831,0.303372,0.184614,-0.35479,-0.366055,-0.469167,0.249425,-0.261704,-0.504995,-0.143888],[0.431018,0.016137,-0.241954,0.223291,0.00611,0.02028,-0.044418,-0.432546,0.509408,0.29822,-0.102281,-0.144139,0.263776,-0.039846,-0.228041,-0.048675],[-0.274294,0.239598,0.11238,0.065012,0.028018,0.386147,-0.097698,-0.293709,0.152328,0.21898,-0.004353,0.184136,-0.324619,-0.033797,0.101858,-0.039481],[-0.030935,0.321394,0.341823,-0.704399,0.734285,0.427244,0.195646,1.334263,-0.708893,-0.12832,0.35597,0.78224,-0.489544,-0.448323,0.283291,0.330406],[0.187381,-0.066543,-0.087657,-0.007707,-0.011094,-0.435997,0.018379,0.112767,-0.135951,-0.146108,-0.20781,-0.141222,0.102481,0.182577,0.015934,0.015774],[0.428095,0.19682,0.235214,-0.15824,0.163771,0.177457,0.175143,0.187549,-0.110722,0.389203,0.012115,-0.063744,-0.103634,-0.266275,0.063662,0.198585],[-0.348579,0.511805,0.240244,0.135386,0.273273,0.951875,-0.519455,-0.089473,-0.417268,0.498078,-0.132445,0.779043,-0.492442,0.263808,0.039033,0.124225]]}  # trained RO-PPO weights
RO_SAMPLE = False    # sample from softmax(z) instead of argmax
RO_RECORD = None     # list => append one record per decision (training only)
RO_RNG = _random.Random(0)
RO_LABELS = ("capture", "garrison", "build", "scout", "home", "launch",
             "cyc_attack", "cyc_castle", "other")
_RO_LIDX = {n: i for i, n in enumerate(RO_LABELS)}
RO_NPHI = 24
RO_NPSI = 8
_RO_CACHE = [None, None]


def _ro_lab(label):
    i = _RO_LIDX.get(label)
    if i is not None:
        return i
    if label == "cyc_castle_g":
        return _RO_LIDX["cyc_castle"]
    return len(RO_LABELS) - 1


def _ro_clip(x, lo=-1.0, hi=1.0):
    return lo if x < lo else (hi if x > hi else x)


def _ro_phi(b, nopt, need_g):
    t = b.turn
    A, g = b.A, b.general
    Ag = A[g] if g >= 0 else 0
    lg = math.log
    cyc = b.cyc
    mode = cyc.get("mode") if cyc else None
    ret = 1.0 if b.threat_eta >= INF else _ro_clip(min(b.threat_eta, 20) / 10.0 - 1.0)
    trk = max((tr[1] for tr in b.tracks), default=0)
    hidden = b.opp_army - b.vis_enemy_army - max(0, b.opp_land - b.vis_enemy_cells - 1)
    ph = 2.0 * math.pi * (t % 50) / 50.0
    ll = getattr(b, "ro_last_launch", -1000)
    return [
        _ro_clip(t / 600.0 - 1.0),
        math.tanh(lg(max(1, b.my_army) / max(1, b.opp_army))),
        math.tanh(lg(max(1, b.my_land) / max(1, b.opp_land))),
        _ro_clip(math.log1p(b.my_army) / 4.0 - 1.0),
        _ro_clip(math.log1p(b.opp_army) / 4.0 - 1.0),
        math.tanh(need_g / (Ag + 1.0) - 1.0),
        _ro_clip(math.log1p(Ag) / 4.0 - 1.0),
        ret,
        math.tanh(trk / (Ag + 1.0) - 1.0),
        min(len(b.enemy_castles), 5) / 2.5 - 1.0,
        min(len(b.my_castles), 5) / 2.5 - 1.0,
        1.0 if b.egen >= 0 else -1.0,
        _ro_clip(min(b.fog_distance(), 20) / 10.0 - 1.0),
        -1.0 if mode is None else (0.0 if mode == "gather" else 1.0),
        1.0 if (cyc and cyc.get("purpose") == "attack") else -1.0,
        _ro_clip(min(t - ll, 100) / 50.0 - 1.0),
        math.sin(ph),
        math.cos(ph),
        2.0 * b.enemy_gather_ratio() - 1.0,
        nopt / 3.0 - 1.0,
        1.0 if b.defence < need_g else -1.0,
        _ro_clip(math.log1p(max(0, hidden)) / 5.0 - 1.0),
        _ro_clip(math.log1p(b.vis_enemy_army) / 5.0 - 1.0),
        2.0 * b.my_land / max(1, b.n) - 1.0,
    ]


def _ro_psi(b, opt):
    a = opt[1]
    out = [0.0] * RO_NPSI
    try:
        W = b.W
        A, O = b.A, b.O
        if a and a[0] in (0, 2):
            src = a[1] * W + a[2]
            out[0] = _ro_clip(math.log1p(A[src]) / 5.0 - 1.0)
            out[5] = 1.0 if src == b.general else -1.0
            if a[0] == 0:
                dr, dc = DIRS[a[3]]
                dst = (a[1] + dr) * W + a[2] + dc
                out[1] = _ro_clip(min(b.dist_g[dst], 20) / 10.0 - 1.0)
                o = O[dst]
                out[2] = 1.0 if o == 1 else -1.0
                out[3] = 1.0 if o == 2 else -1.0
                out[4] = 1.0 if o == 0 else -1.0
                if o != 1:
                    out[7] = math.tanh(math.log(max(1, A[src]) / (A[dst] + 1.0)))
            cyc = b.cyc
            if opt[2].startswith("cyc") and cyc and cyc.get("mode") == "gather":
                out[6] = _ro_clip(2.0 * cyc.get("moves", 0) / max(1, cyc.get("budget", 1)) - 1.0)
    except Exception:
        pass
    return out


def _ro_prep(W):
    """Cache numeric structures for a weights dict."""
    if _RO_CACHE[0] is W:
        return _RO_CACHE[1]
    c = {"alpha": float(W["alpha"]), "alpha0": float(W["alpha0"]), "u": W["u"], "v": W["v"],
         "b": W["b"], "W1": W.get("W1") or [], "b1": W.get("b1") or [], "Uh": W.get("Uh") or []}
    _RO_CACHE[0], _RO_CACHE[1] = W, c
    return c


def _ro_logits(c, phi, psis, labs, scores):
    h = None
    if c["W1"]:
        tanh = math.tanh
        h = [tanh(bb + sum(w * x for w, x in zip(row, phi))) for row, bb in zip(c["W1"], c["b1"])]
    v = c["v"]
    al = c["alpha"]
    zs = []
    for s, lab, psi in zip(scores, labs, psis):
        z = al * s + c["b"][lab] + sum(w * x for w, x in zip(c["u"][lab], phi)) \
            + sum(w * x for w, x in zip(v, psi))
        if h is not None:
            z += sum(w * x for w, x in zip(c["Uh"][lab], h))
        zs.append(z)
    return zs


def _ro_logsm(zs):
    m = max(zs)
    lse = m + math.log(sum(math.exp(z - m) for z in zs))
    return [z - lse for z in zs]


def _ro_pick(b, options, need_g):
    """Index of the chosen option in `options` (sorted by heuristic score)."""
    c = _ro_prep(RO_W)
    scores = [o[0] for o in options]
    labs = [_ro_lab(o[2]) for o in options]
    phi = _ro_phi(b, len(options), need_g)
    psis = [_ro_psi(b, o) for o in options]
    lp = _ro_logsm(_ro_logits(c, phi, psis, labs, scores))
    if RO_SAMPLE:
        r = RO_RNG.random()
        acc = 0.0
        k = len(lp) - 1
        for i, x in enumerate(lp):
            acc += math.exp(x)
            if r < acc:
                k = i
                break
    else:
        k = max(range(len(lp)), key=lambda i: lp[i])
    if options[k][2] == "launch":
        b.ro_last_launch = b.turn
    if RO_RECORD is not None:
        lr = _ro_logsm([c["alpha0"] * s for s in scores])
        RO_RECORD.append({"phi": phi, "psi": psis, "s": scores, "lab": labs, "a": k,
                          "logp": lp[k], "logp_ref": lr[k], "turn": b.turn,
                          "cnt": (b.my_army, b.opp_army, b.my_land, b.opp_land)})
    return k


class Bot:
    def __init__(self, obs):
        self.H = H = int(obs["height"])
        self.W = W = int(obs["width"])
        self.pid = int(obs.get("player_id", 0) or 0)
        self.n = n = H * W
        nb = []
        for i in range(n):
            r, c = divmod(i, W)
            lst = []
            for d, (dr, dc) in enumerate(DIRS):
                rr, cc = r + dr, c + dc
                if 0 <= rr < H and 0 <= cc < W:
                    lst.append((rr * W + cc, d))
            nb.append(lst)
        self.nb = nb
        self.mountain = [False] * n
        self.castle = [False] * n          # known castle cells (any owner)
        self.last_owner = [0] * n          # 0 unknown/neutral, 1 me, 2 enemy
        self.last_army = [0] * n
        self.last_seen = [-1] * n
        self.first_enemy_seen = [-1] * n
        self.ever_visible = [False] * n
        self.general = -1
        self.egen = -1
        self.prev = None                   # previous turn scalars and structures
        self.last_action = PASS
        self.enemy_castles = set()
        self.my_castles = set()
        self.enemy_struct = 1              # enemy general + castles
        self.enemy_builds = []             # (turn, cell, price)
        self.ecastle_built = {}            # enemy castle cell -> build turn (from accounting)
        self.rings = []                    # (cell, exact_d or None for >=7)
        self.plan = None
        self.cyc = None
        self.open_cfg = {"div": 2.0, "w_free": 1.0, "w_dist": 0.3, "w_toward": 0.15, "late": 44}
        self.open_dc = None
        self.t_start = time.perf_counter()
        self.budget_s = PARAMS["soft_budget_ms"] / 1000.0
        self.last_label = ""
        self.pending_stack = None
        self.threat_eta = INF
        self.mod = {g: 1.0 for g in self.MOD_GROUPS}
        self.tracks = []
        self.prevT = self.prevO = self.prevA = None
        self.prev_opp_land = 0
        self.built_this_turn = False
        self.last_enemy_kind = "fog"
        self.enemy_kind_hist = []
        self.fq = [0] * 64
        self.fog_stack = 0
        self.last_purpose = None
        self.need_g = 2
        self.turn = -1
        self.err = 0
        self.cands = None
        self.cand_dist = {}

    # ------------------------------------------------------------------ utils
    def late(self, frac=1.0):
        """True once the soft per-move time budget is (frac) used."""
        return time.perf_counter() > self.t_start + frac * self.budget_s

    def manh(self, a, b):
        W = self.W
        return abs(a // W - b // W) + abs(a % W - b % W)

    def bfs(self, sources, passable=None, limit=INF):
        dist = [INF] * self.n
        dq = deque()
        for s in sources:
            dist[s] = 0
            dq.append(s)
        nb = self.nb
        pas = self.pas if passable is None else passable
        while dq:
            i = dq.popleft()
            nd = dist[i] + 1
            if nd > limit:
                continue
            for j, _ in nb[i]:
                if pas[j] and dist[j] > nd:
                    dist[j] = nd
                    dq.append(j)
        return dist

    def dir_to(self, i, j):
        for k, d in self.nb[i]:
            if k == j:
                return d
        return -1

    def mv(self, i, j, split=0):
        d = self.dir_to(i, j)
        if d < 0:
            return None
        return [0, i // self.W, i % self.W, d, split]

    # --------------------------------------------------------------- parsing
    def parse(self, obs):
        H, W = self.H, self.W

        def flat(g):
            if g is None:
                return [0] * (H * W)
            if hasattr(g, "tolist"):
                g = g.tolist()
            if len(g) == H * W and not isinstance(g[0], (list, tuple)):
                return [int(v) for v in g]
            out = []
            for row in g:
                out.extend(int(v) for v in row)
            return out

        return flat(obs["type"]), flat(obs["owner"]), flat(obs["army"])

    # ------------------------------------------------------------- turn zero
    def setup(self, T):
        n = self.n
        for i in range(n):
            if T[i] in (2, 5):
                self.mountain[i] = True
        self.pas = [not m for m in self.mountain]
        g = self.general
        self.dist_g = self.bfs([g])
        # room7 for every passable cell (cells reachable within 7, excluding self)
        room = [0] * n
        for i in range(n):
            if self.pas[i]:
                d = self.bfs([i], limit=7)
                room[i] = sum(1 for v in d if v <= 7) - 1
        self.room = room
        mine = room[g]
        far = [i for i in range(n) if self.pas[i] and self.dist_g[i] < INF and self.dist_g[i] >= 17]
        cands = [i for i in far if abs(room[i] - mine) <= 5]
        if not cands:
            if far:
                best = min(abs(room[i] - mine) for i in far)
                cands = [i for i in far if abs(room[i] - mine) == best]
            else:
                mx = max(v for v in self.dist_g if v < INF)
                cands = [i for i in range(n) if self.dist_g[i] == mx]
        self.cands0 = list(cands)
        self.cands = set(cands)
        for c in cands:
            self.cand_dist[c] = self.bfs([c])
        self.prior = {}
        try:
            feats = {c: self.cand_features(c) for c in cands}
            for c, f in feats.items():
                self.prior[c] = sum(w * x for w, x in zip(PRIOR_W, f)) + PRIOR_B
        except Exception:
            self.prior = {c: 0.0 for c in cands}
        try:
            self.plan_opening()
        except Exception:
            if DEBUG:
                raise
            self.open_cfg = {"div": 2.0, "w_free": 1.0, "w_dist": 0.3, "w_toward": 0.15, "late": 44}

    def cand_features(self, c):
        """Static features of a spawn candidate (relative to our general)."""
        H, W = self.H, self.W
        g = self.general
        dc = self.cand_dist[c]
        reach = [v for v in dc if v < INF]
        span = max(reach)
        ecc_g = max(v for v in self.dist_g if v < INF)
        r, col = divmod(c, W)
        edge = min(r, H - 1 - r, col, W - 1 - col)
        nb_open = sum(1 for j, _ in self.nb[c] if self.pas[j])
        gap = abs(self.room[c] - self.room[g])
        ncand = len(self.cands0)
        return [self.dist_g[c] / 30.0, self.manh(c, g) / 30.0, gap / 5.0, self.room[c] / 100.0,
                span / 40.0, (span - ecc_g) / 10.0, edge / 5.0, nb_open / 4.0,
                (self.room[c] - self.room[g]) / 5.0, 1.0 / max(1, ncand)]

    # ------------------------------------------------------------ bookkeeping
    def update(self, obs, T, O, A):
        t = self.turn
        n = self.n
        my_struct = 0
        my_castles = set()
        vis_enemy_army = 0
        vis_enemy_cells = 0
        vis_enemy_castles = set()
        new_struct_fog = []
        for i in range(n):
            ti = T[i]
            if ti == 0 or ti == 5:
                if ti == 5 and not self.mountain[i] and not self.castle[i]:
                    new_struct_fog.append(i)
                continue
            self.ever_visible[i] = True
            self.last_seen[i] = t
            oi = O[i]
            self.last_owner[i] = oi
            self.last_army[i] = A[i]
            if ti == 3:
                self.castle[i] = True
            if oi == 1:
                if ti == 3 or ti == 4:
                    my_struct += 1
                    if ti == 3:
                        my_castles.add(i)
            elif oi == 2:
                vis_enemy_army += A[i]
                vis_enemy_cells += 1
                if self.first_enemy_seen[i] < 0:
                    self.first_enemy_seen[i] = t
                if ti == 4:
                    self.egen = i
                if ti == 3:
                    vis_enemy_castles.add(i)
            if self.cands and i in self.cands and not (ti == 4 and oi == 2):
                self.cands.discard(i)
        self.pO, self.pA = getattr(self, "O", None), getattr(self, "A", None)
        self.T, self.O, self.A = T, O, A
        self.my_castles = my_castles
        self.vis_enemy_army = vis_enemy_army
        self.vis_enemy_cells = vis_enemy_cells
        my_army = int(obs["my_army"])
        my_land = int(obs["my_land"])
        opp_army = int(obs["opp_army"])
        opp_land = int(obs["opp_land"])
        self.my_army, self.my_land, self.opp_army, self.opp_land = my_army, my_land, opp_army, opp_land

        # castles that changed hands (visible)
        for i in vis_enemy_castles:
            self.enemy_castles.add(i)
        for i in list(self.enemy_castles):
            if O[i] == 1:
                self.enemy_castles.discard(i)
        prev = self.prev
        if prev is not None and prev["turn"] == t - 1:
            lost = prev["my_castles"] - my_castles
            for i in lost:  # we lost a castle -> it is the enemy's now
                self.enemy_castles.add(i)
            gained_from_enemy = {i for i in my_castles - prev["my_castles"] if prev["enemy_castles"] and i in prev["enemy_castles"]}
            # our growth / build bookkeeping
            g2 = 1 if t % 2 == 0 else 0
            g50 = 1 if t % 50 == 0 else 0
            our_build = 0
            la = self.last_action
            if la[0] == 2:
                bi = la[1] * self.W + la[2]
                if bi in my_castles and bi not in prev["my_castles"]:
                    our_build = prev["build_cost_cache"].get(bi, self.price_for(prev["my_structs"], bi))
            X = prev["my_army"] - our_build + g2 * my_struct + g50 * my_land - my_army
            R = opp_army - prev["opp_army"] + X - g50 * opp_land
            S_prev = 1 + len(prev["enemy_castles"])
            S_now = S_prev + len(lost) - len(gained_from_enemy)
            built = 0
            self.built_this_turn = False
            if g2:
                if R != S_now:
                    built = S_now + 1 - R
            else:
                built = -R
            if built >= 35:
                self.built_this_turn = True
                cell = self.locate_new_castle(new_struct_fog, vis_enemy_castles, prev)
                if cell >= 0:
                    self.castle[cell] = True
                    self.register_enemy_build(cell, built, prev["enemy_castles"])
                    self.enemy_castles.add(cell)
            elif built != 0:
                self.acct_desync = t
        for i in new_struct_fog:  # any unexplained new structure is a castle too
            self.castle[i] = True
            self.enemy_castles.add(i)
        self.enemy_struct = 1 + len(self.enemy_castles)
        my_structs = [i for i in my_castles] + ([self.general] if self.general >= 0 else [])
        self.my_structs = my_structs
        self.prev = {"turn": t, "my_army": my_army, "opp_army": opp_army, "my_castles": set(my_castles),
                     "enemy_castles": set(self.enemy_castles), "my_structs": list(my_structs),
                     "build_cost_cache": {}}
        # belief pruning from land count / first sightings
        self.prune_candidates()
        try:
            self.classify_enemy_move()
        except Exception:
            if DEBUG:
                raise
            self.last_enemy_kind = "fog"
        self.prevT, self.prevO, self.prevA = T, O, A
        self.prev_opp_land = opp_land
        self.update_tracks()

    def classify_enemy_move(self):
        """One enemy action per tick: 'vis' (seen), 'build', 'fogcap' (fog capture) or
        'foggather' (fog merge). Maintains a histogram of fog enemy tile armies and the
        army the enemy has concentrated in fog (EklipZ's fog gather queue idea)."""
        t = self.turn
        T, O, A = self.T, self.O, self.A
        pT, pO, pA = self.prevT, self.prevO, self.prevA
        kind = "foggather"
        if pT is None:
            self.last_enemy_kind = "fog"
            return
        la = self.last_action
        my_dst = -1
        if la and la[0] == 0:
            dr, dc = DIRS[la[3]]
            my_dst = (la[1] + dr) * self.W + la[2] + dc
        took = 0
        g2 = t % 2 == 0
        g50 = t % 50 == 0
        if self.built_this_turn:
            kind = "build"
        else:
            for i in range(self.n):
                if T[i] in (0, 5) or pT[i] in (0, 5):
                    continue
                if pO[i] == 2 and O[i] == 1:
                    took += 1
                if O[i] == 2:
                    if pO[i] != 2 and i != my_dst:
                        kind = "vis"
                        break
                    if pO[i] == 2:
                        grow = (1 if g2 and T[i] in (3, 4) else 0) + (1 if g50 else 0)
                        if A[i] < pA[i] + grow and i != my_dst:
                            kind = "vis"
                            break
            if kind != "vis":
                dland = self.opp_land - self.prev_opp_land + took
                kind = "fogcap" if dland >= 1 else "foggather"
        self.last_enemy_kind = kind
        hist = self.enemy_kind_hist
        hist.append(kind)
        if len(hist) > 50:
            del hist[0]
        fq = self.fq
        if g50:
            fq.insert(1, 0)
            fq.pop()
            fq[1] += fq[0]
            fq[0] = 0
        if kind == "foggather":
            v = 0
            for k in range(len(fq) - 1, 1, -1):
                if fq[k]:
                    v = k
                    break
            if v >= 2:
                fq[v] -= 1
                fq[1] += 1
                self.fog_stack += v - 1
        elif kind == "fogcap":
            if fq[2]:
                fq[2] -= 1
                fq[1] += 2
            else:
                fq[1] += 1
                self.fog_stack = max(0, self.fog_stack - 1)
        gen_vis = self.egen >= 0 and T[self.egen] == 4
        fog_castles = sum(1 for c in self.enemy_castles if T[c] in (0, 5))
        nfog = max(0, self.opp_land - self.vis_enemy_cells - (0 if gen_vis else 1) - fog_castles)
        cur = sum(fq)
        if cur < nfog:
            fq[1] += nfog - cur
        elif cur > nfog:
            extra = cur - nfog
            for k in range(1, len(fq)):
                take = min(fq[k], extra)
                fq[k] -= take
                extra -= take
                if extra <= 0:
                    break
        hidden = self.opp_army - self.vis_enemy_army - max(0, self.opp_land - self.vis_enemy_cells)
        self.fog_stack = min(self.fog_stack, max(0, hidden))

    def fog_risk(self, in_turns=0):
        """Army that could reach us from the fog: concentrated stack + next pops."""
        r = self.fog_stack
        fq = self.fq
        left = in_turns
        for v in range(len(fq) - 1, 1, -1):
            if left <= 0:
                break
            k = min(fq[v], left)
            r += k * (v - 1)
            left -= k
        return r

    def enemy_gather_ratio(self):
        h = self.enemy_kind_hist
        g = sum(1 for k in h if k == "foggather")
        c = sum(1 for k in h if k == "fogcap")
        return g / (g + c) if g + c else 0.5

    def update_tracks(self):
        """Track big enemy stacks through fog (position, army, last seen turn)."""
        t = self.turn
        A, O, T = self.A, self.O, self.T
        big = max(PARAMS["track_min"], int(PARAMS["track_frac"] * self.opp_army))
        seen = [i for i in range(self.n) if O[i] == 2 and A[i] >= big]
        tracks = self.tracks
        used = set()
        for e in sorted(seen, key=lambda i: -A[i]):
            best, bd = None, None
            for k, tr in enumerate(tracks):
                if k in used:
                    continue
                d = self.manh(e, tr[0])
                if d <= (t - tr[2]) + 1 and (bd is None or d < bd):
                    bd, best = d, k
            if best is None:
                tracks.append([e, A[e], t, 0, self.dist_g[e], 0])
                used.add(len(tracks) - 1)
                if A[e] >= 0.87 * (self.fog_stack + 1):
                    self.fog_stack = 0
                else:
                    self.fog_stack = max(0, self.fog_stack - A[e])
            else:
                old = tracks[best]
                nd = self.dist_g[e]
                closing = old[5] + 1 if nd < old[4] else (old[5] if nd == old[4] else max(0, old[5] - 2))
                tracks[best] = [e, A[e], t, 0, nd, closing]
                used.add(best)
        fog_move = self.last_enemy_kind not in ("vis", "build")
        keep = []
        for k, tr in enumerate(tracks):
            if k in used:
                keep.append(tr)
                continue
            c = tr[0]
            if t - tr[2] > PARAMS["track_ttl"]:
                continue
            if T[c] != 0 and T[c] != 5 and O[c] == 1 and t - tr[2] <= 1:
                # we took its cell: it fought us; keep only if a big enemy cell is adjacent
                continue
            if fog_move:
                tr[3] += 1
            keep.append(tr)
        self.tracks = keep[:8]

    def locate_new_castle(self, new_fog, vis_enemy_castles, prev):
        for i in new_fog:
            return i
        for i in vis_enemy_castles:
            if i not in prev["enemy_castles"]:
                return i
        return -1

    def register_enemy_build(self, cell, price, prev_castles):
        self.enemy_builds.append((self.turn, cell, price))
        self.ecastle_built[cell] = self.turn
        surcharge = price - 35
        W = self.W
        for c in prev_castles:
            d = self.manh(c, cell)
            if d <= 6:
                surcharge -= 14 - 2 * d
        if surcharge < 0 or surcharge > 12 or surcharge % 2:
            return
        if surcharge > 0:
            ring = (14 - surcharge) // 2
            self.rings.append((cell, ring))
        else:
            self.rings.append((cell, None))
        self.apply_rings()

    def apply_rings(self):
        if self.egen >= 0 or not self.cands:
            return
        keep = set()
        for c in self.cands:
            ok = True
            for cell, ring in self.rings:
                d = self.manh(c, cell)
                if ring is None:
                    if d <= 6:
                        ok = False
                        break
                elif d != ring:
                    ok = False
                    break
            if ok:
                keep.add(c)
        if keep:
            self.cands = keep

    def prune_candidates(self):
        if self.egen >= 0:
            self.cands = {self.egen}
            return
        if not self.cands:
            # belief collapsed (shouldn't happen): fall back to unseen far cells
            self.cands = {i for i in self.cands0 if not self.ever_visible[i]} or set(self.cands0)
            return
        # an enemy cell e seen at turn te must be within reach of the general:
        # dist(general, e) <= te  and (connected territory) dist <= opp_land
        enemy_seen = [i for i in range(self.n) if self.last_owner[i] == 2 and self.first_enemy_seen[i] >= 0]
        if not enemy_seen:
            return
        keep = set()
        for c in self.cands:
            dc = self.cand_dist.get(c)
            if dc is None:
                keep.add(c)
                continue
            ok = True
            for e in enemy_seen:
                if dc[e] > self.first_enemy_seen[e] + 1:
                    ok = False
                    break
            if ok:
                keep.add(c)
        if keep:
            self.cands = keep

    def price_for(self, structs, i):
        cost = 35
        W = self.W
        r, c = divmod(i, W)
        for j in structs:
            rj, cj = divmod(j, W)
            d = abs(rj - r) + abs(cj - c)
            if d <= 6:
                cost += 14 - 2 * d
        return cost

    # ------------------------------------------------------------ belief API
    def belief_scores(self, frm=None):
        """Lower is better: exploration cost plus distance to recent enemy land."""
        cands = self.cands or set(self.cands0)
        if not cands:
            return {}
        t = self.turn
        if frm is None:
            frm = [i for i in range(self.n) if self.O[i] == 1]
        elif isinstance(frm, int):
            frm = [frm]
        dfrm = self.bfs(frm) if frm else self.dist_g
        recent = [i for i in range(self.n) if self.last_owner[i] == 2 and self.last_seen[i] >= t - 80]
        de = self.bfs(recent) if recent else None
        P = PARAMS
        out = {}
        pr = self.prior
        for c in cands:
            sc = P["belief_explore_w"] * min(dfrm[c], 60)
            if de is not None:
                sc += P["belief_enemy_w"] * min(de[c], 40)
            sc -= P["belief_prior_w"] * pr.get(c, 0.0)
            out[c] = sc
        return out

    def belief_target(self, frm=None):
        """Most likely / cheapest-to-check enemy general cell."""
        if self.egen >= 0:
            return self.egen
        sc = self.belief_scores(frm)
        if not sc:
            return -1
        return min(sc, key=sc.get)

    # ---------------------------------------------------------- local resolve
    def resolve(self, mine_act, en_act):
        """Exact one-turn resolution on the visible board. Returns (my_gen_lost, en_gen_taken)."""
        A, O = self.A, self.O
        ov = {}

        def get(i):
            if i in ov:
                return ov[i]
            return (O[i], A[i])

        def amount(owner, a):
            k = a[0]
            if k != 0:
                return None
            r, c, d, sp = a[1], a[2], a[3], a[4]
            src = r * self.W + c
            dr, dc = DIRS[d]
            rr, cc = r + dr, c + dc
            if not (0 <= rr < self.H and 0 <= cc < self.W):
                return None
            dst = rr * self.W + cc
            o, arm = get(src)
            if o != owner or self.mountain[dst]:
                return None
            amt = arm // 2 if sp == 1 else arm - 1
            amt = min(amt, arm - 1)
            if amt <= 0:
                return None
            return src, dst, amt

        # builds first
        if mine_act[0] == 2:
            bi = mine_act[1] * self.W + mine_act[2]
            o, arm = get(bi)
            ov[bi] = (o, arm - self.price_for(self.my_structs, bi))
            mine_act = PASS
        if en_act[0] == 2:
            en_act = PASS
        acts = {1: mine_act, 2: en_act}

        def key(owner):
            a = acts[owner]
            other = acts[3 - owner]
            if a[0] != 0:
                return (True, True, 2 ** 31, 0)
            dr, dc = DIRS[a[3]]
            di, dj = a[1] + dr, a[2] + dc
            chasing = other[0] == 0 and di == other[1] and dj == other[2]
            ci = min(max(di, 0), self.H - 1) * self.W + min(max(dj, 0), self.W - 1)
            reinf = get(ci)[0] == owner
            arm = get(min(max(a[1], 0), self.H - 1) * self.W + min(max(a[2], 0), self.W - 1))[1]
            pidx = self.pid if owner == 1 else 1 - self.pid
            return (not chasing, not reinf, arm, pidx)

        order = (1, 2) if key(1) < key(2) else (2, 1)
        late = self.turn >= 800
        egen = self.egen
        result = {1: False, 2: False}  # owner -> captured/touched the other general
        for owner in order:
            m = amount(owner, acts[owner])
            if m is None:
                continue
            src, dst, amt = m
            target_gen = self.general if owner == 2 else egen
            if late and dst == target_gen and target_gen >= 0:
                result[owner] = True
            o, arm = get(dst)
            if o == owner:
                ov[dst] = (o, arm + amt)
            else:
                if amt > arm:
                    ov[dst] = (owner, amt - arm)
                    if dst == target_gen and target_gen >= 0:
                        result[owner] = True
                else:
                    ov[dst] = (o, arm - amt)
            so, sa = get(src)
            ov[src] = (so, sa - amt)
        return result[2], result[1]

    def enemy_general_threats(self):
        """Enemy candidate moves that could hit our general this turn."""
        g = self.general
        out = []
        for j, _ in self.nb[g]:
            if self.O[j] == 2 and self.A[j] >= 2:
                for sp in (0, 1):
                    a = self.mv(j, g, sp)
                    if a:
                        out.append(a)
        return out

    def safe(self, act, threats=None):
        if threats is None:
            threats = self.enemy_general_threats()
        for ea in threats:
            lost, won = self.resolve(act, ea)
            if lost and not won:
                return False
        return True

    # ---------------------------------------------------------------- gather
    def guarded_castles(self):
        """Our castles with a visible enemy stack (>= 2) within castle_guard_r: keep them out of gathers."""
        if getattr(self, "_guard_t", -1) == self.turn:
            return self._guard
        out = set()
        if PARAMS["castle_keep"] and self.my_castles:
            r = PARAMS["castle_guard_r"]
            A, O = self.A, self.O
            foes = [j for j in range(self.n) if O[j] == 2 and A[j] >= 2]
            for c in self.my_castles:
                for j in foes:
                    if self.manh(c, j) <= r:
                        out.add(c)
                        break
        self._guard, self._guard_t = out, self.turn
        return out

    def gather_tree(self, root, allowed=None):
        O, A = self.O, self.A
        parent = {root: -1}
        depth = {root: 0}
        order = [root]
        dq = deque([root])
        guard = self.guarded_castles()
        while dq:
            i = dq.popleft()
            for j, _ in self.nb[i]:
                if j not in parent and O[j] == 1 and j not in guard and (allowed is None or allowed(j)):
                    parent[j] = i
                    depth[j] = depth[i] + 1
                    order.append(j)
                    dq.append(j)
        return parent, depth, order

    def gather_select(self, root, need=None, budget=None, allowed=None):
        """Greedy connected-subtree selection for gathering into root.

        Repeatedly adds the root-ward path with the best army-per-move ratio
        until the move budget is spent or `need` army is collected.
        Returns (parent, depth, selected set incl. root, collected value)."""
        A = self.A
        parent, depth, order = self.gather_tree(root, allowed)
        if budget is None:
            budget = PARAMS["gather_default_budget"]
        val = {i: A[i] - 1 for i in order}
        if PARAMS["castle_keep"]:
            for c in self.my_castles:
                if c in val:
                    val[c] = A[c] // 2
        val[root] = 0
        sel = {root}
        total = 0
        used = 0
        rest = order[1:]
        while used < budget and (need is None or total < need):
            if self.late(0.8):
                break
            gain, cost = {root: 0}, {root: 0}
            best, br, bc = -1, 0.0, 0
            room = budget - used
            for u in rest:
                p = parent[u]
                if u in sel:
                    gain[u], cost[u] = 0, 0
                    continue
                gu = val[u] + gain[p]
                cu = 1 + cost[p]
                gain[u], cost[u] = gu, cu
                if cu <= room and gu > 0:
                    r = gu / cu
                    if need is not None and gu >= need - total:
                        r += 1000.0 / cu  # finishing the job cheaply beats ratio
                    if r > br:
                        best, br, bc = u, r, cu
            if best < 0:
                break
            u = best
            while u not in sel:
                sel.add(u)
                total += val[u]
                used += 1
                u = parent[u]
        return parent, depth, sel, total

    def gather_move(self, root, need=None, budget=None, allowed=None):
        A = self.A
        parent, depth, sel, total = self.gather_select(root, need, budget, allowed)
        best, bk = -1, None
        has_child_army = set()
        for i in sel:
            if i != root and A[i] >= 2:
                has_child_army.add(parent[i])
        for i in sel:
            if i == root or A[i] < 2:
                continue
            if i in has_child_army:
                continue
            k = (depth[i], A[i])
            if bk is None or k > bk:
                bk, best = k, i
        if best < 0:
            return None, total
        if PARAMS["castle_keep"] and best in self.my_castles:
            return self.mv(best, parent[best], 1), total
        return self.mv(best, parent[best]), total

    # -------------------------------------------------------------- pathing
    def stealth_mask(self):
        """Cells the enemy can probably see (8-neighbourhood of recently seen enemy land)."""
        if getattr(self, "_stealth_t", -1) == self.turn:
            return self._stealth
        n, W, H = self.n, self.W, self.H
        m = [0] * n
        t = self.turn
        for i in range(n):
            if self.last_owner[i] == 2 and self.last_seen[i] >= t - 40:
                r, c = divmod(i, W)
                for rr in (r - 1, r, r + 1):
                    if 0 <= rr < H:
                        for cc in (c - 1, c, c + 1):
                            if 0 <= cc < W:
                                m[rr * W + cc] = 1
        self._stealth, self._stealth_t = m, t
        return m

    def path_to(self, src, dst, enemy_cost=1.0, avoid_general=True, stealth=False):
        """Cheapest path src->dst by Dijkstra (cost: 1 per step + army to beat).
        Never routes through our own general (a stack merging into it would
        then march the garrison out). With stealth, cells the enemy can see cost
        extra except near the target, so strikes arrive unseen."""
        A, O = self.A, self.O
        gen = self.general if (avoid_general and src != self.general) else -1
        sw = PARAMS["stealth_w"] if stealth else 0.0
        sm = self.stealth_mask() if sw > 0 else None
        slast = PARAMS["stealth_last"]
        W = self.W
        dr0, dc0 = divmod(dst, W)
        n = self.n
        dist = [INF] * n
        prev = [-1] * n
        dist[src] = 0
        h = [(0, src)]
        while h:
            d, i = heappop(h)
            if d > dist[i]:
                continue
            if i == dst:
                break
            for j, _ in self.nb[i]:
                if not self.pas[j] or j == gen:
                    continue
                if O[j] == 1:
                    c = 1
                elif O[j] == 2 or (self.T[j] == 3):
                    c = 1 + enemy_cost * A[j]
                elif self.T[j] == 5 or (self.castle[j] and self.T[j] == 0):
                    c = 1 + enemy_cost * max(20, self.last_army[j])
                else:
                    c = PARAMS["x_neutral_cost"]
                if sm is not None and sm[j]:
                    jr, jc = divmod(j, W)
                    if abs(jr - dr0) + abs(jc - dc0) > slast:
                        c += sw
                nd = d + c
                if nd < dist[j]:
                    dist[j] = nd
                    prev[j] = i
                    heappush(h, (nd, j))
        if dist[dst] >= INF:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    # ----------------------------------------------------------------- decide
    def decide(self):
        t = self.turn
        T, O, A = self.T, self.O, self.A
        g = self.general
        threats = self.enemy_general_threats()

        a = self.win_now()
        if a:
            return a
        if PARAMS["x_dt_guard"] and t >= PARAMS["dt_guard_turn"] and g >= 0:
            a = self.dt_guard()
            if a:
                return a
        a = self.urgent_defense(threats)
        if a:
            return a
        if PARAMS["x_intercept"] and t >= PARAMS["open_end"]:
            a = self.intercept(threats)
            if a:
                return a
        if t < PARAMS["open_end"]:
            a = self.opening()
            if a and self.safe(a, threats):
                return a
        a = self.macro()
        if a is None:
            a = PASS
        if a[0] != 1 and not self.safe(a, threats):
            alt = self.defensive_alternatives(threats)
            if alt:
                return alt
        return a

    # ------------------------------------------------------------- intercept
    def qualified_threats(self):
        """Enemy stacks that threaten the general (near, big, or moving in)."""
        A, O, pO, pA = self.A, self.O, self.pO, self.pA
        Ag = A[self.general]
        out = []
        for e in range(self.n):
            if O[e] != 2 or A[e] < 2:
                continue
            eta = self.dist_g[e]
            if eta > 5:
                continue
            a = A[e]
            moving = pO is not None and ((pO[e] == 2 and pA[e] != a) or any(
                pO[z] == 2 and pA[z] >= a - 2 and pA[e] <= 1 for z, _ in self.nb[e]))
            lethal = eta == 1 and (self.turn >= 800 or a - 1 > Ag)
            if lethal or (eta <= 2 and a >= max(4, Ag // 2)) or (moving and a >= 6):
                out.append((eta, -a, e))
        out.sort()
        return [e for _, _, e in out]

    def intercept(self, threats):
        """Kill a qualified attacker now from an adjacent non-general stack
        (chasing moves resolve first, so it cannot slip away this turn).
        Idea from relh Sentinel / juraj (see agents_shared/algo-study.md 2.1)."""
        best = None
        g = self.general
        A, O = self.A, self.O
        for e in self.qualified_threats():
            for x, _ in self.nb[e]:
                if O[x] == 1 and x != g and A[x] - 1 > A[e]:
                    k = (-self.dist_g[e], -A[x])
                    if best is None or k > best[0]:
                        best = (k, x, e)
        if best:
            a = self.mv(best[1], best[2])
            if a and self.safe(a, threats):
                self.last_label = "intercept"
                return a
        return None

    # ---------------------------------------------------------------- winning
    def win_now(self):
        eg = self.egen
        if eg < 0:
            return None
        A, O = self.A, self.O
        t = self.turn
        best = None
        for j, _ in self.nb[eg]:
            if O[j] == 1 and A[j] >= 2:
                if t >= 800 or A[j] - 1 > A[eg]:
                    if best is None or A[j] > A[best]:
                        best = j
        if best is not None:
            return self.mv(best, eg)
        return None

    def urgent_defense(self, threats):
        if not threats:
            return None
        if self.safe(PASS, threats):
            return None
        # find a move that makes us safe; prefer chase kills of the attacker source,
        # then reinforcing the general with the largest neighbor stack.
        g = self.general
        best, bk = None, None
        for a in self.defensive_candidates(threats):
            if self.safe(a, threats):
                arm = self.A[a[1] * self.W + a[2]]
                k = (1 if a[0] == 0 else 0, arm)
                if bk is None or k > bk:
                    bk, best = k, a
        if best:
            return best
        # cannot be fully safe: if we can win/draw by touching first, do it
        return None

    def defensive_candidates(self, threats):
        g = self.general
        A, O = self.A, self.O
        out = []
        srcs = {t[1] * self.W + t[2] for t in threats}
        # chase the attacker source from a third tile
        for s in srcs:
            for z, _ in self.nb[s]:
                if z != g and O[z] == 1 and A[z] >= 2:
                    out.append(self.mv(z, s))
            if A[g] >= 2:
                out.append(self.mv(g, s))
        # reinforce general
        for j, _ in self.nb[g]:
            if O[j] == 1 and A[j] >= 2:
                out.append(self.mv(j, g))
        return [a for a in out if a]

    def defensive_alternatives(self, threats):
        best, bk = None, None
        for a in self.defensive_candidates(threats) + [PASS]:
            if self.safe(a, threats):
                arm = self.A[a[1] * self.W + a[2]] if a[0] == 0 else 0
                if bk is None or arm > bk:
                    bk, best = arm, a
        return best

    # ---------------------------------------------------------------- opening
    def opening(self):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        g = self.general
        cfg = self.open_cfg
        # 1) continue a wave: a non-general own cell with >= 2 adjacent to neutral
        best, bk = None, None
        for i in range(self.n):
            if O[i] == 1 and i != g and A[i] >= 2:
                for j, _ in self.nb[i]:
                    if self.pas[j] and O[j] == 0 and not self.castle[j]:
                        k = (self.open_score(j), A[i])
                        if bk is None or k > bk:
                            bk, best = k, (i, j)
        if best:
            return self.mv(*best)
        # 2) stack stuck inside own land: move it toward nearest neutral
        stuck = [i for i in range(self.n) if O[i] == 1 and i != g and A[i] >= 2]
        if stuck:
            i = max(stuck, key=lambda k: A[k])
            step = self.step_to_frontier(i)
            if step is not None:
                return self.mv(i, step)
        # 3) launch from general
        a = A[g]
        if a >= 2:
            dfront, step = self.frontier_info(g)
            if step is not None:
                transit = max(0, dfront - 1)
                need = int((49 - t - transit) / cfg["div"])
                if a - 1 >= need or (t >= cfg["late"] and a >= 2):
                    return self.mv(g, step)
        return PASS

    def open_score(self, j):
        """Prefer open areas, away from the general, toward the enemy candidates."""
        cfg = self.open_cfg
        free = 0
        for k, _ in self.nb[j]:
            if self.pas[k] and self.O[k] == 0:
                free += 1
        dc = self.open_dc
        toward = -dc[j] * cfg["w_toward"] if dc else 0.0
        return cfg["w_free"] * free + cfg["w_dist"] * min(self.dist_g[j], 12) + toward

    def plan_opening(self):
        """Pick the opening configuration that maximises land at turn 50 in a
        single-player simulation (the opponent cannot interfere this early)."""
        g = self.general
        tgt = max(self.prior, key=self.prior.get) if self.prior else -1
        self.open_dc = self.cand_dist.get(tgt) if tgt >= 0 else None
        best, bkey = None, None
        save = (getattr(self, "O", None), getattr(self, "A", None), getattr(self, "T", None), self.turn)
        deadline = time.perf_counter() + PARAMS["open_plan_s"]
        for div in (2.0, 1.7, 2.4, 1.5):
            for w_free, w_dist in ((1.0, 0.3), (1.0, 0.0), (0.5, 0.6), (1.5, 0.3)):
                for w_toward in (0.15, 0.0):
                    if time.perf_counter() > deadline:
                        break
                    cfg = {"div": div, "w_free": w_free, "w_dist": w_dist, "w_toward": w_toward, "late": 44}
                    land50, frontier = self.sim_opening(cfg)
                    key = (land50, frontier + (1 if w_toward > 0 else 0))
                    if bkey is None or key > bkey:
                        bkey, best = key, cfg
        self.O, self.A, self.T, self.turn = save
        self.open_cfg = best or {"div": 2.0, "w_free": 1.0, "w_dist": 0.3, "w_toward": 0.15, "late": 44}
        self.open_expect = bkey

    def sim_opening(self, cfg):
        n = self.n
        g = self.general
        O = [0] * n
        A = [0] * n
        O[g] = 1
        A[g] = 1
        self.T = [1] * n
        self.open_cfg = cfg
        W = self.W
        for t in range(50):
            self.O, self.A, self.turn = O, A, t
            a = self.opening()
            if a and a[0] == 0:
                i = a[1] * W + a[2]
                dr, dc = DIRS[a[3]]
                j = (a[1] + dr) * W + a[2] + dc
                amt = A[i] // 2 if a[4] == 1 else A[i] - 1
                if amt > 0 and O[i] == 1 and self.pas[j]:
                    A[i] -= amt
                    if O[j] == 1:
                        A[j] += amt
                    else:
                        O[j] = 1
                        A[j] = amt
            if (t + 1) % 2 == 0:
                A[g] += 1
        land = sum(O)
        frontier = 0
        for i in range(n):
            if O[i] == 0 and self.pas[i]:
                for j, _ in self.nb[i]:
                    if O[j] == 1:
                        frontier += 1
                        break
        return land, frontier

    def frontier_info(self, src):
        """(distance to nearest neutral cell through own cells, first step)."""
        O = self.O
        prev = {src: -1}
        dq = deque([src])
        while dq:
            i = dq.popleft()
            for j, _ in self.nb[i]:
                if j in prev or not self.pas[j]:
                    continue
                if O[j] == 0 and not self.castle[j]:
                    prev[j] = i
                    # reconstruct first step
                    path = [j]
                    while prev[path[-1]] != src:
                        path.append(prev[path[-1]])
                    return len(path), path[-1]
                if O[j] == 1:
                    prev[j] = i
                    dq.append(j)
        return INF, None

    def step_to_frontier(self, i):
        d, step = self.frontier_info(i)
        return step

    # ------------------------------------------------------------------ macro
    def macro(self):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        g = self.general
        P = PARAMS

        self.compute_mods()
        if t >= P["fortress_turn"]:
            a = self.endgame()
            if a:
                return a
        a = self.try_kill()
        if a:
            return a

        need_g = max(P["garrison_min"], int(self.garrison_need() * self.mod["garrison"]))
        self.need_g = need_g
        self.enemy_dist = self.enemy_distance_map()

        options = []  # (score, action)
        cap = self.best_capture(need_g)
        if cap:
            options.append(cap + ("capture",))
        defence = A[g]
        if P["ring_w"] > 0:
            R = P["ring_r"]
            dg = self.dist_g
            ring = 0
            for i in range(self.n):
                if O[i] == 1 and 0 < dg[i] <= R and A[i] > 1:
                    ring += A[i] - 1
            defence += int(P["ring_w"] * ring)
        self.defence = defence
        if defence < need_g:
            eta = self.threat_eta
            budget = None if eta >= INF else max(P["min_def_budget"], eta - 1)
            a, tot = self.gather_move(g, need=need_g - defence, budget=budget)
            if a:
                urgent = eta <= P["track_threat_dist"]
                options.append((P["w_garrison_urgent"] if urgent else P["w_garrison"], a, "garrison"))
        b = self.castle_build_now()
        if b:
            options.append(b + ("build",))
        sc = self.scout_move() if not self.late(0.5) else None
        if sc:
            options.append(sc + ("scout",))
        hf = self.home_fill_move(need_g) if not self.late(0.5) else None
        if hf:
            options.append(hf + ("home",))
        self.pending_stack = None
        c = self.cycle_move(need_g) if not self.late(0.6) else None
        if c:
            launching = self.cyc is not None and self.cyc.get("mode") in ("launch", "march")
            wl = P["w_launch"]
            if P["lead_w"] > 0:
                lead = 0.5 * math.log(max(1, self.my_army) / max(1, self.opp_army)) + \
                    0.5 * math.log(max(1, self.my_land) / max(1, self.opp_land))
                wl *= 1.0 + P["lead_w"] * math.tanh(2.0 * lead)
            wl *= self.mod["launch"]
            options.append((wl if launching else P["w_cycle"] * self.mod["cycle"], c,
                            ("launch" if launching else "cyc_" + str(self.cyc.get("purpose") if self.cyc else ""))))
        if not options:
            a, tot = self.gather_move(g, budget=10)
            if a is None and P["x_no_pass"]:
                a = self.any_capture()
                if a:
                    self.last_label = "nopass_capture"
                    return a
            self.last_label = "idle_gather" if a else "pass"
            return a or PASS
        options.sort(key=lambda x: -x[0])
        pick = 0
        if RO_W is not None and len(options) > 1:
            pick = _ro_pick(self, options, need_g)
        choice = options[pick][1]
        self.last_label = options[pick][2]
        if c is not None and choice is c and self.cyc is not None:
            if self.cyc.get("mode") in ("launch", "march") and self.pending_stack is not None:
                self.cyc["stack"] = self.pending_stack
            elif self.cyc.get("mode") == "gather":
                self.cyc["moves"] = self.cyc.get("moves", 0) + 1
        return choice

    # -------------------------------------------------------------- garrison
    def threat_features(self):
        hidden = self.opp_army - self.vis_enemy_army - max(0, self.opp_land - self.vis_enemy_cells - 1)
        vis_max = 0
        for i in range(self.n):
            if self.O[i] == 2 and self.A[i] > vis_max:
                vis_max = self.A[i]
        tr_army = max((tr[1] for tr in self.tracks), default=0)
        tr_dist = min((max(1, self.dist_g[tr[0]] - (self.turn - tr[2])) for tr in self.tracks), default=40)
        fd = self.fog_distance()
        return [math.log1p(max(0, hidden)), math.log1p(self.opp_army), math.log1p(self.my_army),
                self.opp_land / 100.0, self.my_land / 100.0, min(fd, 20) / 10.0, len(self.enemy_castles) / 5.0,
                self.turn / 1000.0, math.log1p(vis_max), math.log1p(tr_army), min(tr_dist, 40) / 10.0,
                math.log1p(self.vis_enemy_army)]

    MOD_GROUPS = ("launch", "cycle", "capture", "garrison", "scout", "home", "kill")

    def mod_features(self):
        """8 context features in about [-1, 1] (opponent / situation descriptors)."""
        t = self.turn
        f1 = math.tanh(math.log(max(1, self.my_army) / max(1, self.opp_army)))
        f2 = math.tanh(math.log(max(1, self.my_land) / max(1, self.opp_land)))
        f3 = min(len(self.enemy_castles), 5) / 2.5 - 1.0
        f4 = 2.0 * self.enemy_gather_ratio() - 1.0
        f5 = 1.0 if self.egen >= 0 else -1.0
        tr = max((tr[1] for tr in self.tracks), default=0)
        f6 = math.tanh(tr / max(1.0, self.A[self.general] + 1.0) - 1.0)
        return (t / 600.0 - 1.0, f1, f2, f3, f4, f5, f6, 1.0)

    def compute_mods(self):
        P = PARAMS
        self.mod = mod = {}
        if not P["mod_on"]:
            for gname in self.MOD_GROUPS:
                mod[gname] = 1.0
            return
        try:
            f = self.mod_features()
        except Exception:
            if DEBUG:
                raise
            f = (0.0,) * 7 + (1.0,)
        for gname in self.MOD_GROUPS:
            z = 0.0
            for j in range(8):
                z += P["m_%s_%d" % (gname, j)] * f[j]
            mod[gname] = math.exp(max(-1.5, min(1.5, z)))

    def garrison_need(self):
        """Army the general should hold, and the time (eta) we have to get it.

        The binding threat (largest requirement) sets self.threat_eta, which is
        the move budget for defensive gathering."""
        A, O = self.A, self.O
        g = self.general
        t = self.turn
        P = PARAMS
        need = P["garrison_min"]
        eta = INF
        for i in range(self.n):
            if O[i] == 2 and A[i] >= 3:
                d = self.dist_g[i]
                if d < P["threat_vis_range"]:
                    ni = A[i] - int((d - 1) * P["threat_decay"]) + P["threat_margin"]
                    if ni > need:
                        need, eta = ni, d
        for c, army, ts, adv, _pd, closing in self.tracks:
            if ts == t:
                d = max(1, self.dist_g[c])          # visible right now
            else:
                d = max(1, self.dist_g[c] - (adv if P["fog_tracks"] else min(t - ts, P["track_max_adv"])))
            if d <= P["track_threat_dist"] or (closing >= P["closing_min"] and d <= P["threat_far"]):
                ni = army - int((d - 1) * P["threat_decay"] * P["far_decay_mult"]) + P["threat_margin"] \
                    if d > P["track_threat_dist"] else army + P["threat_margin"]
                if ni > need:
                    need, eta = ni, d
        hidden = self.opp_army - self.vis_enemy_army - max(0, self.opp_land - self.vis_enemy_cells - 1)
        if hidden > 0:
            wd = self.fog_distance()
            hidden = min(hidden, int(self.opp_army * P["hidden_stack_frac"]))
            if wd <= 4:
                f = P["garrison_frac_hidden"]
            elif wd <= 7:
                f = P["garrison_frac_hidden"] * 0.6
            else:
                f = P["garrison_frac_hidden"] * 0.3
            ni = int(hidden * f)
            if P["fog_model"] > 0:
                ni = int(P["fog_model"] * self.fog_risk(max(0, wd - 1)))
                ratio = self.enemy_gather_ratio()
                if ratio > 0.6:
                    ni = int(ni * (1 + P["gather_ratio_adj"]))
            if ni > need:
                need, eta = ni, max(2, wd)
        self.threat_eta = eta
        cap = int(self.my_army * P["garrison_cap_frac"])
        return min(need, max(P["garrison_min"], cap))

    def fog_distance(self):
        T = self.T
        best = INF
        dg = self.dist_g
        for i in range(self.n):
            if T[i] == 0 and dg[i] < best:
                best = dg[i]
        return best

    def enemy_distance_map(self):
        cells = [i for i in range(self.n) if self.last_owner[i] == 2 and self.last_seen[i] >= self.turn - 60]
        cells += list(self.enemy_castles)
        if not cells:
            return [INF] * self.n
        return self.bfs(cells)

    # -------------------------------------------------------------- captures
    def best_capture(self, need_g):
        """Best single capture move with a heuristic value (score, action)."""
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        g = self.general
        t = self.turn
        small_cap = max(P["small_min"], int(P["small_frac"] * self.my_army))
        bonus = t % 50 >= 50 - P["bonus_lead"] or t % 50 < P["bonus_window"]
        tgt = self.belief_target()
        dc = self.cand_dist.get(tgt) if tgt >= 0 else None
        best, bs = None, None
        bmult = P["bonus_mult"] if bonus else 1.0
        if P["x_bonus_graded"] and not bonus:
            tt = 50 - t % 50
            if tt <= 5:
                bmult = 1.0 + 0.6 * (P["bonus_mult"] - 1.0)
            elif tt <= 10:
                bmult = 1.0 + 0.4 * (P["bonus_mult"] - 1.0)
            elif tt <= 20:
                bmult = 1.0 + 0.2 * (P["bonus_mult"] - 1.0)
        busy = set()
        if self.cyc:
            busy = {self.cyc.get("root"), self.cyc.get("stack")}
        keep = self.my_castles if P["castle_keep"] else ()
        guard = self.guarded_castles()
        for i in range(self.n):
            ai = A[i]
            if O[i] != 1 or ai < 2:
                continue
            is_g = i == g
            for j, d in self.nb[i]:
                if not self.pas[j] or O[j] == 1:
                    continue
                aj = A[j]
                split = 0
                send = ai - 1
                if i in keep:
                    if i in guard and O[j] != 2:
                        continue
                    split, send = 1, ai // 2
                if is_g:
                    if ai - 1 - aj >= 1 and ai - (ai - 1) >= need_g:
                        split = 0
                    elif ai // 2 > aj and ai - ai // 2 >= need_g:
                        split, send = 1, ai // 2
                    else:
                        continue
                if send <= aj:
                    continue
                if O[j] == 2:
                    v = P["v_enemy"] + P["v_kill"] * aj
                    if T[j] == 3:
                        v += P["v_ecastle"]
                    if self.manh(j, g) <= 3:
                        v += P["v_near_home"]
                    if P["x_early_enemy"] and t < P["early_expand_until"]:
                        v += P["early_expand_bonus"] * P["x_early_enemy"]
                    if P["x_phase_enemy"]:
                        # replays: ~75% of enemy-tile captures land in the 15 ticks before the land bonus
                        phi = t % 50
                        if phi >= P["pe_hi_phi"]:
                            v *= P["pe_hi"]
                        elif phi < P["pe_lo_phi"]:
                            v *= P["pe_lo"]
                else:
                    if T[j] == 3:
                        continue
                    dgj = self.dist_g[j]
                    home = max(0, P["home_r"] - dgj) / P["home_r"]
                    early = t < P["early_expand_until"]
                    if ai > small_cap and not is_g and not (home > 0 and ai <= 3 * small_cap) and not early:
                        if not (P["x_big_route"] and dc and dc[j] < dc[i] and i not in busy):
                            continue
                    v = P["v_neutral"] * bmult + P["v_home"] * home
                    if t < P["early_expand_until"]:
                        v += P["early_expand_bonus"]
                    elif (P["x_spread"] and t % 50 < P["spread_phi"] and ai <= P["spread_max_a"]
                          and not is_g and i not in busy):
                        # replays: right after each land bonus the +1 on every tile is spread into neutral land
                        v += P["early_expand_bonus"] * P["x_spread"]
                    if dc:
                        v -= P["toward_w"] * min(dc[j], 30) / 30.0
                    v -= 0.002 * ai
                if is_g:
                    v -= P["gen_move_pen"]
                if bs is None or v > bs:
                    bs, best = v, [0, i // self.W, i % self.W, d, split]
        if best is None:
            return None
        return (bs * self.mod["capture"], best)

    # ------------------------------------------------------------- home zone
    def home_fill_move(self, need_g):
        """Own every cell within home_r of the general: vision = warning time."""
        t = self.turn
        P = PARAMS
        if t < P["home_fill_start"]:
            return None
        A, O = self.A, self.O
        g = self.general
        R = P["home_r"]
        dg = self.dist_g
        holes = [j for j in range(self.n) if self.pas[j] and O[j] != 1 and dg[j] <= R
                 and not (self.castle[j] and O[j] == 0)]
        if not holes:
            return None
        best, bk = None, None
        for j in holes:
            for i, d in self.nb[j]:
                if O[i] != 1 or A[i] < 2:
                    continue
                if i == g:
                    send = A[i] // 2
                    if send <= A[j] or A[i] - send < need_g:
                        continue
                    split = 1
                else:
                    send = A[i] - 1
                    if send <= A[j]:
                        continue
                    split = 0
                k = (-dg[j], -A[i] if i != g else -10 ** 6)
                if bk is None or k > bk:
                    dd = (d ^ 1)  # direction from i to j is the reverse of j->i
                    bk, best = k, [0, i // self.W, i % self.W, dd, split]
        if best is None:
            # bring a small stack next to the nearest hole
            return None
        return (P["w_home_fill"] * self.mod["home"], best)

    # ---------------------------------------------------------------- scouting
    def scout_move(self):
        """Walk a modest stack toward the nearest unexplored general candidate."""
        t = self.turn
        P = PARAMS
        if self.egen >= 0 or t < P["scout_start"] or not self.cands:
            return None
        A, O = self.A, self.O
        g = self.general
        cyc = self.cyc
        busy = set()
        if cyc:
            busy = {cyc.get("root"), cyc.get("stack")}
        lo, hi = P["scout_min"], max(P["scout_min"] + 1, int(P["scout_max_frac"] * self.my_army))
        srcs = [i for i in range(self.n) if O[i] == 1 and lo <= A[i] <= hi and i != g and i not in busy
                and not (P["castle_keep"] and i in self.my_castles)]
        if not srcs:
            return None
        dc = self.bfs(list(self.cands))
        i = min(srcs, key=lambda k: (dc[k], -A[k]))
        if dc[i] >= INF or dc[i] == 0:
            return None
        best, bv = None, None
        for j, d in self.nb[i]:
            if not self.pas[j] or dc[j] >= dc[i]:
                continue
            if O[j] != 1 and A[i] - 1 <= A[j]:
                continue
            v = -A[j] if O[j] != 1 else 0
            if bv is None or v > bv:
                bv, best = v, j
        if best is None:
            return None
        return (P["w_scout"] * self.mod["scout"], self.mv(i, best))

    def any_capture(self):
        """Any safe capture by a non-general stack (used instead of a PASS)."""
        A, O = self.A, self.O
        g = self.general
        threats = self.enemy_general_threats()
        best, bk = None, None
        for i in range(self.n):
            if O[i] != 1 or A[i] < 2 or i == g:
                continue
            for j, d in self.nb[i]:
                if not self.pas[j] or O[j] == 1 or (self.T[j] == 3 and O[j] == 0):
                    continue
                if A[i] - 1 > A[j]:
                    k = (O[j] == 2, -A[i])
                    if bk is None or k > bk:
                        bk, best = k, [0, i // self.W, i % self.W, d, 0]
        if best and self.safe(best, threats):
            return best
        return None

    # --------------------------------------------------------------- castles
    def castle_build_now(self):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        if not (P["castle_start"] <= t <= P["castle_stop"]):
            return None
        horizon = P["castle_horizon"]
        de = self.enemy_dist
        structs = self.my_structs
        best, bv = None, None
        cyc = self.cyc
        reserved = set()
        stack_ok = None
        if cyc and cyc.get("purpose") == "attack":
            reserved = {cyc.get("root"), cyc.get("stack")}
            if cyc.get("mode") == "launch" and len(self.my_castles) < self.n_castles_wanted():
                stack_ok = cyc.get("stack")
        for i in range(self.n):
            if O[i] == 1 and T[i] == 1 and A[i] >= 35:
                if i in reserved and not (i == stack_ok and self.price_for(structs, i) <= P["castle_stack_price"]):
                    continue
                if de[i] < P["castle_safe_dist"]:
                    continue
                price = self.price_for(structs, i)
                if A[i] < price + P["castle_reserve"]:
                    continue
                v = 0.5 * (horizon - t) - price * P["castle_price_w"]
                if bv is None or v > bv:
                    bv, best = v, i
        if best is None or bv < P["castle_val_min"]:
            return None
        if self.threat_eta <= P["track_threat_dist"] and self.A[self.general] < self.need_g:
            return None
        return (P["w_build"], [2, best // self.W, best % self.W, 0, 0])

    def n_castles_wanted(self):
        t = self.turn
        P = PARAMS
        if t < P["castle_start"]:
            return 0
        return 1 + int((t - P["castle_start"]) / P["castle_every"])

    def castle_site(self):
        """Best site to gather a castle at: (cell, price) or None."""
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        de = self.enemy_dist
        structs = self.my_structs
        g = self.general
        dfront = None
        tgt = -1
        if P["castle_front_w"] != 0:
            tgt = self.belief_target()
            if tgt >= 0:
                dfront = self.cand_dist.get(tgt) or self.bfs([tgt])
        best, bv = None, None
        for i in range(self.n):
            if O[i] != 1 or T[i] != 1 or i == g:
                continue
            if de[i] < P["castle_safe_dist"] + 1:
                continue
            price = self.price_for(structs, i)
            dg = self.dist_g[i]
            v = -price * P["castle_price_w"] - P["castle_move_w"] * max(0, dg - 7) * 0.5
            v += min(de[i], 12) * P["castle_safety_w"]
            near_home = sum(1 for c in self.my_castles if self.dist_g[c] <= P["castle_home_maxd"])
            if near_home < P["castle_home_n"]:
                v -= P["castle_home_w"] * max(0, dg - P["castle_home_maxd"])
            if dfront is not None:
                # enemy-facing: closer to the likely enemy general than our general is
                v += P["castle_front_w"] * (self.dist_g[tgt] - dfront[i]) / 10.0
            if bv is None or v > bv:
                bv, best = v, (i, price)
        return best

    # ------------------------------------------------------------ army cycle
    def choose_target(self, frm=None):
        """Target for the attack stack."""
        A, O, T = self.A, self.O, self.T
        g = self.general
        P = PARAMS
        best, bk = -1, None
        home = [g] + list(self.my_castles)
        for i in range(self.n):
            if O[i] == 2 and A[i] >= 4:
                d = min(self.manh(i, h) for h in home)
                if d <= P["intercept_dist"]:
                    k = A[i] - 3 * d
                    if bk is None or k > bk:
                        bk, best = k, i
        if best >= 0:
            return best
        if self.egen >= 0:
            return self.egen
        if self.enemy_castles and frm is not None:
            df = self.bfs([frm])
            c = min(self.enemy_castles, key=lambda c: df[c])
            est = self.castle_army_est(c)
            if df[c] < P["snipe_range"] and A[frm] > est + df[c] // 2 + P["snipe_margin"]:
                return c
        return self.belief_target(frm)

    def castle_army_est(self, c):
        """Estimated army on an enemy castle: last seen value + growth, or, for a castle we
        only know from the build accounting, a small remainder + growth since the build."""
        t = self.turn
        if self.last_seen[c] >= 0 and self.last_owner[c] == 2 and self.T[c] == 3:
            return self.A[c]
        tb = self.ecastle_built.get(c)
        if tb is not None and (self.last_seen[c] < tb):
            return P_SNIPE_REMAINDER + (t - tb) // 2 + (t // 50 - tb // 50)
        if self.last_seen[c] >= 0:
            return self.last_army[c] + (t - self.last_seen[c]) // 2 + (t // 50 - self.last_seen[c] // 50)
        return t // 2

    def cycle_move(self, need_g):
        t = self.turn
        A, O, T = self.A, self.O, self.T
        P = PARAMS
        g = self.general
        cyc = self.cyc
        if cyc is None or cyc.get("turn") is None:
            cyc = self.cyc = self.new_cycle(need_g)
            if cyc is None:
                return None
        if cyc["mode"] == "march":
            s_, site = cyc["stack"], cyc["site"]
            if O[s_] != 1 or t - cyc["turn"] > P["castle_g_walk"] + 6:
                self.cyc = None
                return None
            if s_ == site:
                self.cyc = None
                if T[site] == 1 and A[site] >= self.price_for(self.my_structs, site) + P["castle_reserve"]:
                    return [2, site // self.W, site % self.W, 0, 0]
                return None
            path = self.bfs_path_own(s_, site)
            if not path or len(path) < 2:
                self.cyc = None
                return None
            self.pending_stack = path[1]
            return self.mv(s_, path[1], 1 if s_ == g else 0)
        if cyc["mode"] == "gather":
            root = cyc["root"]
            if O[root] != 1:
                self.cyc = None
                return None
            if cyc["purpose"] == "castle":
                site, price = cyc["site"], cyc["price"]
                if A[root] >= price + P["castle_reserve"]:
                    self.cyc = None  # build handled by castle_build_now next turn
                    return [2, root // self.W, root % self.W, 0, 0]
                need = price + P["castle_reserve"] - A[root]
            else:
                need = None
            elapsed = cyc.get("moves", 0)
            if t - cyc["turn"] > 3 * cyc["budget"] + 5:
                elapsed = cyc["budget"]
            if elapsed < cyc["budget"]:
                # feed the general's surplus toward the root first
                surplus = A[g] - need_g
                if surplus >= P["feed_min"] and root != g:
                    path = self.bfs_path_own(g, root)
                    if path and len(path) >= 2:
                        split = 0 if need_g <= 1 or A[g] - 1 < need_g else 1
                        if split == 0 and A[g] - (A[g] - 1) < need_g:
                            split = 1
                        if split == 1 and A[g] - A[g] // 2 < need_g:
                            split = None
                        if split is not None:
                            return self.mv(g, path[1], split)
                a, tot = self.gather_move(root, need=need, budget=cyc["budget"] - elapsed,
                                          allowed=lambda j: j != g)
                if a:
                    return a
            if cyc["purpose"] == "castle":
                # could not gather enough: give up this castle
                self.cyc = None
                return None
            if root == g:
                # a castle is due: march the gathered stack to a cheap site instead
                if (P["castle_g"] and P["castle_start"] <= t <= P["castle_stop"]
                        and len(self.my_castles) < self.n_castles_wanted()):
                    site = self.castle_from_general(need_g)
                    if site:
                        cyc.update({"mode": "march", "purpose": "castle_g", "stack": g,
                                    "site": site[0], "price": site[1], "turn": t})
                        return self.cycle_move(need_g)
                # only strike when ahead (replay rule: winners attack at 1.2-1.4x army)
                if (P["attack_ratio"] > 0 and self.egen < 0 and t < P["aggro_turn"]
                        and self.my_army < P["attack_ratio"] * self.opp_army):
                    self.cyc = None
                    return None
                # launch half (or all but the garrison) of the general's stack
                tgt = self.choose_target(g)
                send_half = A[g] // 2
                send_all = A[g] - 1
                if A[g] - 1 - send_all >= need_g - 1 and send_all >= P["min_stack"]:
                    split, send = 0, send_all
                elif A[g] - send_half >= need_g and send_half >= P["min_stack"]:
                    split, send = 1, send_half
                else:
                    self.cyc = None
                    return None
                path = self.path_to(g, tgt, stealth=True) if tgt >= 0 else None
                if not path or len(path) < 2:
                    self.cyc = None
                    return None
                nxt = path[1]
                if O[nxt] != 1 and send <= A[nxt]:
                    self.cyc = None
                    return None
                cyc["mode"] = "launch"
                cyc["stack"] = g
                self.pending_stack = nxt
                return self.mv(g, nxt, split)
            if A[root] >= P["min_stack"]:
                cyc["mode"] = "launch"
                cyc["stack"] = root
            else:
                self.cyc = None
                return None
        # launch
        s = cyc["stack"]
        if O[s] != 1 or A[s] < P["min_stack"] or s == g:
            self.cyc = None
            return None
        tgt = self.choose_target(s)
        if tgt < 0:
            self.cyc = None
            return None
        if s == tgt:
            self.cyc = None
            return None
        path = self.path_to(s, tgt, stealth=True)
        if not path or len(path) < 2:
            self.cyc = None
            return None
        nxt = path[1]
        if (PARAMS["x_stage"] and tgt == self.egen and t < 800 and len(path) - 1 <= PARAMS["stage_d"]
                and cyc.get("staged", 0) < PARAMS["stage_max"]):
            # try_kill (run first) says this stack cannot kill yet: wait here and pull army in
            self.cyc = {"mode": "gather", "purpose": "attack", "root": s, "turn": t,
                        "budget": PARAMS["stage_budget"], "moves": 0, "staged": cyc.get("staged", 0) + 1}
            self.last_label = "stage"
            return None
        if O[nxt] != 1 and A[s] - 1 <= A[nxt]:
            # blocked: re-gather into the stack where it stands
            self.cyc = {"mode": "gather", "purpose": "attack", "root": s, "turn": t,
                        "budget": P["regather_budget"], "moves": 0}
            return None
        self.pending_stack = nxt
        if t - cyc["turn"] > cyc["budget"] + P["launch_max"]:
            self.cyc = None
        return self.mv(s, nxt)

    def castle_from_general(self, need_g):
        """Plan: walk half of the general's stack to a cheap own site and build there
        (how the top Marathon bots fund their first castles)."""
        P = PARAMS
        A, O, T = self.A, self.O, self.T
        g = self.general
        half = A[g] // 2
        if half < 35 + P["castle_reserve"] or A[g] - half < need_g:
            return None
        de = self.enemy_dist
        structs = self.my_structs
        # own-cell BFS from the general
        dist = {g: 0}
        dq = deque([g])
        while dq:
            i = dq.popleft()
            if dist[i] >= P["castle_g_walk"]:
                continue
            for j, _ in self.nb[i]:
                if j not in dist and O[j] == 1:
                    dist[j] = dist[i] + 1
                    dq.append(j)
        best, bv = None, None
        for i, L in dist.items():
            if i == g or T[i] != 1 or de[i] < P["castle_safe_dist"]:
                continue
            price = self.price_for(structs, i)
            if price + P["castle_reserve"] > half:
                continue
            v = -price - P["castle_g_walk_w"] * L
            if bv is None or v > bv:
                bv, best = v, (i, price)
        return best

    def new_cycle(self, need_g):
        t = self.turn
        P = PARAMS
        g = self.general
        A, O = self.A, self.O
        if (P["castle_g"] and P["castle_start"] <= t <= P["castle_stop"]
                and len(self.my_castles) < self.n_castles_wanted()):
            site = self.castle_from_general(need_g)
            if site:
                cell, price = site
                self.last_purpose = "castle"
                return {"mode": "march", "purpose": "castle_g", "stack": g, "site": cell,
                        "price": price, "turn": t}
        if (P["castle_start"] <= t <= P["castle_stop"] and len(self.my_castles) < self.n_castles_wanted()
                and (self.last_purpose != "castle" or self.my_army < P["attack_min_army"])):
            site = self.castle_site()
            if site:
                cell, price = site
                self.last_purpose = "castle"
                return {"mode": "gather", "purpose": "castle", "root": cell, "site": cell,
                        "price": price, "turn": t, "budget": P["castle_gather_budget"]}
        tgt = self.choose_target()
        if tgt < 0:
            return None
        self.last_purpose = "attack"
        de = self.bfs([tgt])
        own = [i for i in range(self.n) if O[i] == 1 and i != g]
        root = g
        budget = P["gather_budget"]
        if own and (P["attack_root_front"] or (P["kill_front"] and self.egen >= 0 and tgt == self.egen)):
            root = min(own, key=lambda i: (de[i], -A[i]))
            if self.egen >= 0:
                budget = P["kill_gather_budget"]
        return {"mode": "gather", "purpose": "attack", "root": root, "turn": t,
                "budget": budget, "target": tgt, "moves": 0}

    def bfs_path_own(self, src, dst):
        O = self.O
        prev = {src: -1}
        dq = deque([src])
        while dq:
            i = dq.popleft()
            if i == dst:
                break
            for j, _ in self.nb[i]:
                if j not in prev and O[j] == 1:
                    prev[j] = i
                    dq.append(j)
        if dst not in prev:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        path.reverse()
        return path

    # ---------------------------------------------------------------- attack
    def try_kill(self):
        eg = self.egen
        if eg < 0:
            return None
        A, O = self.A, self.O
        t = self.turn
        gen_army = A[eg] if self.T[eg] == 4 else self.last_army[eg] + (t - self.last_seen[eg]) // 2
        stacks = [i for i in range(self.n) if O[i] == 1 and A[i] >= 2]
        if not stacks:
            return None
        best = None
        sweep = PARAMS["x_sweep"] and t < 800
        for i in sorted(stacks, key=lambda k: -A[k])[:(PARAMS["sweep_cands"] if sweep else 4)]:
            if self.late(0.5):
                break
            is_g = i == self.general
            send = A[i] // 2 if is_g else A[i] - 1
            if is_g and A[i] - send < getattr(self, "need_g", 2):
                continue
            path = self.path_to(i, eg)
            if not path:
                continue
            ks = PARAMS["x_kill_stealth"]
            if ks and len(path) > ks:
                p2 = self.path_to(i, eg, stealth=True)
                if p2 and len(p2) <= len(path) + PARAMS["kill_stealth_extra"]:
                    path = p2
            if sweep:
                # exact walk: own cells on the way join the stack, one army stays behind per step
                carried = send
                for j in path[1:-1]:
                    if O[j] == 1:
                        carried += A[j]
                    elif carried <= A[j]:
                        carried = -INF
                        break
                    else:
                        carried -= A[j]
                    carried -= 1
                send = carried
                need = (gen_army + PARAMS["kill_margin"] + (len(path) // 2) + 1) * self.mod["kill"]
                if PARAMS["x_stage"]:
                    need += max([A[z] - 1 for z, _ in self.nb[eg] if O[z] == 2] + [0])
                if t >= 800:
                    need = 1
            else:
                cost = sum(A[j] for j in path[1:-1] if O[j] != 1) + len(path)
                need = (gen_army + cost + PARAMS["kill_margin"] + (len(path) // 2)) * self.mod["kill"]
                if t >= 800:
                    need = cost + 2
            if send >= need:
                if best is None or len(path) < best[1]:
                    best = (i, len(path), path, 1 if is_g else 0)
        if best:
            self.last_label = "kill"
            return self.mv(best[2][0], best[2][1], best[3])
        return None

    # --------------------------------------------------------------- endgame
    def dt_guard(self):
        """Deathtouch-era defence of our general (turn >= dt_guard_turn).

        From turn 800 any valid move onto our general wins, so army on the general is useless;
        what matters is that no enemy stack (>= 2) is adjacent to it when it moves.
        1. enemy adjacent: chase its cell from a third tile with A[j] >= A[e] (chasing moves go
           first and leave it <= 1, so its touch is invalid); before 800 the general may hit it too.
        2. enemy two steps away (next to a neighbour c of the general) that can take c: kill it,
           or reinforce c so it holds (reinforcing resolves before the attack)."""
        t = self.turn
        A, O = self.A, self.O
        g = self.general
        nbg = [c for c, _ in self.nb[g] if self.pas[c]]
        adj = [e for e in nbg if O[e] == 2 and A[e] >= 2]
        best, bk = None, None
        for e in adj:
            for j, _ in self.nb[e]:
                if O[j] != 1 or A[j] < 2:
                    continue
                if j == g:
                    if t >= 800 or A[g] - 1 <= A[e]:
                        continue
                elif A[j] < A[e]:
                    continue
                k = (A[e], A[j] - 1 > A[e], j != g, A[j])
                if bk is None or k > bk:
                    bk, best = k, (j, e)
        if best:
            self.last_label = "dt_chase"
            return self.mv(*best)
        threats = []
        for c in nbg:
            for e, _ in self.nb[c]:
                if e == g or O[e] != 2 or A[e] < 2:
                    continue
                arrive = A[e] - 1
                if arrive <= A[c]:
                    continue
                threats.append((arrive - A[c], c, e))
        if not threats:
            return None
        threats.sort(reverse=True)
        for deficit, c, e in threats:
            # (a) kill the threat outright (from c this is chasing if it moves into c: we go first)
            for j, _ in self.nb[e]:
                if O[j] == 1 and j != g and A[j] - 1 > A[e]:
                    self.last_label = "dt_kill"
                    return self.mv(j, e)
            # (b) make c hold: reinforce it from an own neighbour (general last, it is the fallback)
            if O[c] == 1:
                srcs = [j for j, _ in self.nb[c] if O[j] == 1 and j != e and j != g and A[j] - 1 >= deficit]
                if srcs:
                    j = min(srcs, key=lambda k: A[k])
                    self.last_label = "dt_block"
                    return self.mv(j, c)
                if A[g] - 1 >= deficit:
                    split = 1 if (t < 800 and A[g] // 2 >= deficit) else 0
                    self.last_label = "dt_block"
                    return self.mv(g, c, split)
            else:
                # c is not ours: take it with enough to keep it after the enemy walks in
                send = A[g] - 1
                if send > A[c] + A[e]:
                    self.last_label = "dt_block"
                    return self.mv(g, c, 0)
        return None

    def endgame(self):
        t = self.turn
        A, O = self.A, self.O
        g = self.general
        # clear enemy cells near our general
        best, bk = None, None
        for i in range(self.n):
            if O[i] == 2 and self.manh(i, g) <= 2:
                for j, _ in self.nb[i]:
                    if O[j] == 1 and A[j] - 1 > A[i] and j != g:
                        k = (-self.manh(i, g), A[i])
                        if bk is None or k > bk:
                            bk, best = k, (j, i)
        if best:
            return self.mv(*best)
        # own all orthogonal neighbours of the general
        for j, _ in self.nb[g]:
            if self.pas[j] and O[j] != 1:
                if A[g] - 1 > A[j]:
                    return self.mv(g, j, 0 if A[g] < 6 else 1)
        # deathtouch offence: stack adjacent to the enemy general at >= 800
        eg = self.egen
        if eg >= 0 and t >= PARAMS["dt_stage_turn"]:
            own = [i for i in range(self.n) if O[i] == 1 and A[i] >= 3]
            if own:
                de = self.bfs([eg])
                i = min(own, key=lambda k: (de[k], -A[k]))
                if de[i] > 1:
                    path = self.path_to(i, eg)
                    if path and len(path) >= 2:
                        nxt = path[1]
                        if nxt != eg and (O[nxt] == 1 or A[i] - 1 > A[nxt]):
                            return self.mv(i, nxt)
        return None


_BOT = None
_LAST_TURN = None


_ALIASES = {
    "turn": ("turn", "timestep", "time", "t", "tick"),
    "height": ("height", "H", "h", "rows", "n_rows"),
    "width": ("width", "W", "w", "cols", "n_cols"),
    "player_id": ("player_id", "player", "pid", "id", "player_index"),
    "my_land": ("my_land", "owned_land_count", "land"),
    "my_army": ("my_army", "owned_army_count"),
    "opp_land": ("opp_land", "opponent_land_count", "enemy_land"),
    "opp_army": ("opp_army", "opponent_army_count", "enemy_army"),
    "type": ("type", "types", "type_grid", "terrain", "tiles", "cell_type"),
    "owner": ("owner", "owners", "owner_grid", "ownership"),
    "army": ("army", "armies", "army_grid"),
}


def _get(obs, names):
    for k in names:
        if isinstance(obs, dict):
            if k in obs:
                return obs[k]
        elif hasattr(obs, k):
            return getattr(obs, k)
    return None


def _normalize(obs):
    """Canonical observation dict regardless of the adapter's exact spelling."""
    out = {}
    for key, names in _ALIASES.items():
        out[key] = _get(obs, names)
    for g in ("type", "owner", "army"):
        v = out[g]
        if v is not None and hasattr(v, "tolist"):
            out[g] = v.tolist()
    grid = out["type"]
    nested = grid is not None and len(grid) > 0 and isinstance(grid[0], (list, tuple))
    if out["height"] is None and nested:
        out["height"] = len(grid)
    if out["width"] is None and nested:
        out["width"] = len(grid[0])
    for k in ("turn", "height", "width", "player_id", "my_land", "my_army", "opp_land", "opp_army"):
        if out[k] is not None:
            out[k] = int(out[k])
    if out["my_land"] is None or out["my_army"] is None:
        own = [v for row in out["owner"] for v in row] if nested else list(out["owner"])
        arm = [v for row in out["army"] for v in row] if nested else list(out["army"])
        out["my_land"] = sum(1 for o in own if int(o) == 1)
        out["my_army"] = sum(int(a) for o, a in zip(own, arm) if int(o) == 1)
    for k in ("opp_land", "opp_army", "player_id", "turn"):
        if out[k] is None:
            out[k] = 0
    return out


def _decide(obs):
    global _BOT, _LAST_TURN
    obs = _normalize(obs)
    t = int(obs["turn"])
    H, W = int(obs["height"]), int(obs["width"])
    if (_BOT is None or _LAST_TURN is None or t <= _LAST_TURN or t == 0
            or _BOT.H != H or _BOT.W != W):
        _BOT = Bot(obs)
    _LAST_TURN = t
    b = _BOT
    b.t_start = time.perf_counter()
    b.budget_s = (PARAMS["first_budget_ms"] if t == 0 else PARAMS["soft_budget_ms"]) / 1000.0
    T, O, A = b.parse(obs)
    b.turn = t
    if b.general < 0:
        for i in range(b.n):
            if T[i] == 4 and O[i] == 1:
                b.general = i
                break
        if b.general < 0:
            return PASS
        b.setup(T)
        gc.collect()
        gc.freeze()
    b.update(obs, T, O, A)
    a = b.decide()
    return a


def _sanitize(a):
    try:
        k = int(a[0])
        if k not in (0, 1, 2):
            return [1, 0, 0, 0, 0]
        r, c, d, s = int(a[1]), int(a[2]), int(a[3]), int(a[4])
        if k == 1:
            return [1, 0, 0, 0, 0]
        if not (0 <= d <= 3):
            d = 0
        if s not in (0, 1):
            s = 0
        if _BOT is not None:
            r = min(max(r, 0), _BOT.H - 1)
            c = min(max(c, 0), _BOT.W - 1)
        return [k, r, c, d, s]
    except Exception:
        return [1, 0, 0, 0, 0]


def act(observation):
    try:
        a = _sanitize(_decide(observation))
        if _BOT is not None:
            _BOT.last_action = a
        return a
    except Exception:
        if DEBUG:
            raise
        try:
            if _BOT is not None:
                _BOT.err += 1
                _BOT.last_action = [1, 0, 0, 0, 0]
        except Exception:
            pass
        return [1, 0, 0, 0, 0]
