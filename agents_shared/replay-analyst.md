# replay-analyst findings
Code: tools/replay_an/{extract,analyze,kill}.py ; raw: data/replay_an.jsonl (330 player-games per bot), data/replay_an_out.txt (full tables), data/replay_kill.json.
Sample: first 330 games per bot in acts.jsonl (no forfeits). Win rates in sample: ResBot .93, nanomena .85, Kubic .80, bca .04, Chig .07. Median game length ~430 turns, 97-100% of wins before turn 800 (general capture; no deathtouch/draw wins seen in sample).

## 1 Economy (median land / army / castles / gen army)
t=50 all bots identical: 25 land, 51 army, gen 3. t=100: ~50 land, ~105 army, gen 6, 0 castles. t=150: 61 / 147 / 1 / 11. t=200: 67 / 182 / 2 / 12. Top bots and weak bots are INDISTINGUISHABLE through t=200 (Chig even higher: 70 land, 193 army). Divergence starts t>=300: ResBot 76 land / 259 army vs opp 71 / 240; t=400 ResBot 82/335 vs 72/268 (survivorship partly); nanomena/Kubic ~75/295 vs opp 74/260. bca/Chig at 400-500 are BEHIND (69-78 land vs opp 81-86).
General army stays small the whole game: median 11-16 at t>=150 (IQR ~8-25) for every bot, i.e. ~5% of total army. Nobody garrisons the general.
Rule: opening is not where games are won (all bots reach 50 land by t=100); do not over-tune it. Match ~61 land/147 army @150, ~67/182 @200.

## 2 Castles
Count: ~2.9-3.2 built per game; 88-91% of games build >=1. 1st castle at turn ~123 (IQR 121-160) = right after the t=99 land bonus gives a 35-army cell; 2nd ~167; 3rd ~212. Price paid: median 35 (= base, zero proximity penalty); 59-65% of castles cost exactly 35.
Position: median Manhattan distance 7 from own general (52% exactly 7, which is the free-ring: 14-2d=0 at d=7); only 4-7% within 2 of general. Distance to nearest enemy-owned cell at build: median 5-6 (IQR 3-9), 34-39% within 4 -> castles are built on the front, not in the back. Cell army at build ~35-37 (they build the moment 35 is available).
Loss: top bots lose 33-35% of castles before game end (median lost turn ~300, median lifetime 71-88 turns); weak bots lose 51-52% (life 110-130). Lost castles were built nearer the enemy (d_enemy 5 vs 6-7 kept). (Raw 'lost' includes the final capture flip; numbers above exclude last 3 ticks.)
Rule: build castle #1 at the first turn >=100 where a cell at Manhattan dist exactly 7 from the general (price 35) on the enemy-facing side has >=35 army; then #2 ~t165, #3 ~t210; keep building ~1 per 50 turns afterward (build rate ~14 per 1000 moves in t100-300, ~5 in t300-400, ~2 later); cap ~5-6 at t600.

## 3 Strikes (stack>=30 and >=25% of army moving toward enemy general; chained while stack moves on)
~8-10 strikes/game for everyone (this is just normal army marching - a frequent, not a rare event). Median start turn ~300-325, initial stack 64-73, peak 70-77 (30% of own army exactly, IQR .3-.4), length 4-5 moves, start distance to enemy general 15-16. Peak stack is ~7.5x enemy general army (median egen 9-10), 0.3-0.4x enemy total army.
Success per strike: general captured 6.5-8.7% (top bots) vs 0.2-0.5% (bca/Chig, same size strikes!); castle captured 17-20% vs 7-12%; >=5 cells flipped 17-20% vs 11-16%. Strikes that kill the general: peak stack 91-115 (IQR 64-156) at turn ~390-440 against egen ~11. 
=> same stack size, 15x the kill rate: difference is target choice/path (straight shot at the general) not size.
Rule: launch kill stacks at >= ~30% of army (target 70-100 absolute) in a straight line at the general and keep it as ONE unsplit stack.

## 4 Defense (enemy stack>=30 enters dist<=6 of my general; ~1.4-1.6 events/game top bots; 3.5-4.4 for bca/Chig)
Response in next 10 turns (fraction of moves): gather toward general 29-34%, attack enemy-owned cells 35-36%, other 30-34%, pass 0%. ResBot/nanomena/Kubic do NOT reinforce the general: gen+10 turns is only +5 (natural growth). Weak bots gather more (41-49%) and attack less (25-32%) yet lose more.
Loss within 30 turns of event: ResBot 4.9%, nanomena 11%, Kubic 13%, Chig 32%, bca 30%. Strong dependence on garrison within radius 3 of general (gen + neighbors) / stack: <0.5 -> loss 7% (ResBot) / 21-26% (nan/Kub) / 45-54% (bca/Chig); >=1 -> 0-5% top / 17-24% weak. Median gen army at event 11-12 vs stack 30-60; radius-3 garrison ~0.8-0.9 of the stack.
Key: own/enemy total army at event: ResBot 1.0, nan/Kub 0.9, bca/Chig 0.7-0.8 -> weak bots are simply behind when attacked; and counter-attacking (attack share 35%) is the top bot style.
Rule: when enemy stack>=30 within 6 of general, keep garrison (general + r<=3 cells) >= ~1x stack; about a third of moves gather, a third counterattack the enemy-owned cells behind the stack; never pass.

## 5 Land bonus (moves in 10 ticks after bonus)
After t=49: neutral 44-85% (ResBot 69, Kubic 68, nanomena 44), own-gather 15-56%. After t=99: neutral 10-19%, own 80-90%. t>=199: own (gather/redistribution) 84-95%, enemy-capture 4% (t200) -> 8% (t300) -> 10-14% (t400), neutral ~1%, build 1-2%, pass ~0. No special post-bonus burst behaviour; bonus turns are not exploited with attacks.
Rule: after t~100 land is nearly saturated; 90% of moves are internal gathering moves, so efficient gathering (stack consolidation along a path) dominates.

## 6 How games are decided
Winner takes the general before turn 800 in 97-100% of games (no deathtouch/draw outcomes in sample). Median decisive turn: ResBot 446 (IQR 350-532), nanomena 404, Kubic 431; bca/Chig wins ~200-280.
T-50 before capture (winner): land ratio 1.2 (ResBot) / 1.0-1.1 (nan, Kub); ARMY ratio 1.4 / 1.2 / 1.3 (IQR 1.2-1.5); loser's T-50 army ratio .8-.9. Winner total army ~270-330 vs loser 180-210 at the kill. Winner general army 10-15 vs loser's 9-13: general garrison is irrelevant to outcome.
P(win | land lead @100): ResBot .96 vs .90 else, nanomena .91 vs .79, Kubic .91 vs .69; army lead @300: .99 vs .88 / .89 vs .75 / .92 vs .73. Early lead matters but only weakly for the top bot (it comes back).
The kill itself (200 winning games per bot): winner's biggest cell within 8 of enemy general is only 5-13 (median) at T-20/T-10, then a stack of 60-100 appears within 8 cells at T-6 (ResBot 100, nan 73, Kub 70) and 71/54/45 on the cell adjacent at T-1. Stack at T-1 / enemy general army = 4.2x / 3.1x / 3.0x (median); 50-68% of kills use >=3x the general. Enemy has 37-50 army within 3 cells of its general and 180-210 total yet cannot intercept: kills are a single fast ~6-turn run of an unsplit stack.
Rule: the win condition is a one-shot run (<=6-8 turns, one stack ~25-30% of total army ~ 3-5x the target general) when own army ratio >=1.2-1.3; defender cannot stop it.

## 7 Other
- Passes: ResBot/nanomena/Kubic pass 3-6% in turns 0-99 (the first ~20 turns, wait for army), then ~0%. bca passes 24.5% of the first 100 turns (wasteful) yet reaches the same land.
- Splits: very rare - 1-3% of moves (Kubic 7% in first 100 turns, ~2-3% later; ResBot ~1.5%; weak bots 0.4-0.7%). Top bots use splits slightly more than weak ones; splits are for opening expansion & peel-off, not the main mechanism.
- Build use ~1-1.5% of moves in t100-300; the top 3 bots keep building to t500+ (ResBot 4-5, nanomena 5-6 castles by t500-600 vs bca 2-3, Chig 3).
- Castle adjacent to general (d<=2) only 4-7%: avoid, it pays penalty up to 12.
