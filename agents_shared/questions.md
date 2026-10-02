# Open strategic / training questions (append; ✅ = answered, with pointer)

1. Does playing against older versions of ourselves (league / prioritized fictitious self-play) beat
   tuning against a fixed pool? How should the pool be weighted?
2. Should we include exact self-play (current vs current) in tuning fitness? Risk of cycles?
3. Which external bots (heuristic / RL / game-theory) exist that we can run locally as sparring partners?
4. Which top generals.bot competitors published code or write-ups after the event (ResBot, nanomena, Kubic, FreeLunch...)?
5. How many games per comparison are needed to trust a difference (SPRT / Elo CIs)?
6. Does our bot overfit to the tuning pool? Measure on held-out opponents (BC clones of different bots).
7. Are generic game libraries (OpenSpiel, PettingZoo, gymnasium envs) useful here, or only the pinned engine?

(training-research, 2026-10-02) ✅ Q1, Q2, Q5, Q6, Q7 answered in agents_shared/training-research.md; Q3/Q4 partially (public RL repos: quant-eagle/generals-competition-rl-bot MIT but no weights; Amin-Debabeche fork; strakam/AverageJoe).
8. How much does a re-evaluated promotion (top-3 on fresh seeds, 100 games) change the final strength vs shipping raw best-of-generation? (measure on held-out)
9. Are late-game (turn >= 800) deathtouch situations really out of distribution for RL entrants? Check ResBot replays: share of games reaching 800 and who wins them.
10. Does an exploiter run (copy of our bot tuned only vs our current mean for 1 h) find holes that the main pool misses (>0.65 vs main)?
11. Could a Nash/maximin LP over the (candidate x pool) payoff matrix give better pool weights than PFSP p=1?
