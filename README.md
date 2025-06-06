# ⚔️ Playing Jedi Academy with Deep Curiosity ⚔️

⚔️ Six neural networks combine to make the AI.

⚔️ The AI explores half of Tattooine in half an hour, defeats Jedi Trainer 1-0 despite starting at a disadvantage, defeats Cultist 3-1, and defeats the final boss of the game Desann 20-11. 

⚔️ The AI employs curiosity to efficiently learn to do all this in mere hours of training on one modest 8GB GPU.

<img width="1470" alt="Screenshot 2568-05-10 at 00 08 29" src="https://github.com/user-attachments/assets/26a04bc4-7ca3-412b-a8fc-65e8dce597f7" />


⚔️ https://www.youtube.com/watch?v=folA8Rqyi20

⚔️ https://www.youtube.com/watch?v=luNo1oCSoHw

⚔️ https://www.youtube.com/watch?v=9vLv0ws3hoU

## Rewards

```py
r_curiosity = 100 * ( error_future_predictor - error_action_predictor )
r_momentum = 0 * ( momentum )
r_damage = ( damage_dealt - prev_damage_dealt ) + 0.5 * ( health - prev_health ) + 0.5 * ( shield - prev_shield )
r_score = 100 * ( ( score - prev_score ) - 0.5 * ( deaths - prev_deaths ) )
r_baseline = 1800
reward = r_curiosity + r_momentum + r_damage + r_score + r_baseline
```

Note momentum was experimented with early on but was eventually ignored as using OCR to scrape the on-screen momentum indicator wasted too many agent time-steps.

## Explanation

It's mostly explained in the paper -- knight.pdf, but I will explain here in more practical words anyway. I use the A3C program from minimalRL [3] but I don't use the asynchronous part -- the agent learns everything online in realtime with one learner process and no test processes. I re-implement the intrinsic curiosity module from [1] and I use the AI<->game interface from [2]. The implementation is mostly game agnostic (or can easily be made so) however a colleague was kind enough to create a custom game client for me so that things like health and shield information could be recovered. Thus the final version of the AI is dual-input in that it uses raw pixels as well as raw game data. 

The only preprocessing exception is for the variables used in the reward structure above. Otherwise game data is simply all cast to float and piped into an embedding neural module. **We note that exploratory performance is maintained even if only raw pixels are used as input.** Furthermore we believe the raw game data has enough information to play the game alone as well, albeit it does not have map information. Most likely the most useful part of the raw game data are the coordinates and orientations of the so-called entities, which include player characters. In the program the raw game data is referred to as the snapshot. The game itself is now open source and is available here: https://github.com/JACoders/OpenJK. Note that actual saber collisions are computed within the game program.

Neural Networks:

- A3C
- Forward Model
- Action Prediction
- State Embedder
- Snapshot Embedder
- Image Embedder

The forward, action prediction, and state embedder models (referred to as Phi in [1]) are part of the curiosity module from [1]. The snapshot and screenshot embedders are intended to alleviate the issue addressed in [5], i.e. the so-called couch-potato effect where a distracting TV with random noise in the environment derails the AI from truly exploring. All 5 of these are essentially trained through supervised learning. The agent is A3C but this can be swapped in with any other RL algorithm as one would an optimizer.

Aside from this being to my knowledge the first application of curiosity to a realtime competitive 3D game, the other novelty comes from smoothing actions by rewarding negative action prediction error.

## Results

In 36 minutes the AI explores half (about 10 areas or "rooms") of a big free-for-all map meant for 32 players. During this time it picks up the same health pack twice (after it regenerates in a minute) and the same shield pack twice. 

It wins 3-1, 1-0, 20-11 against Cultist, Jedi Trainer (after starting at a disadvantage) and Desann (the final boss of the game), all set to max Jedi Master level.

Actions appear smoother and less jittery when the novel reward structure is used.

Human gaming professionals report the AI is more challenging and more interesting to play against than built-in bots.

## My Discord Musings on Future Directions

alpha — Yesterday at 23:49

the transition errors promote exploring the unknown and prevents the ai from getting stuck (which is huge, ai's without curiosity always get stuck in my experience), rewarding negative action error promotes smooth actions and acts as a sort of difficulty setting -- reminiscent of schmidhuber's optimal learning progression theory, ie you want to be learning stuff at your level, not too difficult and not too easy

alpha — 02:59

i think there's still a lot of work to be done in balancing extrinsic rewards with intrinsic, and balancing the two intrinsic rewards, as well as exploring other aspects of the game, eg force powers, other weapons, the single player campaign, team deathmatch, as well as applying it to other 3d games
ie im thinking a series of papers could come from this if things work out well in the future 
even just starting off with gamma = 0 and increasing over time is interesting to me
similarly i think starting off with coefficient of action error = 0 and increasing over time is a good idea

alpha — 03:10

i think ideally it self-balances somehow over time, ie if one of the intrinsic errors is an order of magnitude higher than the other then the smaller one has its coefficient doubled
i also had some wild ideas about recursive curiosity, where curiosity_n+1 acts on curiosity_n at higher timescales, but i don't know if they will have any practical benefit amidst all the noise that is rl
another wild approach is only using rl initially to boostrap into an sl situation, since the forward model is trained by sl already

_**i think in general the key to making rl efficient is to delegate as much of it to sl as possible**_

so for example if there is a differentiable approximation of the environment like the forward model, then the whole rl process becomes differentiable

alpha — 03:19

and since the forward model is sl and since curiosity doesn't get stuck, you end up training a good forward model that doesn't miss any parts of the distribution, so initially all you're really doing is training an sl forward model, with a bit of rl, and then afterwards you can mostly train in the differentiable regime once the forward model is good enough
hope im making sense here

##  References

We thank the authors of these references for their dedication as well as friends and coworkers who helped with interfacing with the game and editing the paper,

[1] Deepak Pathak, Pulkit Agrawal, Alexei A. Efros and Trevor Darrell. Curiosity-driven Exploration by Self-supervised Prediction. In ICML 2017.

[2] Tim Pearce, Jun Zhu. Counter-Strike Deathmatch with Large-Scale Behavioural Cloning, In proc. of IEEE Conference on Games (CoG), Beijing, China, 2022.

[3] Seungeun Rho. A3C. GitHub, 2019.

[4] Volodymyr Mnih, Adria Puigdomenech Badia, Mehdi Mirza, Alex Graves, Timothy P. Lillicrap, Tim Harley, David Silver, Koray Kavukcuoglu. Asynchronous Methods for Deep Reinforcement Learning. arXiv, 2016.

[5] Yuri Burda, Harrison Edwards, Amos Storkey, Oleg Klimov. Exploration by Random Network Distillation. arXiv, 2018.

[6] Richard Sutton, Andrew Barto. Reinforcement Learning: An Introduction. 2018.

## Refer To As

Dmitri Tkatch. Playing Jedi Academy with Deep Curiosity. GitHub, 2025.



