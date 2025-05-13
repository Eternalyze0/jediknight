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

## My Discord Musings

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

## Reference

Dmitri Tkatch. Playing Jedi Academy with Deep Curiosity. 2025.



