# Playing Jedi Academy with Curious A3C

Six neural networks combine to make the AI.

The AI explores half of Tattooine in half an hour, defeats Jedi Trainer 1-0 despite starting at a disadvantage, defeats Cultist 3-1, and defeats the final boss of the game Desann 20-11. 

The AI employs curiosity to efficiently learn to do all this in mere hours of training on one modest 8GB GPU.

<img width="1470" alt="Screenshot 2568-05-10 at 00 08 29" src="https://github.com/user-attachments/assets/26a04bc4-7ca3-412b-a8fc-65e8dce597f7" />

https://www.youtube.com/watch?v=folA8Rqyi20

https://www.youtube.com/watch?v=luNo1oCSoHw

https://www.youtube.com/watch?v=9vLv0ws3hoU

## Rewards

```py
r_curiosity = 100 * ( error_future_predictor - error_action_predictor )
r_momentum = 0 * ( momentum )
r_damage = ( damage_dealt - prev_damage_dealt ) + 0.5 * ( health - prev_health ) + 0.5 * ( shield - prev_shield )
r_score = 100 * ( ( score - prev_score ) - 0.5 * ( deaths - prev_deaths ) )
r_baseline = 1800
reward = r_curiosity + r_momentum + r_damage + r_score + r_baseline
```




