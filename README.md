# B4 : Bodega Brawl Black Belt

This package provides code to solve the card game [Bodega Brawl](https://www.bodegabrawl.com/).  This is not a particularly complex or difficult game, but I just finished a PhD, so I finally had time for a fun experiment over the holidays.

## Game Rules
The official rules can be found at the [publisher's website](https://www.bodegabrawl.com/).  This game is designed to simulate a martial arts duel.  In it, each player starts with a hand of 14 cards and uses them to attack their opponent, or defend from their opponent's attacks.  Each turn, both players choose a card and reveal it simultaneously, then resolves the cards effects based on the selected combination.  Each card attacks or defends against one of three regions: head, body or legs.  Some cards can only be used to attack the specified region, while others can be used to either attack or defend.

If both players attack the same region simultaneously, the attacks cancel each other out and both are discarded.  If one player attacks a region and the other player does not defend that region, the attack is registered as a strike against the attacked region.  On the other hand if the region was defended, the attack is discarded, and the defender registers a strike against the attacker in that region instead.  If both players attack different regions, then both are registered as a strike.  The game ends when either player has been struck in the head twice, the body three times, the legs four times, or has suffered a combined total of five strikes to any region.

The normal rules state if both players are knocked out simultaneously, the players continue until one lands a successful strike against the other in sudden death, but our implementation instead terminates in a draw.

## Solving the Game
Solving this game is accomplished by computing the Nash Equilibrium for each state in the game, starting with those nearest the end of the game and working backward.  The Nash Equilibria are computed using [linear programming](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linprog.html) via scipy.  The main difficulty is simply churning through all possible game states.

Naively, each player could be holding a subset of 14 cards, and could have zero or one head strikes against them, zero to two body strikes against them and zero to three leg strikes against them.  There are `2^14 = 16386` possible subsets of 14 cards.  Meanwhile, the numbers of possible strikes against a player is `2x3x4=24`.  Combining this for two players result in `(16384 x 24)^2 = 154618822656` possible states.

The first thing we can do to reduce this is to realize that four of the possible hit states are not feasible because they would knock out a player due to containing five or more maximum strikes.  This cuts the total states to `(16384 x 20)^2 = 107374182400`.

Next we realize that each player must have the same number of cards in their hand as their opponent.  If we sort the number of card states into the number of cards it contains we find the following:
```
 1 cards : 6   1-player card states -> 6^2 = 36      2-player card states
 2 cards : 18  1-player card states -> 18^2 = 342    2-player card states
 3 cards : 38  1-player card states -> 38^2 = 1444   2-player card states
 4 cards : 65  1-player card states -> 65^2 = 4225   2-player card states
 5 cards : 94  1-player card states -> 94^2 = 8836   2-player card states
 6 cards : 116 1-player card states -> 116^2 = 13456 2-player card states
 7 cards : 124 1-player card states -> 124^2 = 15376 2-player card states
 8 cards : 116 1-player card states -> 116^2 = 13456 2-player card states
 9 cards : 94  1-player card states -> 94^2 = 8836   2-player card states
10 cards : 65  1-player card states -> 65^2 = 4225   2-player card states
11 cards : 38  1-player card states -> 38^2 = 1444   2-player card states
12 cards : 18  1-player card states -> 18^2 = 342    2-player card states
13 cards : 6   1-player card states -> 6^2 = 36      2-player card states
14 cards : 1   1-player card state  -> 1^2 = 1       2-player card state
```
Combining these with the 20 possible hit states results in:
```
(36+324+1444+4225+8836+13456+15376+13456+8836+4225+1444+324+36+1) x 20^2 = 28807600
```
There are further optimizations you can do, but 28.8 million seems like a manageable number, so we stop here and solve all of these states.

## Installation
First use pip to install the package by running:
```
cd b4
pip install -e .
```
Then to solve the game, run:
```
b4_solve --num-procs N
```
where N is the number of parallel processes you want to use.  I found that with 40 processes, the game can be solved in around 30-40 minutes.  This script will create a `solutions` directory and save a file `large_final.pkl` which is around 2.2 GB and contains the optimal policy and values for each game state.

To play against the computer, run:
```
b4_play
```
This will use a text-based interface to keep track of the state and tell you the opponent's actions.  The `--drive` flag can be used to pilot the agent against another human when playing with actual cards.  The `--verbose` flag will show the agent's action probabilities as well as suggest optimal action probabilities for the human player.

## Results
I think it works?  It beat me 10 wins to 8 losses and 2 draws.  It beat my Mom 19 wins to 13 losses and 5 draws.  It beats a random player 64% of the time over 100000 games.  I think these results speak to the inherent randomness and lack of skill required to play this game, but we are consistently better than two specific humans and a random player, so hey, that's something.
