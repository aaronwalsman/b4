from collections import namedtuple

'''
This module implements all the functionality of the Bodega Brawl game.
The game_mode constant was for debugging purposes, and allowed me to build
smaller versions of the game with different win conditions and starting numbers
of cards.  The 'large' setting is for the default game.
'''

game_mode = 'large' # 'large' 'medium' 'small' or 'tiny'

if game_mode == 'large':
    max_head_hits = 2
    max_body_hits = 3
    max_legs_hits = 4
    max_total_hits = 5
elif game_mode == 'medium':
    max_head_hits = 2
    max_body_hits = 3
    max_legs_hits = 3
    max_total_hits = 4
elif game_mode == 'small':
    max_head_hits = 2
    max_body_hits = 2
    max_legs_hits = 2
    max_total_hits = 3
elif game_mode == 'tiny':
    max_head_hits = 1
    max_body_hits = 2
    max_legs_hits = 2
    max_total_hits = 2

if game_mode == 'large':
    start_head_a = 1
    start_head_ac = 3
    start_body_a = 1
    start_body_ac = 4
    start_legs_a = 1
    start_legs_ac = 4
elif game_mode == 'medium':
    start_head_a = 1
    start_head_ac = 2
    start_body_a = 1
    start_body_ac = 2
    start_legs_a = 1
    start_legs_ac = 2
elif game_mode == 'small':
    start_head_a = 0
    start_head_ac = 2
    start_body_a = 0
    start_body_ac = 2
    start_legs_a = 0
    start_legs_ac = 2
elif game_mode == 'tiny':
    start_head_a = 0
    start_head_ac = 1
    start_body_a = 0
    start_body_ac = 2
    start_legs_a = 0
    start_legs_ac = 0

class Action(namedtuple('Action', ('region', 'card', 'mode'))):
    '''
    Action representation.  Contains three components:
    [0] "region" (head,body,legs)
    [1] "card" (attack,attack/counter)
    [2] "mode" (attack,counter)
    The counter mode can only be used with an attack/counter card.
    '''
    def card_name(self):
        if self.card == 'attack':
            c = 'a'
        elif self.card == 'attack/counter':
            c = 'ac'
        else:
            raise ValueError(
                '"card" must be either "attack" or "attack/counter"')
        return '%s_%s'%(self.region, c)
    
    def __str__(self):
        return ('%s_%s_%s'%self).upper()
    
    def __int__(self):
        i = 0
        if self.region == 'body':
            i += 3
        elif self.region == 'legs':
            i += 6
        if self.card == 'attack/counter':
            if self.mode == 'attack':
                i += 1
            elif self.mode == 'counter':
                i += 2
        return i

# instantiate all possible actions
actions = {
    'HEAD_A_A' : Action('head', 'attack', 'attack'),
    'HEAD_AC_A' : Action('head', 'attack/counter', 'attack'),
    'HEAD_AC_C' : Action('head', 'attack/counter', 'counter'),
    'BODY_A_A' : Action('body', 'attack', 'attack'),
    'BODY_AC_A' : Action('body', 'attack/counter', 'attack'),
    'BODY_AC_C' : Action('body', 'attack/counter', 'counter'),
    'LEGS_A_A' : Action('legs', 'attack', 'attack'),
    'LEGS_AC_A' : Action('legs', 'attack/counter', 'attack'),
    'LEGS_AC_C' : Action('legs', 'attack/counter', 'counter'),
}

action_order = tuple([a for a in actions.values()])
assert tuple([int(a) for a in action_order]) == (0,1,2,3,4,5,6,7,8)

class HitState(namedtuple(
    'HitState',
    ('head', 'body', 'legs'),
    defaults=(0,0,0),
)):
    '''
    Defines how many times each body part has been hit by the opponent.
    '''
    @property
    def total(self):
        return sum(self)
    
    @property
    def is_dead(self):
        return (
            self.head >= max_head_hits or
            self.body >= max_body_hits or
            self.legs >= max_legs_hits or
            self.total >= max_total_hits
        )
    
    def transition(self, actions):
        action, opponent_action = actions
        region = opponent_action.region
        mode = opponent_action.mode
        if region == action.region:
            if mode == action.mode:
                return self
            elif action.mode == 'attack' and mode == 'counter':
                update = {region : getattr(self, region) + 1}
                return self._replace(**update)
            elif action.mode == 'counter' and mode == 'attack':
                return self
        
        else:
            if mode == 'attack':
                update = {region : getattr(self, region) + 1}
                return self._replace(**update)
            else:
                return self
    
    @property
    def terminal(self):
        return self.is_dead

class CardState(namedtuple(
    'CardState',
    ('head_a', 'head_ac', 'body_a', 'body_ac', 'legs_a', 'legs_ac'),
    defaults=(
        start_head_a,
        start_head_ac,
        start_body_a,
        start_body_ac,
        start_legs_a,
        start_legs_ac,
    ),
)):
    '''
    Defines how many of each card type a player has left.
    '''
    @property
    def action_space(self):
        action_space = []
        if self.head_a:
            action_space.append(actions['HEAD_A_A'])
        if self.head_ac:
            action_space.append(actions['HEAD_AC_A'])
            action_space.append(actions['HEAD_AC_C'])
        if self.body_a:
            action_space.append(actions['BODY_A_A'])
        if self.body_ac:
            action_space.append(actions['BODY_AC_A'])
            action_space.append(actions['BODY_AC_C'])
        if self.legs_a:
            action_space.append(actions['LEGS_A_A'])
        if self.legs_ac:
            action_space.append(actions['LEGS_AC_A'])
            action_space.append(actions['LEGS_AC_C'])
        
        return tuple(action_space)
    
    def transition(self, actions):
        action, _ = actions
        card_name = action.card_name()
        update = {card_name : getattr(self, card_name)-1}
        return self._replace(**update)
    
    @property
    def terminal(self):
        return sum(self) == 0
    
    @property
    def total(self):
        return sum(self)

class PlayerState(namedtuple(
    'PlayerState',
    ('hit_state', 'card_state'),
    defaults=(HitState(), CardState()),
)):
    '''
    Contains a HitState and CardState to define the state for a single player.
    '''
    def transition(self, actions):
        return PlayerState(
            hit_state=self.hit_state.transition(actions),
            card_state=self.card_state.transition(actions),
        )
    
    @property
    def is_dead(self):
        return self.hit_state.is_dead
    
    @property
    def terminal(self):
        return self.hit_state.terminal or self.card_state.terminal
    
    @property
    def action_space(self):
        return self.card_state.action_space
    
    @property
    def flat(self):
        return self.hit_state + self.card_state

class State(namedtuple(
    'State',
    ('p1', 'p2'),
    defaults=(PlayerState(), PlayerState()),
)):
    '''
    Contains two PlayerStates to represent the state of an entire game.
    '''
    def transition(self, actions):
        a1, a2 = actions
        return State(
            p1=self.p1.transition(actions),
            p2=self.p2.transition((a2,a1)),
        )
    
    @property
    def terminal(self):
        return self.p1.terminal or self.p2.terminal
    
    @property
    def value(self):
        if self.terminal:
            p1_dead = self.p1.is_dead
            p2_dead = self.p2.is_dead
            if p1_dead and p2_dead:
                return 0.5
            elif p1_dead:
                return 0.1
            elif p2_dead:
                return 0.9
            else:
                return 0.5
        else:
            return None
    
    @property
    def action_space(self):
        return self.p1.action_space, self.p2.action_space
    
    def __str__(self):
        p1_dead = 'DEAD' if self.p1.is_dead else '    '
        p2_dead = 'DEAD' if self.p2.is_dead else '    '
        return (
            'p1: %s    |p2: %s\n'%(p1_dead, p2_dead) +
            '------------+------------\n' +
            'head: %i     |head: %i\n'%(
                self.p1.hit_state.head, self.p2.hit_state.head) +
            'body: %i     |body: %i\n'%(
                self.p1.hit_state.body, self.p2.hit_state.body) +
            'legs: %i     |legs: %i\n'%(
                self.p1.hit_state.legs, self.p2.hit_state.legs) +
            '------------+------------\n' +
            'head_a:  %i  |head_a:  %i\n'%(
                self.p1.card_state.head_a, self.p2.card_state.head_a) +
            'head_ac: %i  |head_ac: %i\n'%(
                self.p1.card_state.head_ac, self.p2.card_state.head_ac) +
            'body_a:  %i  |body_a:  %i\n'%(
                self.p1.card_state.body_a, self.p2.card_state.body_a) +
            'body_ac: %i  |body_ac: %i\n'%(
                self.p1.card_state.body_ac, self.p2.card_state.body_ac) +
            'legs_a:  %i  |legs_a:  %i\n'%(
                self.p1.card_state.legs_a, self.p2.card_state.legs_a) +
            'legs_ac: %i  |legs_ac: %i\n'%(
                self.p1.card_state.legs_ac, self.p2.card_state.legs_ac)
        )
    
    def serialize(self):
        return ','.join(str(i) for i in
            self.p1.hit_state +
            self.p1.card_state +
            self.p2.hit_state +
            self.p2.card_state
        )
    
    @staticmethod
    def deserialize(data):
        data = tuple(int(d) for d in data.split(','))
        p1_hit_state = HitState(data[:3])
        p1_card_state = CardState(data[3:9])
        p1 = PlayerState(p1_hit_state, p1_card_state)
        p2_hit_state = HitState(data[9:12])
        p2_card_state = CardState(data[12:])
        p2 = PlayerState(p2_hit_state, p2_card_state)
        return State(p1, p2)
    
    @property
    def flat(self):
        return self.p1.flat + self.p2.flat
