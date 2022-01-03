from collections import namedtuple

game_mode = 'large' # 'large' or 'small' or 'tiny'

class Action(namedtuple('Action', ('region', 'card', 'mode'))):
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

if game_mode == 'large':
    max_head_hits = 2
    max_body_hits = 3
    max_legs_hits = 4
    max_total_hits = 5
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

class HitState(namedtuple(
    'HitState',
    ('head', 'body', 'legs'),
    defaults=(0,0,0),
)):
    def get_total(self):
        return sum(self)
    
    total = property(get_total)
    
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
    
    def terminal(self):
        return self.is_dead()

if game_mode == 'large':
    start_head_a = 1
    start_head_ac = 3
    start_body_a = 1
    start_body_ac = 4
    start_legs_a = 1
    start_legs_ac = 4
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
    
    def terminal(self):
        return sum(self) == 0
    
    def get_total(self):
        return sum(self)
    
    total = property(get_total)

class PlayerState(namedtuple(
    'PlayerState',
    ('hit_state', 'card_state'),
    defaults=(HitState(), CardState()),
)):
    
    def transition(self, actions):
        return PlayerState(
            hit_state=self.hit_state.transition(actions),
            card_state=self.card_state.transition(actions),
        )
    
    def is_dead(self):
        return self.hit_state.is_dead()
    
    def terminal(self):
        return self.hit_state.terminal() or self.card_state.terminal()
    
    def action_space(self):
        return self.card_state.action_space()

class State(namedtuple(
    'State',
    ('p1', 'p2'),
    defaults=(PlayerState(), PlayerState()),
)):
    
    def transition(self, actions):
        a1, a2 = actions
        return State(
            p1=self.p1.transition(actions),
            p2=self.p2.transition((a2,a1)),
        )
    
    def terminal(self):
        return self.p1.terminal() or self.p2.terminal()
    
    def value(self):
        if self.terminal():
            p1_dead = self.p1.is_dead()
            p2_dead = self.p2.is_dead()
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
    
    def action_space(self):
        return self.p1.action_space(), self.p2.action_space()
    
    def __str__(self):
        p1_dead = 'DEAD' if self.p1.is_dead() else '    '
        p2_dead = 'DEAD' if self.p2.is_dead() else '    '
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
