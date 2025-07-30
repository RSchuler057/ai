import random

class AIPlayer:
    def __init__(self, name):
        self.name = name
        
    def choose_action(self, valid_actions, amount_to_call=0, min_raise=0, max_raise=0):
        action = random.choice(valid_actions)
        if action in ['raise', 'bet']:
            if max_raise > 0 and min_raise <= max_raise:
                raise_amt = random.randint(min_raise, max_raise)

            elif max_raise > 0:
                raise_amt = max_raise

            else:
                fallback_actions = [a for a in valid_actions if a not in ['raise', 'bet']]
                if fallback_actions:
                    action = random.choice(fallback_actions)
                    return action, None
                
                else:
                    return 'fold', None
            return action, raise_amt
        return action, None