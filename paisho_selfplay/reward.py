"""Versioned terminal rewards; length never stops a game."""
import math
from paisho_ppo.ppo import terminal_return

LEGACY='terminal-v1'
HALF='terminal-speed-half-v1'
PROFILES=(LEGACY,HALF)
POWER=math.log(5/18)/math.log(5/9)


def speed_bonus(moves):
    if moves<=100:return .5
    if moves>=1000:return 0.0
    return .5*((1000-moves)/900)**POWER


def game_return(outcome,player,moves,profile=LEGACY):
    if profile not in PROFILES:raise ValueError('Unknown terminal reward profile')
    result=terminal_return(outcome,player)
    return result+speed_bonus(moves) if result>0 and profile==HALF else result
