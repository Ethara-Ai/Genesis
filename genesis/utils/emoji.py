import random


def random_scene():
    pass


def get_clock(t, speed=10):
    return "🕐🕑🕒🕓🕔🕕🕖🕗🕘🕙🕚🕛"[int(t * speed) % 12]
