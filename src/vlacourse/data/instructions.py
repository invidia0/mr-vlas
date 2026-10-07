"""Procedural corpus of "depot scenario" robot instructions.

The running example of the book is a depot/yard where a heterogeneous robot
team (a mobile manipulator and quadrotors) receives natural-language
instructions. This grammar generates such instructions so that the
tokenization, embedding and language-model labs train on text that later
reappears as the language input of our toy VLA.
"""

import random

ROBOTS = ["the drone", "drone one", "drone two", "the arm", "the rover", "the team", "all drones"]
COLORS = ["red", "blue", "green", "yellow", "orange", "white", "black"]
OBJECTS = ["toolbox", "crate", "pallet", "barrel", "cone", "panel", "ladder", "cable reel"]
PLACES = [
    "the east yard", "the west yard", "the loading bay", "the inspection station",
    "the charging pad", "the north gate", "the depot", "the storage rack", "the fuel tank",
]
RELATIONS = ["near", "next to", "behind", "in front of", "left of", "right of"]
ALTITUDES = ["low", "at five meters", "at ten meters", "slowly", "carefully", "quickly"]

TEMPLATES = [
    "{robot}, survey {place}.",
    "{robot}, survey {place} {alt}.",
    "find the {color} {obj}.",
    "find the {color} {obj} {rel} {place}.",
    "pick up the {color} {obj}.",
    "pick up the {color} {obj} and place it at {place}.",
    "bring the {color} {obj} to {place}.",
    "{robot}, fly to {place} and look for the {color} {obj}.",
    "{robot}, return to {place}.",
    "survey {place}, find the {color} {obj}, and bring it to {place2}.",
    "{robot}, inspect the {obj} {rel} {place}.",
    "{robot}, hover above the {color} {obj}.",
    "stack the {color} {obj} on the {color2} {obj2}.",
    "push the {color} {obj} {rel} the {color2} {obj2}.",
    "{robot}, follow {robot2} to {place}.",
    "{robot}, cover {place} while {robot2} searches {place2}.",
]

#: The instruction used as running example throughout the book.
RUNNING_EXAMPLE = "Survey the east yard, find the red toolbox, and bring it to the inspection station."


def sample_instruction(rng: random.Random) -> str:
    t = rng.choice(TEMPLATES)
    place, place2 = rng.sample(PLACES, 2)
    robot, robot2 = rng.sample(ROBOTS, 2)
    color, color2 = rng.sample(COLORS, 2)
    obj, obj2 = rng.sample(OBJECTS, 2)
    s = t.format(
        robot=robot, robot2=robot2, color=color, color2=color2, obj=obj, obj2=obj2,
        place=place, place2=place2, rel=rng.choice(RELATIONS), alt=rng.choice(ALTITUDES),
    )
    return s[0].upper() + s[1:]


def instruction_corpus(n: int = 5000, seed: int = 0) -> list[str]:
    """n instructions, deterministic given the seed. Includes the running example once."""
    rng = random.Random(seed)
    return [RUNNING_EXAMPLE] + [sample_instruction(rng) for _ in range(n - 1)]
