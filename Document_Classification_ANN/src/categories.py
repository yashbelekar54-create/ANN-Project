"""Human-readable descriptions of the 20 Newsgroups categories.

The category *labels* always come from the dataset itself (``target_names``) and the
saved label encoder. This module only adds explanatory text for the UI.
"""

CATEGORY_DESCRIPTIONS = {
    "alt.atheism": "Discussion of atheism, agnosticism and arguments about the existence of God.",
    "comp.graphics": "Computer graphics: image formats, rendering, graphics software and hardware.",
    "comp.os.ms-windows.misc": "Microsoft Windows (3.x era): usage, drivers, applications and troubleshooting.",
    "comp.sys.ibm.pc.hardware": "IBM PC compatible hardware: motherboards, drives, video cards and peripherals.",
    "comp.sys.mac.hardware": "Apple Macintosh hardware: models, upgrades, peripherals and compatibility.",
    "comp.windows.x": "The X Window System: servers, clients, toolkits, window managers and configuration.",
    "misc.forsale": "Classified-style posts offering items for sale, with prices and contact details.",
    "rec.autos": "Cars and driving: models, maintenance, buying advice and automotive news.",
    "rec.motorcycles": "Motorcycles: riding, repair, gear, models and safety.",
    "rec.sport.baseball": "Baseball: teams, players, games and statistics.",
    "rec.sport.hockey": "Ice hockey: teams, players, games and league news.",
    "sci.crypt": "Cryptography: encryption schemes, key management, privacy and security policy.",
    "sci.electronics": "Electronics: circuits, components, repair and design questions.",
    "sci.med": "Medicine and health: conditions, treatments, research and medical advice.",
    "sci.space": "Space exploration and astronomy: NASA, launches, orbits and missions.",
    "soc.religion.christian": "Christian faith: theology, scripture, church practice and belief.",
    "talk.politics.guns": "Firearms politics: gun control, legislation and rights debates.",
    "talk.politics.mideast": "Middle East politics: regional conflicts, history and international relations.",
    "talk.politics.misc": "General political discussion not covered by other politics groups.",
    "talk.religion.misc": "Miscellaneous religion debates and discussion of beliefs.",
}


def describe(category: str) -> str:
    return CATEGORY_DESCRIPTIONS.get(category, "No description available.")
