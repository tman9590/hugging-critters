"""The explicit, reproducible Eastern-US subset of the source taxonomy."""

# A class is included when the broad source label has a native, established,
# introduced, or domestic occurrence in the eastern contiguous United States
# or its Atlantic/Gulf coastal waters. Zoo-only and pet-only exotics are out.
EASTERN_US_CLASSES = (
    "bat", "bear", "bee", "beetle", "boar", "butterfly", "cat",
    "caterpillar", "cockroach", "cow", "coyote", "crab", "crow", "deer",
    "dog", "dolphin", "donkey", "dragonfly", "duck", "eagle", "fly", "fox",
    "goat", "goldfish", "goose", "grasshopper", "hare", "horse",
    "hummingbird", "jellyfish", "ladybugs", "lizard", "lobster", "mosquito",
    "moth", "mouse", "octopus", "otter", "owl", "ox", "oyster",
    "pelecaniformes", "pig", "pigeon", "porcupine", "possum", "raccoon",
    "rat", "sandpiper", "seahorse", "seal", "shark", "sheep", "snake",
    "sparrow", "squid", "squirrel", "starfish", "swan", "turkey", "turtle",
    "whale", "wolf", "woodpecker",
)

EXCLUDED_CLASSES = {
    "antelope": "not an eastern-US wild taxon",
    "badger": "primarily central/western; excluded for a conservative eastern range",
    "bison": "no broadly established eastern wild range",
    "chimpanzee": "zoo-only exotic",
    "elephant": "zoo-only exotic",
    "flamingo": "rare/vagrant rather than broadly established",
    "gorilla": "zoo-only exotic",
    "hamster": "pet-only exotic",
    "hedgehog": "pet-only exotic",
    "hippopotamus": "zoo-only exotic",
    "hornbill": "zoo-only exotic",
    "hyena": "zoo-only exotic",
    "kangaroo": "zoo-only exotic",
    "koala": "zoo-only exotic",
    "leopard": "zoo-only exotic",
    "lion": "source class denotes the Old World/African animal, not cougar",
    "okapi": "zoo-only exotic",
    "orangutan": "zoo-only exotic",
    "panda": "zoo-only exotic",
    "parrot": "localized escapes; excluded for a conservative range",
    "penguin": "not present in eastern-US waters",
    "reindeer": "not established in the eastern contiguous US",
    "rhinoceros": "zoo-only exotic",
    "tiger": "zoo-only exotic",
    "wombat": "zoo-only exotic",
    "zebra": "zoo-only exotic",
}

