import random

greetings = [
    "Hello!",
    "Hi there!",
    "Hey, how can I help you?",
    "Greetings! How can I assist you?"
]

farewells = [
    "Goodbye!",
    "Bye! Have a great day!",
    "See you later!",
    "Take care, and have a wonderful day!"
]

def random_string():
    random_list = [
        "Please try writing something more descriptive.",
        "Oh! It appears you wrote something I don't understand yet",
        "Do you mind trying to rephrase that?",
        "I'm terribly sorry, I didn't quite catch that.",
        "I can't answer that yet, please try asking something else."
    ]

    list_count = len(random_list)
    random_item = random.randrange(list_count)

    return random_list[random_item]
