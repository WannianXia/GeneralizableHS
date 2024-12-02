import random
import json

class Deck:
    def __init__(self, deck_data_path):
        self.decks = []
        self.load_decks(deck_data_path)


    def load_decks(self, deck_data_path):
        with open(deck_data_path, 'r') as file:
            self.decks = json.load(file)

    def get_random_deck(self):
        return random.choice(self.decks)

    def get_deck_by_index(self, idx):
        for deck in self.decks:
            if deck['id'] == idx:
                return deck

    def __len__(self):
        return len(self.decks)

    def __iter__(self):
        return iter(self.decks)


