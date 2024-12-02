from Model.models import Actor, Representation, WorldModel
import torch
import torch.nn.init as init

class Model:
    """
    The wrapper for the models. We also wrap several
    interfaces such as share_memory, eval, etc.
    """
    def __init__(self, card_dim = 16, bert_dim = 768, embed_dim = 256, dim_ff = 256, device=0):
        self.models = {}
        if not device == "cpu":
            device = 'cuda:' + str(device)

        self.entity_dim = card_dim + embed_dim
        self.representation = Representation(card_dim, bert_dim, embed_dim, dim_ff).to(torch.device(device))
        self.world_model = WorldModel(self.entity_dim, dim_ff).to(torch.device(device))
        self.actor = Actor(self.entity_dim, dim_ff).to(torch.device(device))
        self.device = torch.device(device)

    def forward(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, deck_strategy_embed, hand_card_stats, minion_stats, hero_stats):

        repr = self.representation(hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_stats, minion_stats, hero_stats)
        next_repr = self.world_model(repr)
        action = self.actor(repr, deck_strategy_embed)
        return repr, next_repr, action

    def get_repr(self, hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_stats, minion_stats, hero_stats):
        return self.representation(hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_stats, minion_stats, hero_stats)

    def get_action_value(self, repr):
        return self.actor(repr)

    def get_next_repr(self, repr):
        return self.world_model(repr)

    def share_memory(self):
        self.representation.share_memory()
        self.world_model.share_memory()
        self.actor.share_memory()

        return

    def eval(self):
        self.representation.eval()
        self.world_model.eval()
        self.actor.eval()
        return

    def parameters(self):
        return list(self.representation.parameters()) + list(self.world_model.parameters()) + list(self.actor.parameters())

    def get_model(self):
        return self.representation, self.world_model, self.actor

    def load_state_dict(self, representation, wm, actor):
        self.representation.load_state_dict(representation)
        self.world_model.load_state_dict(wm)
        self.actor.load_state_dict(actor)
        return


if __name__ == "__main__":


    def test_model():
        # Initialize the model
        model = Model(card_dim=8, bert_dim=768, embed_dim=64, dim_ff=64, device='cpu')
        model.eval()

        # Generate some random input data
        batch_size = 4
        hand_card_embed = torch.randn(batch_size, 11, 768)  # (batch_size, num_hand_cards, embed_dim)
        minion_embed = torch.randn(batch_size, 14, 768)  # (batch_size, num_minions, embed_dim)
        secret_embed = torch.randn(batch_size, 5, 768)  # (batch_size, num_secrets, embed_dim)
        weapon_embed = torch.randn(batch_size, 2, 768)  # (batch_size, num_weapons, embed_dim)

        hand_card_stats = torch.randn(batch_size, 11, 19)  # (batch_size, num_hand_cards, embed_dim)
        minion_stats = torch.randn(batch_size, 14, 23)  # (batch_size, num_minions, embed_dim)
        hero_stats = torch.randn(batch_size, 2, 29)  # (batch_size, 1, embed_dim)

        # Forward pass
        repr, next_repr, action = model.forward(
            hand_card_embed, 
            minion_embed, 
            secret_embed, 
            weapon_embed, 
            hand_card_stats, 
            minion_stats, 
            hero_stats
        )

        # Print shapes of outputs to verify
        print(f"Representation shape: {repr.shape}")
        print(f"Next representation shape: {next_repr.shape}")
        print(f"Action shape: {action.shape}")

        # Test get_repr method
        repr_test = model.get_repr(
            hand_card_embed, 
            minion_embed, 
            secret_embed, 
            weapon_embed, 
            hand_card_stats, 
            minion_stats, 
            hero_stats
        )
        assert torch.equal(repr, repr_test), "Representation from forward() and get_repr() should match"

        # Test get_action_value method
        action_test = model.get_action_value(repr)
        assert torch.equal(action, action_test), "Action from forward() and get_action_value() should match"

        # Test get_next_repr method
        next_repr_test = model.get_next_repr(repr)
        assert torch.equal(next_repr, next_repr_test), "Next representation from forward() and get_next_repr() should match"

        # Test share_memory method
        model.share_memory()

        # Test eval mode
        model.eval()

        print("All tests passed!")

    # Run the test
    test_model()
