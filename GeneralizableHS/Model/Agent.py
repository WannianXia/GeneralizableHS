from Model.ModelWrapper import Model as PolicyModel

import torch
from Model.models import LanguageEncoder
from Algo.utils import obs2tensor
from simplet5 import SimpleT5
from transformers import AutoModel, AutoTokenizer


class NewAgent:

    def __init__(self, model_path='TrainedModels/t5-ft-wm-0.5', t5_path='t5-ft', device=0, checkpoint_id=100):

        self.model_path = model_path
        self.device_id = device
        self.device_name = 'cuda:' + str(device)
        self.encoder = SimpleT5()
        self.encoder.from_pretrained("t5","./t5-base")
        self.encoder.load_model("t5", t5_path, use_gpu=False)
        self.encoder.model.to(self.device_name)
        self.encoder = LanguageEncoder(model=self.encoder.model, tokenizer=self.encoder.tokenizer, device=self.device_name)
        self.actor_path = self.model_path + '/actor/actor_weights_{}.ckpt'.format(checkpoint_id * 1000000)
        self.re_path = self.model_path + '/representation/representation_weights_{}.ckpt'.format(checkpoint_id * 1000000)
        self.wm_path = self.model_path + '/wm/wm_weights_{}.ckpt'.format(checkpoint_id * 1000000)
        self.model =PolicyModel(device=self.device_id)
        re, wm, ac = self.model.get_model()
        re.load_state_dict(torch.load(self.re_path, map_location=self.device_name))
        ac.load_state_dict(torch.load(self.actor_path, map_location=self.device_name))
        wm.load_state_dict(torch.load(self.wm_path, map_location=self.device_name))
        self.model.eval()

    def act(self, obs):
        hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_stats, minion_stats, hero_stats, deck_strategy_embed = obs2tensor(self.encoder, obs, self.device_name, False)
        if self.no_deck:
            deck_strategy_embed = None
        with torch.no_grad():
            _, _, agent_output = self.model.forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, deck_strategy_embed, hand_card_stats, minion_stats, hero_stats)
        agent_output = agent_output.argmax()
        action_idx = int(agent_output.cpu().detach().numpy())

        return action_idx

    def predict(self, repr):
        return self.model.get_next_repr(repr)

    def act_predict(self, obs):
        hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_stats, minion_stats, hero_stats, deck_strategy_embed = obs2tensor(self.encoder, obs, self.device_name, False)
        if self.no_deck:
            deck_strategy_embed = None
        with torch.no_grad():
            repr, next_repr, agent_output = self.model.forward(hand_card_embed, minion_embed, secret_embed, weapon_embed, deck_strategy_embed, hand_card_stats, minion_stats, hero_stats)
        agent_output = agent_output.argmax()
        action_idx = int(agent_output.cpu().detach().numpy())


        return action_idx, next_repr, repr

    