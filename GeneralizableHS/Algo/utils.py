import torch
import typing
import traceback
import numpy as np

import torch 
from torch import multiprocessing as mp
import torch.nn.functional as F

# import sys
# sys.path.append('/data/xwn/projects/CardsDreamer')
from Env.Hearthstone import Hearthstone
from transformers import AutoModel, AutoTokenizer
from Model.models import LanguageEncoder
from simplet5 import SimpleT5
import random

import logging
import signal

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('HSAgent')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

# Buffers are used to transfer data between actor processes
# and learner processes. They are shared tensors in GPU
Buffers = typing.Dict[str, typing.List[torch.Tensor]]

def get_batch(free_queue,
              full_queue,
              buffers,
              flags,
              lock):
    """
    This function will sample a batch from the buffers based
    on the indices received from the full queue. It will also
    free the indices by sending it to full_queue.
    """
    with lock:
        indices = [full_queue.get() for _ in range(flags.batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }

    for m in indices:
        free_queue.put(m)

    return batch

def create_optimizers(flags, learner_model):
    """
    Create three optimizers for the three positions
    """
    optimizer = torch.optim.RMSprop(
        learner_model.parameters(),
        lr=flags.learning_rate,
        momentum=flags.momentum,
        eps=flags.epsilon,
        alpha=flags.alpha)
    return optimizer


def create_buffers(flags, device_iterator):
    """
    We create buffers for different positions as well as
    for different devices (i.e., GPU). That is, each device
    will have three buffers for the three positions.
    """

    '''
    repr(embed + stats) = latent_obs
    world_model(latent_obs) = next_latent_obs
    actor(latent_obs) = action
    value - actor mse  train actor
    next_latent_obs - buffered next latent_obs mse train world_model
    
    '''

    T = flags.unroll_length
    positions = ['Player1', 'Player2']
    
    buffers = {}
    for device in device_iterator:
        buffers[device] = {}
        for position in positions:
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool), 
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                hand_card_embed=dict(size=(T, 11, 768), dtype=torch.float32),
                minion_embed=dict(size=(T, 14, 768), dtype=torch.float32),
                weapon_embed=dict(size=(T, 2, 768), dtype=torch.float32),
                secret_embed=dict(size=(T, 5, 768), dtype=torch.float32),
                deck_strategy_embed=dict(size=(T, 1, 768), dtype=torch.float32),
                hand_card_stats=dict(size=(T, 11, 23), dtype=torch.float32),
                minion_stats=dict(size=(T, 14, 26), dtype=torch.float32),
                hero_stats=dict(size=(T, 2, 31), dtype=torch.float32),
                hand_card_embed_next=dict(size=(T, 11, 768), dtype=torch.float32),
                minion_embed_next=dict(size=(T, 14, 768), dtype=torch.float32),
                weapon_embed_next=dict(size=(T, 2, 768), dtype=torch.float32),
                secret_embed_next=dict(size=(T, 5, 768), dtype=torch.float32),
                hand_card_stats_next=dict(size=(T, 11, 23), dtype=torch.float32),
                minion_stats_next=dict(size=(T, 14, 26), dtype=torch.float32),
                hero_stats_next=dict(size=(T, 2, 31), dtype=torch.float32),
                )
            _buffers: Buffers = {key: [] for key in specs}
            for _ in range(flags.num_buffers):
                for key in _buffers:
                    if not device == "cpu":
                        _buffer = torch.empty(**specs[key]).to(torch.device('cuda:'+str(device))).share_memory_()
                    else:
                        _buffer = torch.empty(**specs[key]).to(torch.device('cpu')).share_memory_()
                    _buffers[key].append(_buffer)
            buffers[device][position] = _buffers
    return buffers


def obs2tensor(language_encoder, obs, device, is_next):
    
    if is_next:
        with torch.no_grad():
            hand_card_embed = language_encoder.encode_by_names(obs['hand_card_names']).unsqueeze(0)
            minion_embed = language_encoder.encode_by_names(obs['minion_names']).unsqueeze(0)
            secret_embed = language_encoder.encode_by_names(obs['secret_names']).unsqueeze(0)
            weapon_embed = language_encoder.encode_by_names(obs['weapon_names']).unsqueeze(0)
        deck_strategy = None

        if len(obs['hand_card_stats'].shape) == 3:
            hand_card_stats = torch.tensor(obs['hand_card_stats'][0, ...], dtype=torch.float32).to(device).unsqueeze(0)
            minion_stats = torch.tensor(obs['minion_stats'][0, ...], dtype=torch.float32).to(device).unsqueeze(0)
            hero_stats = torch.tensor(obs['hero_stats'][0, ...], dtype=torch.float32).to(device).unsqueeze(0)
        else:
            hand_card_stats = torch.tensor(obs['hand_card_stats'], dtype=torch.float32).to(device).unsqueeze(0)
            minion_stats = torch.tensor(obs['minion_stats'], dtype=torch.float32).to(device).unsqueeze(0)
            hero_stats = torch.tensor(obs['hero_stats'], dtype=torch.float32).to(device).unsqueeze(0)
    else:
        len_options = obs['hand_card_stats'].shape[0]
        with torch.no_grad():

            hand_card_embed = language_encoder.encode_by_names(obs['hand_card_names']).unsqueeze(0).repeat(len_options, 1, 1)
            minion_embed = language_encoder.encode_by_names(obs['minion_names']).unsqueeze(0).repeat(len_options, 1, 1)
            secret_embed = language_encoder.encode_by_names(obs['secret_names']).unsqueeze(0).repeat(len_options, 1, 1)
            weapon_embed = language_encoder.encode_by_names(obs['weapon_names']).unsqueeze(0).repeat(len_options, 1, 1)
            deck_strategy = language_encoder.encode_by_names([obs['deck_strategy'], ]).unsqueeze(0).repeat(len_options, 1, 1)

        hand_card_stats = torch.tensor(obs['hand_card_stats'], dtype=torch.float32).to(device)
        minion_stats = torch.tensor(obs['minion_stats'], dtype=torch.float32).to(device)
        hero_stats = torch.tensor(obs['hero_stats'], dtype=torch.float32).to(device)

    return hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_stats, minion_stats, hero_stats, deck_strategy


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException("Operation timed out")

def act(i, device, free_queue, full_queue, model, buffers, flags):
    """
    This function will run forever until we stop it. It will generate
    data from the environment and send the data to buffer. It uses
    a free queue and full queue to syncup with the main process.
    """


    signal.signal(signal.SIGALRM, handler)


    positions = ['Player1', 'Player2']
    try:
        T = flags.unroll_length
        log.info('Device %s Actor %i started!!!!!!!!!!!!', str(device), i)
        
        env = Hearthstone()
        
        done_buf = {p: [] for p in positions}
        episode_return_buf = {p: [] for p in positions}
        target_buf = {p: [] for p in positions}
        hand_card_embed_buf = {p: [] for p in positions}
        minion_embed_buf = {p: [] for p in positions}
        weapon_embed_buf = {p: [] for p in positions}
        secret_embed_buf = {p: [] for p in positions}
        deck_strategy_buf = {p: [] for p in positions}
        hand_card_stats_buf = {p: [] for p in positions}
        minion_stats_buf = {p: [] for p in positions}
        hero_stats_buf = {p: [] for p in positions}
        next_hand_card_embed_buf = {p: [] for p in positions}
        next_minion_embed_buf = {p: [] for p in positions}
        next_weapon_embed_buf = {p: [] for p in positions}
        next_secret_embed_buf = {p: [] for p in positions}
        next_hand_card_stats_buf = {p: [] for p in positions}
        next_minion_stats_buf = {p: [] for p in positions}
        next_hero_stats_buf = {p: [] for p in positions}


        size = {p: 0 for p in positions}
        if flags.disable_finetune:
            tokenizer = AutoTokenizer.from_pretrained("./t5-base")
            auto_model = AutoModel.from_pretrained("./t5-base")
            encoder = LanguageEncoder(model=auto_model, tokenizer=tokenizer, device=device)

        else:
            encoder = SimpleT5()
            encoder.from_pretrained("t5","./t5-base")
            encoder.load_model("t5",flags.t5_model_path, use_gpu=False)
            encoder.model.eval()
            log.info(f"Loading Finetuned T5 from {flags.t5_model_path}")
            encoder = LanguageEncoder(model=encoder.model, tokenizer=encoder.tokenizer, device=device)
        encoder.to(device)
        

        while True:
            try:
                position, obs, options, reward, done = env.reset()
            except:
                continue
            env_fail = False
            last_buf_len = {
                p: size[p] for p in positions
            }
            while not done:
                cur_position = position
                while not done and cur_position == position:
                    hand_card_embed, minion_embed, secret_embed, weapon_embed, hand_card_stats, minion_stats, hero_stats, deck_strategy_embed = obs2tensor(encoder, obs[cur_position], device, False)

                    num_options = len(options)
                    save_buf = False
                    if num_options == 0:
                        action = options[0]
                    elif type(options[0]).__name__ == 'ChooseTask':
                        action = random.choice(options)
                    else:
                        save_buf = True
                        with torch.no_grad():
                            batch_size = 300
                            agent_output = []
                            for batch_id in range(0, num_options, batch_size):
                                start_idx = batch_id
                                end_idx = min(batch_id + batch_size, num_options)
                                
                                batch_hand_card_embed = hand_card_embed[start_idx:end_idx, ...]
                                batch_minion_embed = minion_embed[start_idx:end_idx, ...]
                                batch_secret_embed = secret_embed[start_idx:end_idx, ...]
                                batch_weapon_embed = weapon_embed[start_idx:end_idx, ...]
                                batch_deck_strategy_embed = deck_strategy_embed[start_idx:end_idx, ...]
                                batch_hand_card_stats = hand_card_stats[start_idx:end_idx, ...]
                                batch_minion_stats = minion_stats[start_idx:end_idx, ...]
                                batch_hero_stats = hero_stats[start_idx:end_idx, ...]

                                _, _, _agent_output = model.forward(
                                    batch_hand_card_embed, 
                                    batch_minion_embed, 
                                    batch_secret_embed, 
                                    batch_weapon_embed, 
                                    batch_deck_strategy_embed,
                                    batch_hand_card_stats, 
                                    batch_minion_stats, 
                                    batch_hero_stats, 
                                    )
                                agent_output.append(_agent_output)
                        agent_output = torch.cat(agent_output, dim = 0)
                        if np.random.rand() < flags.exp_epsilon:
                            _action_idx = torch.randint(len(options), (1, ))[0]
                        else:
                            agent_output = agent_output.argmax()
                            _action_idx = int(agent_output.cpu().detach().numpy())
                        action = options[_action_idx]
                        hand_card_embed_buf[cur_position].append(hand_card_embed[0, ...])
                        minion_embed_buf[cur_position].append(minion_embed[0, ...])
                        weapon_embed_buf[cur_position].append(weapon_embed[0, ...])
                        secret_embed_buf[cur_position].append(secret_embed[0, ...])
                        deck_strategy_buf[cur_position].append(deck_strategy_embed[0, ...])
                        hand_card_stats_buf[cur_position].append(hand_card_stats[_action_idx])
                        minion_stats_buf[cur_position].append(minion_stats[_action_idx])
                        hero_stats_buf[cur_position].append(hero_stats[_action_idx])


                    # save key info buf here
                    try:
                        signal.alarm(3)
                        # print('alarm off!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                        position, obs, options, episode_return, done = env.step(action)
                        if save_buf:
                            hand_card_embed_next, minion_embed_next, secret_embed_next, weapon_embed_next, hand_card_stats_next, minion_stats_next, hero_stats_next, _ = obs2tensor(encoder, obs[cur_position], device, is_next=True)
                            # with torch.no_grad():
                            #     repr_next = model.get_repr(hand_card_embed_next, minion_embed_next, secret_embed_next, weapon_embed_next, hand_card_stats_next, minion_stats_next, hero_stats_next, actor=False)
                            next_hand_card_embed_buf[cur_position].append(hand_card_embed_next)
                            next_minion_embed_buf[cur_position].append(minion_embed_next)
                            next_weapon_embed_buf[cur_position].append(weapon_embed_next)
                            next_secret_embed_buf[cur_position].append(secret_embed_next)
                            next_hand_card_stats_buf[cur_position].append(hand_card_stats_next)
                            next_minion_stats_buf[cur_position].append(minion_stats_next)
                            next_hero_stats_buf[cur_position].append(hero_stats_next)
                            size[cur_position] += 1
                    except Exception as e:
                        if isinstance(e, TimeoutException):
                            log.info("step timeout in device %s", str(device))
                        else:
                            log.error('Exception in env step worker process %i', i)
                            traceback.print_exc()
                            log.info(action.FullPrint())
                        env_fail = True
                        break

                    finally:
                        signal.alarm(0)

                if env_fail:
                    # env failed, restore buf to last game
                    for p in positions:
                        size[p] = last_buf_len[p]
                        for buf_name, buf in [
                            ("hand_card_embed_buf", hand_card_embed_buf),
                            ("minion_embed_buf", minion_embed_buf),
                            ("weapon_embed_buf", weapon_embed_buf),
                            ("secret_embed_buf", secret_embed_buf),
                            ("deck_strategy_buf", deck_strategy_buf),
                            ("hand_card_stats_buf", hand_card_stats_buf),
                            ("minion_stats_buf", minion_stats_buf),
                            ("hero_stats_buf", hero_stats_buf),
                            ("next_hand_card_embed_buf", next_hand_card_embed_buf),
                            ("next_minion_embed_buf", next_minion_embed_buf),
                            ("next_weapon_embed_buf", next_weapon_embed_buf),
                            ("next_secret_embed_buf", next_secret_embed_buf),
                            ("next_hand_card_stats_buf", next_hand_card_stats_buf),
                            ("next_minion_stats_buf", next_minion_stats_buf),
                            ("next_hero_stats_buf", next_hero_stats_buf),
                        ]:
                            buf[p] = buf[p][:last_buf_len[p]]
                    break

                if done:
                    # log.info('Success finishing game in worker %s', str(device))

                    for p in positions:
                        diff = size[p] - len(target_buf[p])
                        if diff > 0:
                            done_buf[p].extend([False for _ in range(diff-1)])
                            done_buf[p].append(True)
                            episode_return = episode_return if p == 'Player1' else -episode_return
                            episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                            episode_return_buf[p].append(episode_return)
                            _target_buf = [episode_return, ]
                            for i in range(diff - 1):
                                _target_buf.append(_target_buf[-1] * flags.gamma)
                            target_buf[p].extend(list(reversed(_target_buf)))
                    break

            for p in positions:
                while size[p] > T: 
                    index = free_queue[p].get()
                    if index is None:
                        log.info('index is none in device %s', str(device))
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['hand_card_embed'][index][t, ...] = hand_card_embed_buf[p][t]
                        buffers[p]['minion_embed'][index][t, ...] = minion_embed_buf[p][t]
                        buffers[p]['weapon_embed'][index][t, ...] = weapon_embed_buf[p][t]
                        buffers[p]['secret_embed'][index][t, ...] = secret_embed_buf[p][t]
                        buffers[p]['deck_strategy_embed'][index][t, ...] = deck_strategy_buf[p][t]

                        buffers[p]['hand_card_stats'][index][t, ...] =	hand_card_stats_buf[p][t]
                        buffers[p]['minion_stats'][index][t, ...] = minion_stats_buf[p][t]
                        buffers[p]['hero_stats'][index][t, ...] = hero_stats_buf[p][t]

                        buffers[p]['hand_card_embed_next'][index][t, ...] = next_hand_card_embed_buf[p][t]
                        buffers[p]['minion_embed_next'][index][t, ...] = next_minion_embed_buf[p][t]
                        buffers[p]['weapon_embed_next'][index][t, ...] = next_weapon_embed_buf[p][t]
                        buffers[p]['secret_embed_next'][index][t, ...] = next_secret_embed_buf[p][t]
                        buffers[p]['hand_card_stats_next'][index][t, ...] = next_hand_card_stats_buf[p][t]
                        buffers[p]['minion_stats_next'][index][t, ...] = next_minion_stats_buf[p][t]
                        buffers[p]['hero_stats_next'][index][t, ...] = next_hero_stats_buf[p][t]


                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    hand_card_embed_buf[p] = hand_card_embed_buf[p][T:]
                    minion_embed_buf[p] = minion_embed_buf[p][T:]
                    weapon_embed_buf[p] = weapon_embed_buf[p][T:]
                    secret_embed_buf[p] = secret_embed_buf[p][T:]
                    deck_strategy_buf[p] = deck_strategy_buf[p][T:]

                    hand_card_stats_buf[p] = hand_card_stats_buf[p][T:]
                    minion_stats_buf[p] = minion_stats_buf[p][T:]
                    hero_stats_buf[p] = hero_stats_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    next_hand_card_embed_buf[p] = next_hand_card_embed_buf[p][T:]
                    next_minion_embed_buf[p] = next_minion_embed_buf[p][T:]
                    next_weapon_embed_buf[p] = next_weapon_embed_buf[p][T:]
                    next_secret_embed_buf[p] = next_secret_embed_buf[p][T:]
                    next_hand_card_stats_buf[p] = next_hand_card_stats_buf[p][T:]
                    next_minion_stats_buf[p] = next_minion_stats_buf[p][T:]
                    next_hero_stats_buf[p] = next_hero_stats_buf[p][T:]

                    size[p] -= T

    except KeyboardInterrupt:
        pass  
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e



if __name__ == "__main__":
    from Algo.arguments import parser
    from torch import multiprocessing as mp
    from Model.ModelWrapper import Model

    flags = parser.parse_args()
    
    i = 0
    device = 6
    buffers = create_buffers(flags, [0, ])
    buffers = buffers[0]

    ctx = mp.get_context('spawn')

    free_queue = {}
    full_queue = {}
    for device in [0, ]:
        _free_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        _full_queue = {
            'Player1': ctx.SimpleQueue(),
            'Player2': ctx.SimpleQueue()
        }
        free_queue[device] = _free_queue
        full_queue[device] = _full_queue

    free_queue = free_queue[0]
    full_queue = full_queue[0]
    
    for m in range(flags.num_buffers):
        free_queue['Player1'].put(m)
        free_queue['Player2'].put(m)
    model = Model(device=0)
    model.share_memory()
    model.eval()
    act(i, device, free_queue, full_queue, model, buffers, flags)