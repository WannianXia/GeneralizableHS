import argparse
from pickle import TRUE

parser = argparse.ArgumentParser(description='Hearthstone agent')

# General Settings
parser.add_argument('--xpid', default='001',
                    help='Experiment id')
parser.add_argument('--frame_interval', default=1000000, type=int, 
                    help='Frame interval to save checkpoints')


# Training settings

parser.add_argument('--disable_finetune', action='store_true',
                    help='Disable using finetuned T5')
parser.add_argument('--t5_model_path', type=str, default='t5-ft',
                    help='finetuned t5 model path')
parser.add_argument('--actor_device_cpu', action='store_true',
                    help='Use CPU as actor device')
parser.add_argument('--gpu_devices', default='0,1,2,3,4,5', type=str,
                    help='Which GPUs to be used for training')
parser.add_argument('--num_actor_devices', default=6, type=int,
                    help='The number of devices used for simulation')
parser.add_argument('--num_actors', default=2, type=int,
                    help='The number of actors for each simulation device')
parser.add_argument('--training_device', default='0', type=str,
                    help='The index of the GPU used for training models. `cpu` means using cpu')
parser.add_argument('--load_model', action='store_true',
                    help='Load an existing model')
parser.add_argument('--disable_checkpoint', action='store_true',
                    help='Disable saving checkpoint')
parser.add_argument('--savedir', default='TrainedModels',
                    help='Root dir where experiment data will be saved')

# Hyperparameters
parser.add_argument('--total_frames', default=10000000000, type=int,
                    help='Total environment frames to train for')
parser.add_argument('--exp_epsilon', default=0.01, type=float,
                    help='The probability for exploration')
parser.add_argument('--wm_weight', default=0.5, type=float,
                    help='The weight for world model training loss')
parser.add_argument('--gamma', default=1, type=float,
                    help='The discount factor for rl training')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Learner batch size')
parser.add_argument('--unroll_length', default=100, type=int,
                    help='The unroll length (time dimension)')
parser.add_argument('--num_buffers', default=4, type=int,
                    help='Number of shared-memory buffers')
parser.add_argument('--num_threads', default=1, type=int,
                    help='Number learner threads')
parser.add_argument('--max_grad_norm', default=40., type=float,
                    help='Max norm of gradients')

# Optimizer settings
parser.add_argument('--learning_rate', default=0.0001, type=float,
                    help='Learning rate')
parser.add_argument('--alpha', default=0.99, type=float,
                    help='RMSProp smoothing constant')
parser.add_argument('--momentum', default=0, type=float,
                    help='RMSProp momentum')
parser.add_argument('--epsilon', default=1e-5, type=float,
                    help='RMSProp epsilon')

