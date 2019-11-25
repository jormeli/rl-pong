"""Cluster/local submission script."""

import numpy as np
import os

from aalto_submit import AaltoSubmission
from submit_utils import EasyDict

from networks import DuelingDQN, VanillaDQN, CNN
from loss import td_loss, double_dqn_loss, categorical_loss


NET_NAMES = {'dqn': VanillaDQN, 'cnn': CNN, 'dueling_dqn': DuelingDQN}
LOSS_NAMES = {'td_loss': td_loss, 'double_dqn': double_dqn_loss, 'categorical': categorical_loss}

#----------------------------------------------------------------------------

if __name__ == "__main__":
    for eps in [0.1]:  #[0.1, 0.2]:
        for mb in [32]:  #[64, 128, 256]:
            for loss_name in ['categorical']:  #['double_dqn', 'td_loss', 'categorical']
                for net_name in ['dueling_dqn']:  #['dqn', 'dueling_dqn', 'cnn']
                    for noisy in [True]:  #[True, False]
                        for normalize in [False]:  #[True, False]
                            for rep in range(1):
                                submit_config = EasyDict()
                                run_func_args = EasyDict()

                                # Define the function that we want to run.
                                submit_config.run_func = 'cluster_training_loop.training_loop'

                                # Define where results from the run are saved and give a name for the run.
                                submit_config.run_dir_root = '<RESULTS>/graphics/kynkaat1/results/nbc-pong'

                                # Define parameters for run time, number of GPUs, etc.
                                submit_config.time = '5-00:00:00'  # In format d-hh:mm:ss.
                                submit_config.num_gpus = 0
                                submit_config.num_cores = 1
                                submit_config.cpu_memory = 60  # In GB.
                                submit_config.extra_packages = ['torch',
                                                                'gym',
                                                                'matplotlib',
                                                                'numpy',
                                                                'pillow',
                                                                'opencv-python',
                                                                'tensorboard',
                                                                'pandas']

                                # Define the envinronment where the task is run.
                                # Pick one of: 'L' (local), 'DGX' (dgx01, dgx02), 'DGX-COMMON' (dgx-common)
                                # or GPU (nodes gpu[1-10], gpu[28-37]).
                                submit_config.env = 'L'

                                # Configure environment.
                                run_func_args.num_episodes = 25000000
                                run_func_args.start_training_at_frame = 50000
                                run_func_args.target_epsilon = eps
                                run_func_args.beta_0 = 0.4
                                run_func_args.reach_target_at_frame = 1.5e6
                                run_func_args.model_update_freq = 4
                                run_func_args.target_update_freq = 8000
                                run_func_args.save_every_n_ep = 250
                                run_func_args.player_id = 1
                                run_func_args.log_freq = 10

                                # Config agent.
                                run_func_args.agent_config = EasyDict(network_name=net_name,
                                                                      input_shape=(1, 84, 84),
                                                                      num_actions=3,
                                                                      stack_size=4,
                                                                      replay_memory_size=1000000,
                                                                      minibatch_size=mb,
                                                                      prioritized=True,
                                                                      normalize=normalize,
                                                                      gamma=0.99,
                                                                      learning_rate=2.5e-4,
                                                                      loss_fn=LOSS_NAMES[loss_name],
                                                                      network_fn=NET_NAMES[net_name])

                                # Config network.
                                categorical = True if loss_name == 'categorical' else False
                                run_func_args.network_fn_kwargs = EasyDict(noisy=noisy,
                                                                           categorical=categorical)

                                if normalize:
                                    submit_config.task_description = '%s-%s-noisy_%s-normalize-target_eps%0.2f-mb%i-stack_size%i-target_up%i-rep%i' % \
                                        (net_name, loss_name, str(noisy), eps, mb, run_func_args.agent_config.stack_size, run_func_args.target_update_freq, rep)
                                    run_func_args.run_description = submit_config.task_description
                                else:
                                    submit_config.task_description = '%s-%s-noisy_%s-target_eps%0.2f-mb%i-stack_size%i-target_up%i-rep%i' % \
                                        (net_name, loss_name, str(noisy), eps, mb, run_func_args.agent_config.stack_size, run_func_args.target_update_freq, rep)
                                    run_func_args.run_description = submit_config.task_description                                

                                # Create submission object.
                                submission = AaltoSubmission(run_func_args, **submit_config)

                                # All set. Run the task in the desired environment.
                                submission.run_task()
                                print()

#----------------------------------------------------------------------------
