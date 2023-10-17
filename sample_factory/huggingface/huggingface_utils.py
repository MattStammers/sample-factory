import os

import cv2
import numpy as np
from huggingface_hub import HfApi, Repository, repocard, upload_folder

from sample_factory.utils.typing import Config
from sample_factory.utils.utils import log, project_tmp_dir

MIN_FRAME_SIZE = 180


def generate_replay_video(dir_path: str, frames: list, fps: int, cfg: Config):
    video_fname = "replay.mp4" if cfg.video_name is None else cfg.video_name
    if not video_fname.endswith(".mp4"):
        video_fname += ".mp4"

    tmp_name = os.path.join(project_tmp_dir(), video_fname)
    video_name = os.path.join(dir_path, video_fname)
    if frames[0].shape[0] == 3:
        frame_size = (frames[0].shape[2], frames[0].shape[1])
    else:
        frame_size = (frames[0].shape[1], frames[0].shape[0])
    resize = False

    if min(frame_size) < MIN_FRAME_SIZE:
        resize = True
        scaling_factor = MIN_FRAME_SIZE / min(frame_size)
        frame_size = (int(frame_size[0] * scaling_factor), int(frame_size[1] * scaling_factor))

    video = cv2.VideoWriter(tmp_name, cv2.VideoWriter_fourcc(*"mp4v"), fps, frame_size)
    for frame in frames:
        if frame.shape[0] == 3:
            frame = frame.transpose(1, 2, 0)
        if resize:
            frame = cv2.resize(frame, frame_size, interpolation=cv2.INTER_AREA)
        video.write(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    video.release()
    os.system(f"ffmpeg -y -i {tmp_name} -vcodec libx264 {video_name}")
    log.debug(f"Replay video saved to {video_name}!")


def generate_model_card(
    dir_path: str,
    algo: str,
    env: str,
    repo_id: str,
    rewards: list = None,
    enjoy_name: str = None,
    train_name: str = None,
):
    readme_path = os.path.join(dir_path, "README.md")
    repo_name = repo_id.split("/")[1]

    readme = f"""
## About the Project\n
This project is an attempt to maximise performance of high sample throughput APPO RL models in Atari environments in as carbon efficient a manner as possible using a single, not particularly high performance single machine. It is about demonstrating the generalisability of on-policy algorithms to create good performance quickly (by sacrificing sample efficiency) while also proving that this route to RL production is accessible to even hobbyists like me (I am a gastroenterologist not a computer scientist). \n
In terms of throughput I am managing to reach throughputs of 2,500 - 3,000 across both policies using sample factory using two Quadro P2200's (not particularly powerful GPUs) each loaded up about 60% (3GB). Previously using the stable baselines 3 (sb3) implementation of PPO it would take about a week to train an atari agent to 100 million timesteps synchronously. By comparison the sample factory async implementation takes only just over 2 hours to achieve the same result. That is about 84 times faster with only typically a 21 watt burn per GPU. I am thus very grateful to Alex Petrenko and all the sample factory team for their work on this.

## Project Aims\n
This model as with all the others in the benchmarks was trained initially asynchronously un-seeded to 10 million steps for the purposes of setting a sample factory async baseline for this model on this environment but only 3/57 made it anywhere near sota performance. \n
I then re-trained the models with 100 million timesteps- at this point 2 environments maxed out at sota performance (Pong and Freeway) with four approaching sota performance - (atlantis, boxing, tennis and fishingderby.) =6/57 near sota. \n
The aim now is to try and reach state-of-the-art (SOTA) performance on a further block of atari environments using up to 1 billion training timesteps initially with appo. I will flag the models with SOTA when they reach at or near these levels. \n
After this I will switch on V-Trace to see if the Impala variations perform any better with the same seed (I have seeded '1234')\n

## About the Model\n
The hyperparameters used in the model are described in my shell script on my fork of sample-factory: https://github.com/MattStammers/sample-factory. Given that https://huggingface.co/edbeeching has kindly shared his parameters, I saved time and energy by using many of his tuned hyperparameters to reduce carbon inefficiency:
```
hyperparameters =  {{
  "help": false,
  "algo": "APPO",
  "env": "atari_asteroid",
  "experiment": "atari_asteroid_APPO",
  "train_dir": "./train_atari",
  "restart_behavior": "restart",
  "device": "gpu",
  "seed": 1234,
  "num_policies": 2,
  "async_rl": true,
  "serial_mode": false,
  "batched_sampling": true,
  "num_batches_to_accumulate": 2,
  "worker_num_splits": 1,
  "policy_workers_per_policy": 1,
  "max_policy_lag": 1000,
  "num_workers": 16,
  "num_envs_per_worker": 2,
  "batch_size": 1024,
  "num_batches_per_epoch": 8,
  "num_epochs": 4,
  "rollout": 128,
  "recurrence": 1,
  "shuffle_minibatches": false,
  "gamma": 0.99,
  "reward_scale": 1.0,
  "reward_clip": 1000.0,
  "value_bootstrap": false,
  "normalize_returns": true,
  "exploration_loss_coeff": 0.0004677351413,
  "value_loss_coeff": 0.5,
  "kl_loss_coeff": 0.0,
  "exploration_loss": "entropy",
  "gae_lambda": 0.95,
  "ppo_clip_ratio": 0.1,
  "ppo_clip_value": 1.0,
  "with_vtrace": false,
  "vtrace_rho": 1.0,
  "vtrace_c": 1.0,
  "optimizer": "adam",
  "adam_eps": 1e-05,
  "adam_beta1": 0.9,
  "adam_beta2": 0.999,
  "max_grad_norm": 0.0,
  "learning_rate": 0.0003033891184,
  "lr_schedule": "linear_decay",
  "lr_schedule_kl_threshold": 0.008,
  "lr_adaptive_min": 1e-06,
  "lr_adaptive_max": 0.01,
  "obs_subtract_mean": 0.0,
  "obs_scale": 255.0,
  "normalize_input": true,
  "normalize_input_keys": [
    "obs"
  ],
  "decorrelate_experience_max_seconds": 0,
  "decorrelate_envs_on_one_worker": true,
  "actor_worker_gpus": [],
  "set_workers_cpu_affinity": true,
  "force_envs_single_thread": false,
  "default_niceness": 0,
  "log_to_file": true,
  "experiment_summaries_interval": 3,
  "flush_summaries_interval": 30,
  "stats_avg": 100,
  "summaries_use_frameskip": true,
  "heartbeat_interval": 10,
  "heartbeat_reporting_interval": 60,
  "train_for_env_steps": 100000000,
  "train_for_seconds": 10000000000,
  "save_every_sec": 120,
  "keep_checkpoints": 2,
  "load_checkpoint_kind": "latest",
  "save_milestones_sec": 1200,
  "save_best_every_sec": 5,
  "save_best_metric": "reward",
  "save_best_after": 100000,
  "benchmark": false,
  "encoder_mlp_layers": [
    512,
    512
  ],
  "encoder_conv_architecture": "convnet_atari",
  "encoder_conv_mlp_layers": [
    512
  ],
  "use_rnn": false,
  "rnn_size": 512,
  "rnn_type": "gru",
  "rnn_num_layers": 1,
  "decoder_mlp_layers": [],
  "nonlinearity": "relu",
  "policy_initialization": "orthogonal",
  "policy_init_gain": 1.0,
  "actor_critic_share_weights": true,
  "adaptive_stddev": false,
  "continuous_tanh_scale": 0.0,
  "initial_stddev": 1.0,
  "use_env_info_cache": false,
  "env_gpu_actions": false,
  "env_gpu_observations": true,
  "env_frameskip": 4,
  "env_framestack": 4,
  "pixel_format": "CHW"
}}\n
  ```
\n
    """  

    readme += f"""
A(n) **{algo}** model trained on the **{env}** environment.\n
This model was trained using Sample-Factory 2.0: https://github.com/alex-petrenko/sample-factory. Sample factory is a 
high throughput on-policy RL framework. I have been using 
Documentation for how to use Sample-Factory can be found at https://www.samplefactory.dev/\n\n
## Downloading the model\n
After installing Sample-Factory, download the model with:
```
python -m sample_factory.huggingface.load_from_hub -r {repo_id}
```\n
    """

    if enjoy_name is None:
        enjoy_name = "<path.to.enjoy.module>"

    readme += f"""
## Using the model\n
To run the model after download, use the `enjoy` script corresponding to this environment:
```
python -m {enjoy_name} --algo={algo} --env={env} --train_dir=./train_dir --experiment={repo_name}
```
\n
You can also upload models to the Hugging Face Hub using the same script with the `--push_to_hub` flag.
See https://www.samplefactory.dev/10-huggingface/huggingface/ for more details
    """

    if train_name is None:
        train_name = "<path.to.train.module>"

    readme += f"""
## Training with this model\n
To continue training with this model, use the `train` script corresponding to this environment:
```
python -m {train_name} --algo={algo} --env={env} --train_dir=./train_dir --experiment={repo_name} --restart_behavior=resume --train_for_env_steps=10000000000
```\n
Note, you may have to adjust `--train_for_env_steps` to a suitably high number as the experiment will resume at the number of steps it concluded at.
    """

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme)

    metadata = {}
    metadata["library_name"] = "sample-factory"
    metadata["tags"] = [
        "deep-reinforcement-learning",
        "reinforcement-learning",
        "sample-factory",
    ]

    if rewards is not None:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

        eval = repocard.metadata_eval_result(
            model_pretty_name=algo,
            task_pretty_name="reinforcement-learning",
            task_id="reinforcement-learning",
            metrics_pretty_name="mean_reward",
            metrics_id="mean_reward",
            metrics_value=f"{mean_reward:.2f} +/- {std_reward:.2f}",
            dataset_pretty_name=env,
            dataset_id=env,
        )

        metadata = {**metadata, **eval}

    repocard.metadata_save(readme_path, metadata)


def push_to_hf(dir_path: str, repo_name: str):
    repo_url = HfApi().create_repo(
        repo_id=repo_name,
        private=False,
        exist_ok=True,
    )

    upload_folder(
        repo_id=repo_name,
        folder_path=dir_path,
        path_in_repo=".",
        ignore_patterns=[".git/*"],
    )

    log.info(f"The model has been pushed to {repo_url}")


def load_from_hf(dir_path: str, repo_id: str):
    temp = repo_id.split("/")
    repo_name = temp[1]

    local_dir = os.path.join(dir_path, repo_name)
    Repository(local_dir, repo_id)
    log.info(f"The repository {repo_id} has been cloned to {local_dir}")
