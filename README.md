[![tests](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml/badge.svg?branch=master)](https://github.com/alex-petrenko/sample-factory/actions/workflows/test-ci.yml)
[![codecov](https://codecov.io/gh/alex-petrenko/sample-factory/branch/master/graph/badge.svg?token=9EHMIU5WYV)](https://codecov.io/gh/alex-petrenko/sample-factory)
[![pre-commit](https://github.com/alex-petrenko/sample-factory/actions/workflows/pre-commit.yml/badge.svg?branch=master)](https://github.com/alex-petrenko/sample-factory/actions/workflows/pre-commit.yml)
[![docs](https://github.com/alex-petrenko/sample-factory/actions/workflows/docs.yml/badge.svg)](https://samplefactory.dev)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/alex-petrenko/sample-factory/blob/master/LICENSE)
[![Downloads](https://pepy.tech/badge/sample-factory)](https://pepy.tech/project/sample-factory)
[<img src="https://img.shields.io/discord/987232982798598164?label=discord">](https://discord.gg/BCfHWaSMkr)
<!-- [![pre-commit.ci status](https://results.pre-commit.ci/badge/github/wmFrank/sample-factory/master.svg)](https://results.pre-commit.ci/latest/github/wmFrank/sample-factory/master)-->
<!-- [![wakatime](https://wakatime.com/badge/github/alex-petrenko/sample-factory.svg)](https://wakatime.com/badge/github/alex-petrenko/sample-factory)-->

# SOTA Quest

This is an attempt by a hobbyist working in a resource-constrained context to create state of the art (SOTA) reinforcement learning models using sample factory. An excellent open source framework developed by Alex Petrenko. In essence it allows you to squeeze maximal juice out of your reinforcement learning (RL) hardware. Thus, it is highly valuable to someone like me who doesn't have access to a high-performance compute cluster but does have access to some GPU's.

# Understanding Key Concepts

PPO or proximal policy optimisation is a state of the art RL algorithm that looks to build on prior on-policy algorithms like reinforce by adding something called a clip-function. This helps to prevent too-big (destructive) policy updates from occuring which hampered prior algorithms and makes the algorithm very generally useful in a wide variety of settings. PPO is not the 'best' algorithm currently available (see MuZero, Agent 57 and Go-Explore for better examples) but it is very powerful and accessible to hobbyists which is why huggingface have focused on it. 

PPO is not very sample efficient in that it requires a lot of examples to learn from. This is why Alex created Sample Factory to try and overcome some of the limitations inherent in more accessible implementations such as SB3 and cleanRL. I am one of the benficiaires of this effort. 

# Objectives of Project

The aim is to reach SOTA performance in as many environments as I can and learn a lot about RL and particularly on-policy algorithms in the process. 

I have started with the Atari 57 environments as they provide a canonical baseline which is both well documented and also express a very general list of capabilities: https://paperswithcode.com/task/atari-games. The big companies are probably starting to move away from these now because they are largely considered 'solved' but there are still some games such as 'Venture' and 'Tutankham' who have been more generally ignored. Additionally games like 'Solaris' and 'Breakout' cannot be considered 'solved' as even SOTA algorithms obtain relatively meagre scores on these games. The best algorithms can only clear the Breakout board twice and then tend to get stuck in loops - a fact that is typically ignored.

# Progress So Far

So far I have SOTA completed two of the easier Atari Environments:
- https://huggingface.co/MattStammers/appo-atari_freeway-sota
- https://huggingface.co/MattStammers/appo-atari_pong-sota

Four environments are approaching SOTA performance:
- https://huggingface.co/MattStammers/appo-atari_tennis-approaching_sota
- https://huggingface.co/MattStammers/appo-atari_fishingderby-approaching_sota
- https://huggingface.co/MattStammers/appo-atari_atlantis-approaching_sota
- https://huggingface.co/MattStammers/appo-atari_boxing-approaching_sota

These environments should be considered the generally easier ones to solve as evidenced by the timelines of the papers with code. I will not be able to sota them all (no algorithm can) but I should be able to prove that even a hobbyist can 'do' these things and make significant headway in their spare time. The aim is to achieve at least near SOTA results in at least 25% of the environments by the end of this year. This simply wouldn't be possible without sample-factory and the high performance it brings to the experiments. 

# Usage Notes

To use this version of sample factory you will need to do the following in order:

```sh
git clone https://github.com/MattStammers/sample-factory-sotaquest.git /
cd sample-factory-sotaquest  /
pip install pipenv /
pipenv --python={python version} /
pipenv shell /
pip install --upgrade setuptools pip wheel /
pip install -e . /
```

You may also need to run the below or a variant of it depending on your GPU setup - making sure the CUDA versions match exactly

```sh
pip install nvidia-cuda-runtime-cu11 --index-url https://pypi.ngc.nvidia.com --upgrade --force-reinstall
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --upgrade --force-reinstall
```

Then you can run the script to run the atari games (however you must change your huggingface name and load the CLI credentials for this and WandB first or the models will not push to the hub):
```sh
bash ./scripts/atari_algo_runner.sh
```

# Issues
Good luck. Feel free to reach out if you want help. Below is the original SF documentation which explains the framework excellently. 

The one thing which confused me initially is the --ALGO flag in the config. This as far as I can tell is just a placeholder which COULD be implemented if needed but SF is purely for on-policy RF (primarily PPO). There is a way to activate V-trace in the config and thus make the model an IMPALA-PPO model but SF is not designed for off-policy algo's like DQN.

# Sample Factory

High-throughput reinforcement learning codebase. Version 2.0.0 is out! 🤗

**Resources:**

* **Documentation:** [https://samplefactory.dev](https://samplefactory.dev) 

* **Paper:** https://arxiv.org/abs/2006.11751

* **Citation:** [BibTeX](https://github.com/alex-petrenko/sample-factory#citation)

* **Discord:** [https://discord.gg/BCfHWaSMkr](https://discord.gg/BCfHWaSMkr)

* **Twitter (for updates):** [@petrenko_ai](https://twitter.com/petrenko_ai)

* **Talk (circa 2021):** https://youtu.be/lLG17LKKSZc

### What is Sample Factory?

Sample Factory is one of the fastest RL libraries.
We focused on very efficient synchronous and asynchronous implementations of policy gradients (PPO). 

Sample Factory is thoroughly tested, used by many researchers and practitioners, and is actively maintained.
Our implementation is known to reach SOTA performance in a variety of domains in a short amount of time.
Clips below demonstrate ViZDoom, IsaacGym, DMLab-30, Megaverse, Mujoco, and Atari agents trained with Sample Factory:

<p align="middle">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/vizdoom.gif?raw=true" width="360" alt="VizDoom agents traned using Sample Factory 2.0">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/isaac.gif?raw=true" width="360" alt="IsaacGym agents traned using Sample Factory 2.0">
<br/>
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/dmlab.gif?raw=true" width="380" alt="DMLab-30 agents traned using Sample Factory 2.0">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/megaverse.gif?raw=true" width="340" alt="Megaverse agents traned using Sample Factory 2.0">
<br/>
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/mujoco.gif?raw=true" width="390" alt="Mujoco agents traned using Sample Factory 2.0">
<img src="https://github.com/alex-petrenko/sf_assets/blob/main/gifs/atari.gif?raw=true" width="330" alt="Atari agents traned using Sample Factory 2.0">
</p>

**Key features:**

* Highly optimized algorithm [architecture](https://www.samplefactory.dev/06-architecture/overview/) for maximum learning throughput
* [Synchronous and asynchronous](https://www.samplefactory.dev/07-advanced-topics/sync-async/) training regimes
* [Serial (single-process) mode](https://www.samplefactory.dev/07-advanced-topics/serial-mode/) for easy debugging
* Optimal performance in both CPU-based and [GPU-accelerated environments](https://www.samplefactory.dev/09-environment-integrations/isaacgym/)
* Single- & multi-agent training, self-play, supports [training multiple policies](https://www.samplefactory.dev/07-advanced-topics/multi-policy-training/) at once on one or many GPUs
* Population-Based Training ([PBT](https://www.samplefactory.dev/07-advanced-topics/multi-policy-training/))
* Discrete, continuous, hybrid action spaces
* Vector-based, image-based, dictionary observation spaces
* Automatically creates a model architecture by parsing action/observation space specification. Supports [custom model architectures](https://www.samplefactory.dev/03-customization/custom-models/)
* Library is designed to be imported into other projects, [custom environments](https://www.samplefactory.dev/03-customization/custom-environments/) are first-class citizens
* Detailed [WandB and Tensorboard summaries](https://www.samplefactory.dev/05-monitoring/metrics-reference/), [custom metrics](https://www.samplefactory.dev/05-monitoring/custom-metrics/)
* [HuggingFace 🤗 integration](https://www.samplefactory.dev/10-huggingface/huggingface/) (upload trained models and metrics to the Hub)
* [Multiple](https://www.samplefactory.dev/09-environment-integrations/mujoco/) [example](https://www.samplefactory.dev/09-environment-integrations/atari/) [environment](https://www.samplefactory.dev/09-environment-integrations/vizdoom/) [integrations](https://www.samplefactory.dev/09-environment-integrations/dmlab/) with tuned parameters and trained models

This Readme provides only a brief overview of the library.
Visit full documentation at [https://samplefactory.dev](https://samplefactory.dev) for more details.

## Installation

Just install from PyPI:

```pip install sample-factory```

SF is known to work on Linux and macOS. There is no Windows support at this time.
Please refer to the [documentation](https://samplefactory.dev) for additional environment-specific installation notes.

## Quickstart

Use command line to train an agent using one of the existing integrations, e.g. Mujoco (might need to run `pip install sample-factory[mujoco]`):

```bash
python -m sf_examples.mujoco.train_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Stop the experiment (Ctrl+C) when the desired performance is reached and then evaluate the agent:

```bash
python -m sf_examples.mujoco.enjoy_mujoco --env=mujoco_ant --experiment=Ant --train_dir=./train_dir
```

Do the same in a pixel-based VizDoom environment (might need to run `pip install sample-factory[vizdoom]`, please also see docs for VizDoom-specific instructions):

```bash
python -m sf_examples.vizdoom.train_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir --num_workers=16 --num_envs_per_worker=10 --train_for_env_steps=1000000
python -m sf_examples.vizdoom.enjoy_vizdoom --env=doom_basic --experiment=DoomBasic --train_dir=./train_dir
```

Monitor any running or completed experiment with Tensorboard:

```bash
tensorboard --logdir=./train_dir
```
(or see the docs for WandB integration).

To continue from here, copy and modify one of the existing env integrations to train agents in your own custom environment. We provide
examples for all kinds of supported environments, please refer to the [documentation](https://samplefactory.dev) for more details.

## Acknowledgements

This project would not be possible without amazing contributions from many people. I would like to thank:

* [Vladlen Koltun](https://vladlen.info) for amazing guidance and support, especially in the early stages of the project, for
helping me solidify the ideas that eventually became this library.
* My academic advisor [Gaurav Sukhatme](https://viterbi.usc.edu/directory/faculty/Sukhatme/Gaurav) for supporting this project
over the years of my PhD and for being overall an awesome mentor.
* [Zhehui Huang](https://zhehui-huang.github.io/) for his contributions to the original ICML submission, his diligent work on
testing and evaluating the library and for adopting it in his own research.
* [Edward Beeching](https://edbeeching.github.io/) for his numerous awesome contributions to the codebase, including
hybrid action distributions, new version of the custom model builder, multiple environment integrations, and also
for promoting the library through the HuggingFace integration!
* [Andrew Zhang](https://andrewzhang505.github.io/) and [Ming Wang](https://www.mingwang.me/) for numerous contributions to the codebase and documentation during their HuggingFace internships!
* [Thomas Wolf](https://thomwolf.io/) and others at HuggingFace for the incredible (and unexpected) support and for the amazing
work they are doing for the open-source community.
* [Erik Wijmans](https://wijmans.xyz/) for feedback and insights and for his awesome implementation of RNN backprop using PyTorch's `PackedSequence`, multi-layer RNNs, and other features!
* [Tushar Kumar](https://www.linkedin.com/in/tushartk/) for contributing to the original paper and for his help
with the [fast queue implementation](https://github.com/alex-petrenko/faster-fifo).
* [Costa Huang](https://costa.sh/) for developing CleanRL, for his work on benchmarking RL algorithms, and for awesome feedback
and insights!
* [Denys Makoviichuk](https://github.com/Denys88/rl_games) for developing rl_games, a very fast RL library, for inspiration and 
feedback on numerous features of this library (such as return normalizations, adaptive learning rate, and others).
* [Eugene Vinitsky](https://eugenevinitsky.github.io/) for adopting this library in his own research and for his valuable feedback.
* All my labmates at RESL who used Sample Factory in their projects and provided feedback and insights!

Huge thanks to all the people who are not mentioned here for your code contributions, PRs, issues, and questions!
This project would not be possible without a community!

## Citation

If you use this repository in your work or otherwise wish to cite it, please make reference to our ICML2020 paper.

```
@inproceedings{petrenko2020sf,
  author    = {Aleksei Petrenko and
               Zhehui Huang and
               Tushar Kumar and
               Gaurav S. Sukhatme and
               Vladlen Koltun},
  title     = {Sample Factory: Egocentric 3D Control from Pixels at 100000 {FPS}
               with Asynchronous Reinforcement Learning},
  booktitle = {Proceedings of the 37th International Conference on Machine Learning,
               {ICML} 2020, 13-18 July 2020, Virtual Event},
  series    = {Proceedings of Machine Learning Research},
  volume    = {119},
  pages     = {7652--7662},
  publisher = {{PMLR}},
  year      = {2020},
  url       = {http://proceedings.mlr.press/v119/petrenko20a.html},
  biburl    = {https://dblp.org/rec/conf/icml/PetrenkoHKSK20.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

For questions, issues, inquiries please join Discord. 
Github issues and pull requests are welcome! Check out the [contribution guidelines](https://www.samplefactory.dev/community/contribution/).
