
# Craftax

This repository contains the code for the machine learning course project, made by group 40. Thanks to original code at [Craftex_Baselines](https://github.com/MichaelTMatthews/Craftax_Baselines), our code is modified from it.

# Installation
```commandline
git clone https://github.com/ZhangCXLVII/Craftax.git
cd Craftax
pip install -r requirements.txt -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pre-commit install
```

# Run Experiments

To train a Transformer- or Mamba-based actor, please modify the corresponding comment in class ActorCritic inside models/actor-critic (around line 389).

### PPO
```commandline
python ppo.py
```


# Visualisation

We have released our trained model weights and configuration files at [huggingface](https://huggingface.co/Kupper/craftax_policy/tree/main). Please download the corresponding folder and use the following command for visualization:

```commandline
python view_ppo_agent.py --path <path to the downloaded folder, e.g., /home/username/Craftax/mam2/files>
```


If you train a policy from scratch, you can save trained policy with the `--save_policy` flag.  These can then be viewed with the `view_ppo_agent` script (pass in the path up to the `files` directory).
