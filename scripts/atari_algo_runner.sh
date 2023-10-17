#!/bin/bash

declare -a envs=(
"atari_alien"
"atari_amidar"
"atari_assault"
"atari_asterix"
"atari_asteroid"
"atari_atlantis"
"atari_bankheist"
"atari_battlezone"
"atari_beamrider"
"atari_berzerk"
"atari_bowling"
"atari_boxing"
"atari_breakout"
"atari_centipede"
"atari_choppercommand"
"atari_crazyclimber"
"atari_defender"
"atari_demonattack"
"atari_doubledunk"
"atari_enduro"
"atari_fishingderby"
"atari_freeway"
"atari_frostbite"
"atari_gopher"
"atari_gravitar"
"atari_hero"
"atari_icehockey"
"atari_jamesbond"
"atari_kangaroo"
"atari_krull"
"atari_kongfumaster"
"atari_montezuma"
"atari_mspacman"
"atari_namethisgame"
"atari_phoenix"
"atari_pitfall"
"atari_pong"
"atari_privateye"
"atari_qbert"
"atari_riverraid"
"atari_roadrunner"
"atari_robotank"
"atari_seaquest"
"atari_skiing"
"atari_solaris"
"atari_spaceinvaders"
"atari_stargunner"
#"atari_surround"
"atari_tennis"
"atari_timepilot"
"atari_tutankham"
"atari_upndown"
"atari_venture"
"atari_videopinball"
"atari_wizardofwor"
"atari_yarsrevenge"
"atari_zaxxon"

)

declare -a algos=("APPO") 
declare -a seed="1234"

for env_name in "${envs[@]}"; do
    for algo in "${algos[@]}"; do
        echo "Training $env_name with $algo..."

        # Running training session
        python -m sf_examples.atari.train_atari --algo=$algo --env=$env_name --experiment=${env_name}_${algo} --num_policies=2 --restart_behavior="restart" --train_dir=./train_atari --train_for_env_steps=10000000 --seed=$seed \
        --num_workers=16  \
        --num_envs_per_worker=8  \
        --num_batches_per_epoch=8 \
        --worker_num_splits=2 \
        --async_rl=true \
        --batched_sampling=true \
        --batch_size=1024 \
        --max_grad_norm=0 \
        --learning_rate=0.0003033891184 \
        --heartbeat_interval=10 \
        --heartbeat_reporting_interval=60 \
        --save_milestones_sec=1200 \
        --num_epochs=4 \
        --exploration_loss_coeff=0.0004677351413 \
        --summaries_use_frameskip=False \
        --with_wandb=true \
        --wandb_user="matt-stammers" \
        --wandb_project="atari_$algo" \
        --wandb_group="$env_name" \
        --wandb_job_type="SF" \
        --wandb_tags="atari"  

        echo "Pushing results of $env_name with $algo to huggingface hub..."

        # Pushing results to huggingface hub
        python -m sf_examples.atari.enjoy_atari --algo=$algo --env=$env_name --experiment=${env_name}_${algo} --train_dir=./train_atari --max_num_episodes=10 --push_to_hub --hf_repository=MattStammers/${algo}-${env_name} --save_video --no_render --enjoy_script=sf_examples.atari.enjoy_atari --train_script=sf_examples.atari.train_atari 

    done
done

echo "All done!"

echo "All done!"
