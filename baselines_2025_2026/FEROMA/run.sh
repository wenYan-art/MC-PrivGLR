#!/bin/bash

k_folds=$(python -c "from public.config import k_folds; print(k_folds)")
model_name=$(python -c "from public.config import model_name; print(model_name)")
dataset_name=$(python -c "from public.config import dataset_name; print(dataset_name)")
strategy=$(python -c "from public.config import strategy; print(strategy)")
drifting_type=$(python -c "from public.config import drifting_type; print(drifting_type)")
non_iid_type=$(python -c "from public.config import non_iid_type; print(non_iid_type)")
n_clients=$(python -c "from public.config import n_clients; print(n_clients)")
n_rounds=$(python -c "from public.config import n_rounds; print(n_rounds)")
current_user=$(id -un)

echo -e "\n\033[1;36mExperiment settings:\033[0m\n\033[1;36m \
    MODEL: $model_name\033[0m\n\033[1;36m \
    Dataset: $dataset_name\033[0m\n\033[1;36m \
    Strategy: $strategy\033[0m\n\033[1;36m \
    Drifting type: $drifting_type\033[0m\n\033[1;36m \
    Data non-IID type: $non_iid_type\033[0m\n\033[1;36m \
    Number of clients: $n_clients\033[0m\n\033[1;36m \
    Number of rounds: $n_rounds\033[0m\n \
    \033[1;36mK-Folds: $k_folds\033[0m\n"

# Check non-IID type ['Px','Py','Px_y','Py_x']
# if [ "$non_iid_type" == "Px" ]; then
#     tot_scaling=2
# elif [ "$non_iid_type" == "Py" ]; then
#     tot_scaling=8
# elif [ "$non_iid_type" == "Py_x" ]; then
#     tot_scaling=2
# elif [ "$non_iid_type" == "Px_y" ]; then
#     tot_scaling=8
# else
#     echo -e "\n\033[1;31mError: non-IID type not recognized\033[0m\n"
#     exit 1
# fi
tot_scaling=0


# For loop across epoch_num [3, 5, 7, 10 20]
for epoch_num in 10; do
    echo -e "\n\033[1;36mStarting experiment with epoch_num: $epoch_num\033[0m\n"

    # For loop across scaling options
    for s in $(seq 0 $tot_scaling); do
        echo -e "\n\033[1;36mStarting experiment with scaling: $s and epoch_num: $epoch_num\033[0m\n"

        # K-Fold evaluation, if k_folds > 1
        for fold in $(seq 0 $(($k_folds - 1))); do
            echo -e "\n\033[1;36mStarting fold $((fold + 1)) - (scaling $s and epoch_num $epoch_num)\033[0m\n"

            # Clean and create datasets
            rm -rf data/cur_datasets/*
            if [ "$dataset_name" == "CheXpert" ]; then
                python public/chexpert_data_gen.py --fold "$fold" --scaling "$s" --epoch_num "$epoch_num" --n_clients "$n_clients"
            else
                python public/generate_datasets.py --fold "$fold" --scaling "$s" --epoch_num "$epoch_num"
            fi
 
            # exit

            cd "$strategy"

            python server.py --fold "$fold" &
            sleep 2  # Sleep for 2s to give the server enough time to start

            for i in $(seq 0 $(($n_clients - 1))); do
                echo "Starting client ID $i"
                # python client_dynamic.py --id "$i" &
                python client.py --id "$i" --fold "$fold" &
            done

            # This will allow you to use CTRL+C to stop all background processes
            trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
            # Wait for all background processes to complete
            wait

            # Clean up
            echo "Fold completed correctly"
            trap - SIGTERM 
            pkill -u "$current_user" -f client.py -9 >/dev/null 2>&1 || true
            pkill -u "$current_user" -f server.py -9 >/dev/null 2>&1 || true

            # Change back to the root directory
            cd ..
            sleep 2

        done

        # K-Fold evaluation, if k_folds > 1
        if [ "$k_folds" -gt 1 ]; then

            echo -e "\n\033[1;36mAveraging the results of all folds\033[0m\n"
            python public/average_results.py --epoch_num "$epoch_num" --scaling "$s"
        fi

    done
    
done


echo -e "\n\033[1;36mExperiment completed successfully\033[0m\n"
# kill
trap - SIGTERM && kill -- -$$
