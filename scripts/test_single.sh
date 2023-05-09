set -ex

START=$(date +%s.%N)

python test.py --dataroot {path_to_test_images} --name {name_of_experiment_set_in_training} --results_dir {path_to_results_set_in_training} --name {name_of_experiment_set_in_training} --model_suffix {_A_for_normalize_to_B_or_B_for_normalize_to_A} --model test --dataset_mode single --no_dropout

END=$(date +%s.%N)
DIFF=$(echo "$END - $START" | bc)

