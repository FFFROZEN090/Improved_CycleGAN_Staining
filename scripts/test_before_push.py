# Simple script to make sure basic usage
# such as training, testing, saving and loading
# runs without errors.
import os


def run(command):
    print(command)
    exit_status = os.system(command)
    if exit_status > 0:
        exit(1)


if __name__ == '__main__':
    # check if test_bfore_push datasets exists
    if not os.path.exists('./datasets/test_bfore_push'):
        print('no dataset at ./dataset/test_bfore_push found')
        exit(1)

    # cyclegan train/test
    nname = 'test_before_push_cyclegan'
    run('python train.py --model cycle_gan --name test_before_push_cyclegan --dataroot ./datasets/test_bfore_push --n_epochs 1 --n_epochs_decay 0 --save_latest_freq 10  --print_freq 1 --display_id -1')
    run('python test.py --model cycle_gan --name test_before_push_cyclegan --dataroot ./datasets/test_bfore_push --num_test 1 --phase train --suffix model_cycle_gan')
    run('python test.py --model test --name test_before_push_cyclegan --dataroot ./datasets/test_bfore_push --num_test 1 --model_suffix "_A" --phase train --no_dropout --suffix model_test')
