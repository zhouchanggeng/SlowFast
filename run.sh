
pgrep -f run_net.py|xargs kill -9
python tools/run_net.py --cfg configs/Kinetics/SLOWFAST_8x8_R50.yaml