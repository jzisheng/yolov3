#python3 train.py --data '/data/zjc4/chipped-30/xview_data.txt' --epochs=50 --resume
#mv weights/best.pt weights/best-30.pt
python3 train.py --data '/data/zjc4/chipped-90/xview_data.txt' --epochs=50 --resume
mv weights/best.pt weights/best-90.pt
