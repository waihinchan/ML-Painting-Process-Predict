# readme

# scar

deep learning model for predict the process of an illustration

so far there only a pretrain model target at decoloring from manga. Further updating in progress...<br/>

****usage****: for the master branch/

1. put into ./checkpoint/color
2. run test.py
3. check the result at ./result

pretrain model:

[60_net_G.pth](https://drive.google.com/file/d/1-Y33Kh_-MfOozs5HxcDLKkcWUAC2XXCd/view?usp=sharing)

**********************************update*******************************************
****usage****: for the experiment branch/
1. run utils/make_pair_dataset.script
2. run main.py to make the pairwise_optimization
3. change option.py forward = 'seq'
4. run main.py again to make the seqwise_optimization

make sure the dataset format like:
1.jpg,2.jpg,3.jpg...last,if you have segmap please make a folder and put it in each data

see the demo at:
