![](darknet.png)
***

# tiny-tiny-tiny-yolo model(ttt5_224_160.cfg) training method  

**[pjreddie](https://pjreddie.com/darknet/yolov2/) recommends ensamble training for YOLO**.  
For ensamble training, we need **2-dataset imagenet and VOC**.  
And *classification task* by imagenet dataset and *object detection task* by VOC dataset.  

### requirement
bash  
python2.x(>2.7)  
opencv-python binding(ex. yum install -y opencv-python)  
wget command  
[darknet_sdl](https://github.com/k5iogura/darknet_sdl) or [darknet](https://pjreddie.com/darknet/yolov2/)    

### Why not darknet_ttt, Why darknet_sdl?
[darknet_ttt](https://github.com/k5iogura/darknet_ttt) for ttt5_224_160.cfg framework is dedicaded only for forwarding prediction. We modify darknet_sdl because to use FPGA as GEMM.  Slo, backwarding process is not maintenanced.  
Therefore for training process, we have to use [darknet_sdl](https://github.com/k5iogura/darknet_sdl) or [original darknet](https://pjreddie.com/darknet/yolov2/) framework.  
In this time, we are using [darknet_sdl](https://github.com/k5iogura/darknet_sdl) framework in bellow works.  

## classification task by imagenet

### preparation dataset
First of all, to get imagenet dataset, we need registration and downloading permission of WebMaster Stanford Univ. But not easy,,,  
Another way to get imagenet dataset is downloading URLs from imagenet Website and download many images using 'wget' command.  

**git clone [get_imagenet](https://github.com/k5iogura/get_imagenet), and use scripts in it.**  
these scripts perform downloading many jpeg images from internet, and finaly you get 3-files, trainval_1000c.txt, train_1000c.txt, val_1000c.txt.  
3-files mean jpeg image path-list on your PC for 1000 category training, validation and all.  
Images downloading from intenet takes a long time, **few days**.  

### run classification training

After long time, image downloading is done.  
For training process, we have to git clone [darknet_sdl](https://github.com/k5iogura/darknet_sdl).  

$ git clone https://github.com/k5iogura/darknet_sdl  
$ cd darknet_sdl  

Next is to create darknet control card(cfg/*.data) like belllow,  
$ cat classifier_500c.data  
classes=500  
train  = <'path'>/get_imagenet/train_500c.txt  
valid  = <'path'>/get_imagenet/val_500c.txt  
labels = <'path'>/get_imagenet/darknet.labels_500c.txt  
backup = INET  
top = 5  

And issue training command,  
$ mkdir INET  
$ darknet classifier train classifier_500c.data cfg/ttt5_pre.cfg  
Classification training takes a long time too, **1week on tesla v100**.  
Classification training may not finished after 1week.  
After training, you can see weights in INET/ directory.

## object detection task by VOC

### preparation dataset

Other side, VOC dataset downloading is easy.  We just type 3 commands,  
$ wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar  
$ wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar  
$ wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar  
detail of download VOC dataset is in [pjreddie Website](https://pjreddie.com/darknet/yolov2/)

### run object detection training

Next is to create darknet control card(cfg/*.data) like bellow,  
$ cat cfg/voc.data  
classes= 20  
train  = <'path'>/train.txt  
valid  = <'path'>/2012.txt  
names = data/voc.names  
backup = VOC_WEIGHTS  

And issu training command,  
$ mkdir VOC_WEIGHTS  
$ darknet partial cfg/ttt5_pre.cfg INET/ttt5_pre.weights INET/ttt5_pre_7.weights 7  
$ darknet detector train cfg/voc.data cfg/ttt5_224_160.cfg INET/ttt5_pre_7.weights INET/ttt5_pre_7.weights  
VOC object detection trainig take **a day on tesla v100**.  
After training, you can see weights file in VOC_WEIGHTS/ directory.  

We get accuracy of training by issue command bellow,  
$ darknet detector recall cfg/voc.data cfg/ttt5_224_160.cfg  VOC_WEIGHTS/ttt5_224_160_final.weights  
