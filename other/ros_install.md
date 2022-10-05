# ubuntu安装与ROS环境配置
本教程将从0开始安装ubuntu与配置ROS环境
## ubuntu安装
准备好ubuntu的官方iso文件，这里我建议下载ubuntu18或ubuntu20,其他版本由于与ros兼容的问题有很大的可能会出bug。
[ubuntu18官网](https://releases.ubuntu.com/18.04/)。同时还要制作启动u盘，这里我推荐使用[ventory](https://www.ventoy.net/cn/index.html),这是
一个制作启动u盘的工具，详细的使用教程可以参考[ventory使用方法](https://blog.csdn.net/qq_24330181/article/details/125486279)
## ROS配置
相信你已经按照教程安装好了ubuntu的初始版本，现在要开始配置ros环境。ubuntu18对应的ros版本是melodic，ubuntu相对应的ros版本可以查看[这里](https://blog.csdn.net/maizousidemao/article/details/119846292)

详细的配置教程可以参考[这里](https://blog.csdn.net/HHB791829200/article/details/122715008)。总结一下可以分为以下的步骤
1. 将源切换为国内的源，这里建议使用中科大的镜像源
2. 设置key
```commandline
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
```
3. 更新源
```commandline
sudo apt update
```
4. 安装ros
```commandline
sudo apt install ros-melodic-desktop-full
```
建议安装完整的版本
5. 配置环境变量
```commandline
echo "source /opt/ros/melodic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```
6. 安装依赖
```
sudo apt install python-rosdep python-rosinstall python-rosinstall-generator python-wstool build-essential
```
7. 安装rosdep
```commandline
sudo apt install python-rosdep
```
8. rosdep初始化
```
sudo apt init
```
注意:很多情况下rosdep init都会报错,这里使用以下方法
```commandline
sudo apt-get install python3-pip 
sudo pip install rosdepc
```
报错就用
```
sudo pip3 install rosdepc
```
完成之后在进行sudo rosdep init
9. rosdep升级
```commandline
rosdep update
```
这个命令一般会报错,因为中国这边的网络防火墙会阻止访问,解决方法有使用手机热点等方法,最简单的还是使用翻墙软件进行翻墙,可以参考这篇[教程](./defence wall.md)
10. 简单测试
只要小乌龟出现都算配置成功
分别打开3个终端
```
roscore
```
```
rosrun turtlesim turtlesim_node
```
```commandline
rosrun turtlesim turtle_teleop_key
```
### 一些小问题
安装ubuntu系统的时候要进入主机的bios模式,并把主机设置为u盘启动优先,详细的情况请百度给类型的主机u盘启动的教程.
选择硬盘安装的时候一定不要整个盘都 `全覆盖`,一定要先开辟一个新的工作盘,或者使用一个空的盘,详情自己百度如何建立双系统