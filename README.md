## 🦾 LeRobot Training Notes for Koch Arm

### 📁 参考内容：

- [hugging face lerobot](https://github.com/huggingface/lerobot)
- [lerobot-joycon](https://github.com/box2ai-robotics/lerobot-joycon)

本项目记录了使用 [LeRobot](https://github.com/huggingface/lerobot) 框架在 **Koch机械臂** 上训练多种策略（如 ACT、Diffusion Policy、Pi0）的过程与经验，包含数据采集、训练参数设置、硬件选型建议、模型源码解读等。

1. ACT和DP模型都是按照lerobot的V1版本的代码，根据[lerobot-joycon](https://github.com/box2ai-robotics/lerobot-joycon)增加DP的Unet transformer实现部分；
2. Pi0模型是按照[lerobot的V2版本的代码](https://github.com/huggingface/lerobot)；
3. Diffusion Policy训练过程中建议DP的Unet部分采用Transformer，参数少，性能较好；
   - Unet Resnet：278M（total parameters）
   - Unet Transformer：41M （total parameters）

---

### 🧩 机械臂平台说明

1. 选择已经装好的机械臂[WowRobo](https://wowrobo.com/tutorial)(课程带的)，Dynamixel舵机，双臂套餐4398元。（⚠️注：贵，不太好用）
2. 也可以选择自己3D打印机械臂进行配置，B站可以搜索教程，比较划算，双臂不到2000元。
3. 可以选择**Joy-Con 控制器** 来遥控或交互控制**一台机械臂**方式，这个价格比较便宜，[见链接](https://github.com/box2ai-robotics/lerobot-joycon)。
4. **注意事项**：
   - 夹爪承重有限，请务必选择轻质物体；
   - 抓取过重物体容易导致从机械臂剧烈晃动、卡顿、中止采集；
   - 要始终保持摄像头位置一致不变；

---

### 🛠 AutoDL 训练经验总结

1. ACT、DP建议选择RTX3090比较经济，1.58元/h，设置num_works=8，batch_size=32。
2. Pi0训练选择L20，98G显存比较合适。
3. 不要开启wandb=true，不然系统盘(一般30G)会随着其日志增多，崩掉。

### 📎 使用建议

- ✅ 1. 需要熟悉深度学习，以及Transformer原理，不需要报相关课程，市面上课程一言难尽，基本上都是讲给他们自己听的。
- ✅ 2. 如果你满足第1条，推荐使用 `Cursor`, `Copilot`, `WinSurf` 等 AI 编程工具，复现算法基本没问题；
- ✅ 3. 如果你使用**Joy-Con**，按照这个[教程](https://github.com/box2ai-robotics/lerobot-joycon)复现，基本没问题。

## 📄 [Lerobot ACT 训练](lerobot-koch-ACT.md)

- [Lerobt安装后数据及介绍](lerobot-koch-ACT.md#lerobt安装后数据及介绍)
- [机械臂硬件介绍](lerobot-koch-ACT.md#机械臂硬件介绍)
- [机器人初步部署和测试](lerobot-koch-ACT.md#机器人初步部署和测试)
- [主从臂零位校准](lerobot-koch-ACT.md#主从臂零位校准)
- [摄像头数据获取与检验](lerobot-koch-ACT.md#摄像头数据获取与检验)
- [遥操作数据采集](lerobot-koch-ACT.md#遥操作数据采集)
- [ACT训练和推理完整流程](lerobot-koch-ACT.md#act训练和推理完整流程)
- [ACT原理算法流程介绍](lerobot-koch-ACT.md#act原理算法流程介绍)

### 📺 ACT结果演示

<table>
  <tr>
    <td><strong>ACT-single-object</strong></td>
    <td><strong>ACT-multi-objects</strong></td>
  </tr>
  <tr>
    <td><img src="assets/single_object_ACT-ezgif.com-optimize.gif"/></td>
    <td><img src="assets/multi_obj_ACT-ezgif.com-optimize.gif"/></td>
  </tr>
</table>



## 📄 [Diffusion Policy 训练](lerobot-koch-DP.md)

- [Diffusion Policy多目标任务抓取训练](lerobot-koch-DP.md#diffusion-policy多目标任务抓取训练)
- [数据采集-多目标抓取](lerobot-koch-DP.md#数据采集-多目标抓取)
- [DP训练和推理完整流程](lerobot-koch-DP.md#dp训练和推理完整流程)
- [DP模型推理过程](lerobot-koch-DP.md#dp模型推理过程)
- [Diffusion Policy (扩散模型) 整数据处理流程](lerobot-koch-DP.md#diffusion-policy-扩散模型-整数据处理流程)

### 📺 DP结果演示

<table>
  <tr>
    <td style="text-align: center;">
      <strong>DP-multi-objects: 和 ACT 比较，可以捕捉 “如果需要把盒子夹近一些” 这个动作</strong><br>
      <img src="assets/multi_obj_DP-ezgif.com-optimize.gif" />
    </td>
  </tr>
</table>




## 📄 [Lerobot Pi0 训练](lerobot-koch-Pi.md)

- [基础配置环境安装](lerobot-koch-Pi0.md#基础配置环境安装)
- [Pi0数据格式介绍](lerobot-koch-Pi0.md#pi0数据格式介绍)
- [Pi0训练流程和推理](lerobot-koch-Pi0.md#pi0训练流程和推理)
- [Pi0算法原理和整体流程](lerobot-koch-Pi0.md#pi0算法原理和整体流程)
