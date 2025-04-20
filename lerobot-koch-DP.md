## Diffusion Policy多目标任务抓取训练

**任务描述**：本任务旨在利用具身智能系统(基于koch机器人)完成多目标抓取任务，探索Diffusion Policy的训练要点，引入视觉大模型，增加场景理解，提高模型的泛化能力。

## 数据采集-多目标抓取

1. 机械臂采用ACT训练的机械臂，标定、测试等见：

2. 数据采集，准备3个物品：橡皮泥小碗、象皮、胶带，

3. 数据采集要求：

   采集之前需要先调节摄像头角度，确保顶部视角和侧视角可以清晰全面的看清楚从动机械臂的完整动作，在整个操作过程中，主动臂和操作员不能进入视野范围

4. 测试摄像头角度：

    ***laptop phone的图片***

5. 录制训练数据

- 测试能否跑通遥操作

  ```bash
  python lerobot/scripts/control_robot.py teleoperate --robot-path lerobot/configs/robot/koch.yaml
  ```

- 采集数据集

  - 数据推送到huggingface，必须在当前环境中登录

  ```bash
  huggingface-cli login --token hf_token--add-to-git-credential
  HF_USER=$(huggingface-cli whoami | head -n 1)
  echo $HF_USER
  ```

  - 数据采集

  ```bash
  python lerobot/scripts/control_robot.py record \   # 运行control_robot.py中record子命令
      --robot-path lerobot/configs/robot/koch.yaml \
      --fps 30 \
      --root data \                                  # 本地文件夹
      --repo-id $HF_USER/koch_grasp_multiple_objects. \       # 文件夹下的目录，hugging face中也是这种目录
      --tags koch tutorial \
      --warmup-time-s 5 \                            # 预热时间，以防前几帧图片质量不好
      --episode-time-s 25  \                         # 一个episode的时间，s，遥操动作做完的时间；
      --reset-time-s 10  \                           # 把场景人为复原的时间
      --num-episodes 150  \                           # 采集的数据量
      --push-to-hub 1 \                              # 是否上传到hugging face， 0-否，1-是
      --force-override 0                             # 采集数据是否覆盖之前的数据；
  ```

- num-episodes ：抓取1个物体，50个episodes，episode-time-s=15s；抓取2个，50个episode，sepisode-time-s=20s；抓取3个，100个episodes，episode-time-s=25s，共计200个episodes；

- **←（左方向键）**：当前采集失败，重新录制（**回到起始状态，不保存**）；**Enter（回车键）**：当前采集成功，保存为一个 Episode；**q** 或 **Ctrl+C**：中止录制进程或退出采集脚本

- 在采集过程中，num-episodes =200，分批采集即可；

6. 数据展示



## DP训练和推理完整流程



## 视觉大模型



## 

## Diffusion Model (扩散模型) 的完整数据处理流程

1. 输入数据预处理

```python
# 输入数据准备
observation = robot.capture_observation()         # 包含相机图像和机器人状态
normalized_data = policy.normalize_inputs(observation)   # 数据归一化处理
```

```python
{
    "observation": {
        "images": {
            "camera_1": [B, T, H, W, 3],  # B批次大小，T个时间步的图像序列
            "camera_2": [B, T, H, W, 3]   # 多相机可选
        },
        "state": [B, T, state_dim]  # T个时间步的机器人状态
    },
    "action": [B, T, action_dim]  # T个时间步的动作序列
}
```

2. 视觉特征提取与全局条件准备

```python
# 准备全局条件特征
batch_size, n_obs_steps = batch["observation.state"].shape[:2]  # [B, T, state_dim]
global_cond_feats = [batch["observation.state"]]  # 首先添加状态向量 [B, T, state_dim]

# 多相机视觉特征提取
if self._use_images:
    if self.config.use_separate_rgb_encoder_per_camera:
        # 对每个相机使用独立的编码器
        # 重新排列输入：[B, T, N, C, H, W] -> [N, (B*T), C, H, W]
        images_per_camera = einops.rearrange(batch["observation.images"], "b s n ... -> n (b s) ...")
        
        # 对每个相机编码，得到 N 个 [(B*T), feature_dim] 的特征向量
        img_features_list = torch.cat(
            [encoder(images) for encoder, images in zip(self.rgb_encoder, images_per_camera)]
        )  # [(N*B*T), feature_dim]
        
        # 重新排列: [(N*B*T), feature_dim] -> [B, T, (N*feature_dim)]
        img_features = einops.rearrange(
            img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        )
    else:
        # 所有相机共享一个编码器
        # 重新排列: [B, T, N, C, H, W] -> [(B*T*N), C, H, W]
        flat_images = einops.rearrange(batch["observation.images"], "b s n ... -> (b s n) ...")
        
        # 编码后得到 [(B*T*N), feature_dim]
        img_features = self.rgb_encoder(flat_images)
        
        # 重新排列: [(B*T*N), feature_dim] -> [B, T, (N*feature_dim)]
        img_features = einops.rearrange(
            img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
        )
    
    # 将图像特征添加到条件特征列表
    global_cond_feats.append(img_features)  # [B, T, (N*feature_dim)]

# 如果使用环境状态
if self._use_env_state:
    # 添加环境状态 [B, env_dim]
    global_cond_feats.append(batch["observation.environment_state"])

# 拼接特征然后展平
# [B, T, combined_dim] -> [B, (T*combined_dim)]
global_cond = torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
```

**解释：**
- 这部分处理多个观测源的特征提取和融合，包括状态数据和图像数据
- 支持多相机配置，可以为每个相机使用独立编码器或共享一个编码器
- 对图像特征和状态特征进行拼接，形成一个全局条件向量
- 对时序信息进行展平处理，便于后续使用

**用途：**
- 提取图像中的高级视觉特征，降低原始像素的维度
- 将多模态信息（图像、状态、环境信息）统一为一个条件向量
- 这个全局条件向量将作为UNet的条件输入，指导动作生成过程
- 通过不同相机视角的信息融合，增强对环境的感知

## 3）DiffusionRgbEncoder视觉编码器详细处理流程

```python
def forward(self, x: Tensor) -> Tensor:
    """
    Args:
        x: [B, C, H, W] 像素值在[0, 1]范围内的图像张量
    Returns:
        [B, feature_dim] 图像特征向量
    """
    # 预处理：根据配置进行裁剪 [B, C, H, W] -> [B, C, crop_H, crop_W]
    if self.do_crop:
        if self.training:
            x = self.maybe_random_crop(x)  # 训练时可能随机裁剪
        else:
            x = self.center_crop(x)  # 评估时始终使用中心裁剪
            
    # 提取骨干网络特征 [B, C, crop_H, crop_W] -> [B, backbone_C, h, w]
    backbone_features = self.backbone(x)
    
    # 使用空间Softmax提取关键点 [B, backbone_C, h, w] -> [B, num_kp, 2]
    spatial_features = self.pool(backbone_features)
    
    # 展平特征 [B, num_kp, 2] -> [B, num_kp*2]
    x = torch.flatten(spatial_features, start_dim=1)
    
    # 最终线性层和非线性激活 [B, num_kp*2] -> [B, feature_dim]
    x = self.relu(self.out(x))
    return x
```

**解释：**
- 这是处理图像的核心编码器，将RGB图像编码为固定维度的特征向量
- 首先进行可选的图像裁剪（训练时可随机裁剪增强数据多样性）
- 使用预训练的ResNet等骨干网络提取深层特征
- 采用空间Softmax操作提取关键点特征，这是机器人视觉领域的常用技术
- 最后通过线性层和ReLU激活输出最终特征

**用途：**
- 将高维图像数据压缩为低维特征表示，大幅降低计算复杂度
- 空间Softmax可以捕获图像中的关键点信息，对物体位置特别敏感
- 提供与任务相关的视觉表示，为后续动作生成提供重要视觉线索
- 随机裁剪等数据增强技术提高模型对各种视角变化的鲁棒性

## 4）前向扩散过程（训练时）

```python
# 获取原始干净的动作轨迹 [B, horizon, action_dim]
trajectory = batch["action"]

# 采样噪声添加到轨迹 [B, horizon, action_dim]
eps = torch.randn(trajectory.shape, device=trajectory.device)

# 为批次中的每个样本随机采样一个噪声时间步 [B]
timesteps = torch.randint(
    low=0,
    high=self.noise_scheduler.config.num_train_timesteps,
    size=(trajectory.shape[0],),
    device=trajectory.device,
).long()

# 根据每个时间步的噪声幅度将噪声添加到干净轨迹
# [B, horizon, action_dim] -> [B, horizon, action_dim]
noisy_trajectory = self.noise_scheduler.add_noise(trajectory, eps, timesteps)
```

**解释：**
- 这是扩散模型训练的核心步骤之一，实现从干净数据到噪声数据的前向扩散过程
- 为每个样本随机选择一个时间步，对应扩散过程中的不同阶段
- 根据选定的时间步，使用噪声调度器将噪声添加到原始动作轨迹中
- 不同时间步对应不同噪声程度，从轻微噪声到完全噪声

**用途：**
- 为扩散模型创建训练样本，让模型学习从噪声中恢复原始动作的能力
- 随机时间步的采样让模型学习扩散过程的所有阶段，而不仅仅是某一固定阶段
- 通过噪声调度器控制噪声添加的程度和方式，如线性、余弦或平方根衰减
- 这个过程模拟了动作轨迹的随机破坏过程，为后续的去噪训练准备数据

## 5）U-Net去噪网络

```python
# U-Net的输入：
# noisy_trajectory: [B, horizon, action_dim]
# timesteps: [B]
# global_cond: [B, global_cond_dim]
# 输出: [B, horizon, action_dim]
pred = self.unet(noisy_trajectory, timesteps, global_cond=global_cond)

# U-Net的前向传播详细过程
def forward(self, x, timestep, global_cond=None):
    # 转换为1D卷积所需的格式 [B, T, D] -> [B, D, T]
    x = einops.rearrange(x, "b t d -> b d t")
    
    # 时间步嵌入 [B] -> [B, diffusion_step_embed_dim]
    timesteps_embed = self.diffusion_step_encoder(timestep)
    
    # 如果有全局条件特征，与时间步嵌入拼接
    # [B, diffusion_step_embed_dim] + [B, global_cond_dim] -> [B, cond_dim]
    if global_cond is not None:
        global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
    else:
        global_feature = timesteps_embed
    
    # 运行编码器，跟踪跳跃连接特征
    encoder_skip_features = []  # 存储每层的特征用于跳跃连接
    # 初始 x: [B, action_dim, T]
    for resnet, resnet2, downsample in self.down_modules:
        x = resnet(x, global_feature)  # [B, dim_out, T]
        x = resnet2(x, global_feature)  # [B, dim_out, T]
        encoder_skip_features.append(x)  # 添加到跳跃连接列表
        x = downsample(x)  # [B, dim_out, T/2] (除了最后一层)
    
    # 中间处理模块
    # x: [B, down_dims[-1], T/(2^L)]
    for mid_module in self.mid_modules:
        x = mid_module(x, global_feature)  # [B, down_dims[-1], T/(2^L)]
    
    # 运行解码器，使用编码器的跳跃连接特征
    for resnet, resnet2, upsample in self.up_modules:
        # 与编码器对应层的特征拼接 [B, dim_in, T/(2^l)] + [B, dim_in, T/(2^l)] -> [B, dim_in*2, T/(2^l)]
        x = torch.cat((x, encoder_skip_features.pop()), dim=1)
        x = resnet(x, global_feature)  # [B, dim_out, T/(2^l)]
        x = resnet2(x, global_feature)  # [B, dim_out, T/(2^l)]
        x = upsample(x)  # [B, dim_out, T/(2^(l-1))] (除了最后一层)
    
    # 最终卷积层 [B, down_dims[0], T] -> [B, action_dim, T]
    x = self.final_conv(x)
    
    # 转回原始格式 [B, action_dim, T] -> [B, T, action_dim]
    x = einops.rearrange(x, "b d t -> b t d")
    return x
```

**解释：**
- U-Net是扩散模型的核心组件，负责学习从噪声轨迹中恢复原始轨迹
- 采用1D卷积处理时序动作数据，与传统图像处理中的2D U-Net类似
- 使用双路径结构：下采样路径压缩特征，上采样路径还原维度
- 使用跳跃连接保留下采样过程中的细节信息
- 通过FiLM（特征线性调制）机制将全局条件融入每个卷积块

**用途：**
- 学习从不同噪声水平的轨迹中恢复原始干净轨迹的能力
- 结合观测条件生成与当前环境相关的特定动作轨迹
- 通过多分辨率处理捕获动作序列中的短期和长期依赖关系
- 跳跃连接帮助保留下采样过程中可能丢失的细节信息
- 时间步编码让网络知道当前处理的是扩散过程中的哪个阶段

## 6）损失计算

```python
# 根据预测类型计算目标
# pred: [B, horizon, action_dim]
if self.config.prediction_type == "epsilon":
    target = eps  # 模型预测的是添加的噪声 [B, horizon, action_dim]
elif self.config.prediction_type == "sample":
    target = batch["action"]  # 模型预测的是原始轨迹 [B, horizon, action_dim]
else:
    raise ValueError(f"不支持的预测类型 {self.config.prediction_type}")

# 计算MSE损失 [B, horizon, action_dim]
loss = F.mse_loss(pred, target, reduction="none")

# 如果配置了掩码（针对填充动作），应用掩码
if self.config.do_mask_loss_for_padding:
    # action_is_pad: [B, horizon]
    if "action_is_pad" not in batch:
        raise ValueError("当启用填充掩码时，需要提供'action_is_pad'")
    in_episode_bound = ~batch["action_is_pad"]  # [B, horizon]
    # [B, horizon] -> [B, horizon, 1] 用于广播
    loss = loss * in_episode_bound.unsqueeze(-1)  # [B, horizon, action_dim]

# 返回平均损失，标量
return loss.mean()
```

**解释：**
- 这一步计算扩散模型的训练损失，支持两种预测目标：预测噪声或预测原始样本
- 对于预测噪声模式("epsilon")，模型学习估计添加到样本上的噪声
- 对于预测样本模式("sample")，模型直接学习从噪声数据恢复原始样本
- 使用均方误差作为损失函数度量预测与目标之间的差异
- 可选掩码机制忽略填充动作（数据集边缘的复制填充部分）的损失

**用途：**
- 提供优化信号指导模型学习从噪声中恢复动作轨迹
- 两种预测类型在理论上等价，但"epsilon"模式通常在实践中效果更好
- 掩码机制确保模型只对真实数据而非填充数据负责，提高训练质量
- 损失计算在动作的每个维度和时间步上，确保完整轨迹的精确重建

## 7）推理时的采样过程

```python
def conditional_sample(self, batch_size, global_cond=None, generator=None):
    # global_cond: [B, global_cond_dim]
    device = get_device_from_parameters(self)
    dtype = get_dtype_from_parameters(self)
    
    # 从先验分布采样 [B, horizon, action_dim]
    sample = torch.randn(
        size=(batch_size, self.config.horizon, self.config.output_shapes["action"][0]),
        dtype=dtype,
        device=device,
        generator=generator,
    )
    
    # 设置时间步
    self.noise_scheduler.set_timesteps(self.num_inference_steps)
    
    # 逐步去噪
    for t in self.noise_scheduler.timesteps:  # 从最大时间步递减到0
        # timestep_input: [B] 填充相同时间步值t
        timestep_input = torch.full(sample.shape[:1], t, dtype=torch.long, device=sample.device)
        
        # 预测模型输出 [B, horizon, action_dim]
        model_output = self.unet(sample, timestep_input, global_cond=global_cond)
        
        # 计算前一个样本：x_t -> x_t-1 [B, horizon, action_dim]
        sample = self.noise_scheduler.step(model_output, t, sample, generator=generator).prev_sample
    
    # 返回去噪后的样本 [B, horizon, action_dim]
    return sample
```

**解释：**
- 这是扩散模型的推理过程，从随机噪声开始逐步去噪生成动作轨迹
- 首先从标准正态分布采样初始噪声作为起点
- 设置推理时使用的时间步数（可能少于训练时的时间步）
- 按照降序遍历时间步（从最大噪声到最小噪声），逐步去噪
- 每一步使用U-Net预测当前噪声或原始样本，然后通过噪声调度器计算下一步状态

**用途：**
- 基于当前观测条件生成相应的动作轨迹
- 通过多步迭代去噪过程产生高质量、符合物理约束的动作序列
- 每一步去噪都考虑全局条件（视觉特征和状态），保证生成动作与环境一致
- 支持通过种子控制随机采样过程，可用于生成确定性或多样化的轨迹

## 8）动作选择流程

```python
@torch.no_grad
def select_action(self, batch):
    # batch: 包含当前观测的字典
    # 标准化输入
    batch = self.normalize_inputs(batch)
    
    # 如果使用图像，整合多相机图像
    if len(self.expected_image_keys) > 0:
        batch = dict(batch)  # 浅拷贝避免修改原始数据
        # 将多个相机图像堆叠 [B, C, H, W] * N -> [B, N, C, H, W]
        batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
    
    # 维护队列中的观测历史
    # self._queues: 字典，包含各类型观测和动作的队列
    self._queues = populate_queues(self._queues, batch)
    
    # 如果动作队列为空，运行模型生成新的动作序列
    if len(self._queues["action"]) == 0:
        # 堆叠最新的n个观测 -> [B, n_obs_steps, dim] 每种类型
        batch = {k: torch.stack(list(self._queues[k]), dim=1) for k in batch if k in self._queues}
        
        # 生成动作序列 [B, n_action_steps, action_dim]
        actions = self.diffusion.generate_actions(batch)
        
        # 反归一化输出
        actions = self.unnormalize_outputs({"action": actions})["action"]
        
        # 将预测的动作序列加入队列
        # [B, n_action_steps, action_dim] -> [n_action_steps, B, action_dim]
        self._queues["action"].extend(actions.transpose(0, 1))
    
    # 返回下一个动作 [B, action_dim]
    action = self._queues["action"].popleft()
    return action
```

**解释：**
- 这是模型实际与环境交互的接口，每次返回一个动作
- 维护观测历史队列，记录最近n_obs_steps个时间步的观测
- 采用缓冲策略：生成一系列动作并存储在队列中，每次调用只返回一个
- 当动作队列为空时才重新运行模型生成新的动作序列，提高效率
- 处理输入标准化和输出反标准化，确保动作值在原始范围内

**用途：**
- 将扩散模型集成到实时控制循环中，实现连续的机器人控制
- 通过记忆历史观测提供时序上下文，增强模型对环境动态的理解
- 通过动作缓冲减少计算开销，适应实时控制的速度要求
- 在训练-执行环路中提供平滑一致的动作输出，减少抖动和不连续性
- 支持批处理模式，可同时控制多个环境实例（如并行模拟）