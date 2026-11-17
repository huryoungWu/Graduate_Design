下面按「改进方向 → 代表模型/论文 → 核心改动 → 链接」整理一批中英文改进版 TCN，重点挑和时序预测、可复现、能写进论文对比实验相关的。

一、注意力增强型 TCN（最常用改进路线）

1. TCN-Attention / 动态注意力 TCN

• 思路：TCN 提局部特征 → 接 Temporal/Self-Attention 做全局加权

• 典型场景：光伏、太阳能、海浪、流量预测

• 论文：

  • A TCN-Attention Model for High-Accuracy Solar Energy Generation Forecasting（IEEE OAJPE 2026）

  - ITCN-A: Improved TCN + Attention + Dynamic ReLU + WOA 超参优化（MDPI 2024，海浪高度预测）

2. SE-TCN / SE-TCNeXt（通道注意力）

- 思路：在 TCN 残差块里插 Squeeze-Excitation，自动学“哪个变量/通道更重要”
• 优点：多变量预测里防特征平权，计算量还降了

• 论文：

  • An improved TCN with attention mechanism for photovoltaic forecasting（SE-TCN / SE-TCNeXt，ScienceDirect）

  - 中文应用：基于改进时序卷积网络的短时交通流预测（SE-TCN+RevIN） 
  • TCN-SENet++（岩土/工程预测）

3. DAMixer（TCN + 双阶段注意力）

- 思路：并行多卷积提特征 + 变量注意力 + 时间注意力
• 适合：多变量、强噪声、要可解释权重

• 论文：DAMixer, ScienceDirect 2025 

二、多尺度 / 空洞卷积改进（抓周期+趋势）

4. SCINet（Sample Convolution & Interaction）

• 路线：把序列下采样成奇偶子序列，分别卷积再交互，递归多分辨率

• 本质：TCN 思想的多尺度升级版，NeurIPS 2022

- 代码开源：https://github.com/cure-lab/SCINet
• 论文：SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction 

5. MSDCN（Multi-Scale Dilated Conv Net）

• 思路：浅层多分支空洞卷积（不同 kernel + dilation）+ AR 线性分支补趋势

• 论文：arXiv 2405.05499 Multi-Scale Dilated Convolution Network for Long-Term Time Series Forecasting 

6. TSCND（子序列空洞卷积 + 差分补偿）

• 痛点：标准因果膨胀卷积“越靠前信息越多，近期信息丢”

• 改动：子序列级膨胀卷积 + difference & compensation 抗分布漂移

• 论文：TSCND, ScienceDirect 2024 

7. ModernTCN（NeurIPS 2024）

• 思路：把 TCN 重构成“大卷积核 + 多尺度空洞 + 门控 + 周期归一化”

- 主打：用卷积打平 Transformer，速度快、参数量小
• 搜索名：ModernTCN: A Modern Pure Convolution Structure for Time Series

三、TCN + Transformer 混合（你做融合实验直接可用）

8. TCNFormer

• 结构：TCN 前置提局部 → Transformer Encoder（CT-MSA + TEA）→ 直出预测

- 场景：风速预测
• 论文：arXiv 2408.15737 

9. TCN-Transformer / SE-TCN-Transformer

• 双分支：TCN 抓局部，Transformer 抓跨变量长依赖，特征级融合

- 常配 Kalman Filter 后处理
• 论文：

  • Enhancing time series forecasting: a hybrid TCN-transformer approach（Neural Computing 2026）

  - SE-TCN–Transformer + Kalman（畜禽舍温湿度预测）

10. DAT-TimeXer（TCN 增强 TimeXer）

• 思路：在 TimeXer 分词前先过一层 TCN，补细粒度局部动态

- 金融多变量场景新工作（Springer 2026）

四、时空图卷积类（多节点管网/交通）

11. MSGWTCN（Graph Wavelet + Gated TCN）

• 空间：图小波多尺度；时间：门控膨胀因果卷积

• 场景：路网交通、管网节点

• 论文：Multi-Scale Graph Wavelet Temporal Convolution Network 

12. STGCN / ST-TCN 系列

• 时间维 TCN + 空间维 GCN，经典基线

五、中文期刊/学位常用改进表述（方便写文献综述）

• 多尺度 TCN（不同 kernel 并行）

- 空洞因果卷积 + 残差 + 门控（GLU）
• TCN + SE/CBAM 注意力

• TCN + RevIN（抗分布漂移）

• TCN + 小波分解（低频趋势 / 高频扰动分支路）

• TCN + ARIMA 残差修正

- TCN + 差分整合（类似 TSCND）

六、推荐你精读 5 篇（按优先级）

1. Bai et al. 2018 An Empirical Evaluation of Generic CNN Architectures for Time Series（TCN 原论文，必引）
2. SCINet NeurIPS 2022（多尺度卷积路线代表）
3. SE-TCN / SE-TCNeXt（通道注意力改进，工程好复现）
4. TCNFormer / TCN-Transformer（和你“TCN+Transformer 融合”方向最贴）
5. ModernTCN NeurIPS 2024（新基线，对比实验能打）

七、对你融合项目的落地方向建议

• 局部细节：TCN / ModernTCN / SCINet

• 全局依赖：iTransformer / TimeXer / Differential Attention

- 融合方式：TCN 输出 → 作为时序 token 或残差支路接入 Transformer
• 可解释：在 TCN 后接 SE 通道权重 + Attention map，做变量贡献分析（RCA 友好）

