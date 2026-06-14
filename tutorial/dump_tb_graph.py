"""将 FastSAC 网络的静态计算图写入 TensorBoard。

使用方式:
    uv run python tutorial/dump_tb_graph.py

然后在同目录启动 TensorBoard:
    tensorboard --logdir tutorial/tb_graph_logs --bind_all
"""
from torch.utils.tensorboard import SummaryWriter
import torch

# ----- 导入 UniLab 的 SAC 网络 -----
from unilab.algos.torch.fast_sac.learner import SACActor, SACCritic, DistributionalQNetwork

OBS_DIM = 48       # 示例：Ant-v5 的观测维度
ACTION_DIM = 8     # 示例：Ant-v5 的动作维度
BATCH = 4          # 小 batch，图更清晰

device = "cpu"

# 创建网络（用真实参数构造，与训练时一致）
actor = SACActor(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM,
    hidden_dim=512, use_layer_norm=True, use_tanh=True, device=device,
)
qnet = SACCritic(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM,
    num_atoms=101, hidden_dim=768, use_layer_norm=True, num_q_networks=2, device=device,
)
single_q = DistributionalQNetwork(
    obs_dim=OBS_DIM, action_dim=ACTION_DIM,
    num_atoms=101, hidden_dim=768, use_layer_norm=True, device=device,
)

# 构造输入
obs = torch.randn(BATCH, OBS_DIM, device=device)
actions = torch.randn(BATCH, ACTION_DIM, device=device)

BASE_DIR = "tutorial/tb_graph_logs"

# 每个网络独立 log_dir，避免 TensorBoard GRAPHS 标签冲突
graphs = [
    ("01_SACActor",             actor,     obs),
    ("02_SACCritic_x2",        qnet,      (obs, actions)),
    ("03_DistributionalQNet",  single_q,  (obs, actions)),
]

for tag, model, inp in graphs:
    w = SummaryWriter(log_dir=f"{BASE_DIR}/{tag}")
    w.add_graph(model, inp)
    w.close()
    print(f"[{tag}] 写入完成")

print(f"\n✅ 全部完成。启动 TensorBoard:")
print(f"   tensorboard --logdir tutorial/tb_graph_logs --bind_all")
print(f"\n   打开后顶部 TAB 选 GRAPHS → 左侧 Runs 下拉选择要查看的网络")
