import json
import numpy as np
import torch
import time
from importlib import import_module
from common.geometry import project_point_on_polyline
from planners.mind.scenario_tree import ScenarioTreeGenerator
from planners.mind.trajectory_tree import TrajectoryTreeOptimizer
from av2.datasets.motion_forecasting.data_schema import Track, ObjectState, TrackCategory, ObjectType


class MINDPlanner:
    def __init__(self, config_dir):
        self.planner_cfg = None
        self.network_cfg = None
        self.device = None
        self.network = None
        self.scen_tree_gen = None
        self.traj_tree_opt = None

        # 设置观察和规划的长度。
        self.obs_len = 50
        self.plan_len = 50

        # 初始化用于存储智能体观察和当前状态的字典，以及控制序列。
        self.agent_obs = {}
        self.state = None
        self.ctrl = None

        # 用于存储地面真值的目标车道和上一次控制序列。
        self.gt_tgt_lane = None
        self.last_ctrl_seq = []

        # 从指定目录读取配置文件并将其内容存储为规划配置。
        with open(config_dir, 'r') as file:
            self.planner_cfg = json.load(file)
        self.init_device()
        self.init_network()
        self.init_scen_tree_gen()
        self.init_traj_tree_opt()
        print("[MINDPlanner] Initialized.")

    def init_device(self):
        if self.planner_cfg['use_cuda'] and torch.cuda.is_available():
            self.device = torch.device("cuda", 0)
        else:
            self.device = torch.device('cpu')

    def init_network(self):
        self.network_cfg = import_module(self.planner_cfg['network_config']).NetCfg()
        
        # 获取网络配置
        net_cfg = self.network_cfg.get_net_cfg()
        
        # 解析网络配置，获取网络模块文件和网络类名
        net_file, net_name = net_cfg['network'].split(':')
        
        # 根据网络模块文件和类名动态导入并实例化网络模型
        # getattr(module, net_name): 从导入的模块（module）中通过属性名（net_name，通常也是一个字符串）获取对应的类或对象
        # getattr 是 Python 中的一个内置函数，用于从对象中获取属。当需要根据运行时的数据来决定访问哪个属性时，getattr 非常有用
        self.network = getattr(import_module(net_file), net_name)(net_cfg, self.device)
        
        # self.planner_cfg['ckpt_path']：这是一个字符串，表示模型检查点文件的路径
        # map_location：这是一个可选参数，用于指定如何将模型参数映射到当前设备。如果模型是在 GPU 上训练的，但在 CPU 上加载，或者需要在不同编号的 GPU 之间进行迁移，这个参数就非常重要
        # lambda storage, loc: storage：这是一个简单的 lambda 函数，它告诉 torch.load 不要将加载的数据移动到任何特定设备，而是直接返回存储在磁盘上的数据。这通常用于确保数据在加载时不会自动移动到 GPU，即使模型是在 GPU 上训练的也是如此
        # 这样做(确保所有数据都保留在内存中，而不是自动移动到 GPU)有啥目的？
        # 1. 如果模型是在 GPU 上训练的，但在没有 GPU 的环境下运行，直接加载可能会导致错误。通过 map_location=lambda storage, loc: storage 可以确保模型在 CPU 上也能正常加载
        # 2. 在多 GPU 环境下，可以更灵活地控制模型参数的分配。如果模型是在某个特定 GPU 上训练的，但需要在另一个 GPU 上加载，可以通过 map_location 参数来指定目标设备
        # 3. 如果模型非常大，而当前设备的 GPU 内存不足，直接加载到 GPU 可能会导致内存溢出。通过先加载到内存中，再手动移动到 GPU，可以更好地控制内存使用。
        ckpt = torch.load(self.planner_cfg['ckpt_path'], map_location=lambda storage, loc: storage)
        self.network.load_state_dict(ckpt["state_dict"])
        self.network = self.network.to(self.device)
        self.network.eval()

    def init_scen_tree_gen(self):
        scen_tree_cfg = import_module(self.planner_cfg['planning_config']).ScenTreeCfg()
        self.scen_tree_gen = ScenarioTreeGenerator(self.device, self.network, self.obs_len, self.plan_len, scen_tree_cfg)

    def init_traj_tree_opt(self):
        traj_tree_cfg = import_module(self.planner_cfg['planning_config']).TrajTreeCfg()
        self.traj_tree_opt = TrajectoryTreeOptimizer(traj_tree_cfg)


    def to_object_state(self, agent):
        """
        将代理的状态转换为对象状态。

        此方法创建一个ObjectState实例，该实例表示代理在当前时间步长、位置和方向上的状态。
        它使用代理的当前时间步长、位置（x, y）和方向来计算并封装这些信息，以便进一步处理或通信。

        参数:
        - agent: 当前代理实例，其状态信息将被用来创建ObjectState。

        返回:
        - ObjectState实例，包含代理的状态信息。
        """
        obj_state = ObjectState(True, agent.timestep, (agent.state[0], agent.state[1]), agent.state[3],
                                (agent.state[2] * np.cos(agent.state[3]),
                                 agent.state[2] * np.sin(agent.state[3])))
        return obj_state

    def update_observation(self, lcl_smp):
        """
        更新观察数据

        该方法主要用于更新内部维护的关于各个智能体（agent）的观察状态。包括自我智能体（ego agent）和其他智能体（exo agents）的状态更新。
        如果某个智能体在当前样本中没有被观察到，将为该智能体分配一个虚拟的状态。

        参数:
        - lcl_smp: 当前的样本数据，包含自我智能体和其他智能体的信息
        """
        
        # 1. 更新自车智能体的状态
        if 'AV' not in self.agent_obs:
            # 如果自我智能体尚未在观察列表中，则初始化它
            self.agent_obs['AV'] = Track('AV', [self.to_object_state(lcl_smp.ego_agent)],
                                         lcl_smp.ego_agent.type,
                                         TrackCategory.FOCAL_TRACK)
        else:
            # 如果自车智能体已在观察列表中，则追加其新的状态
            self.agent_obs['AV'].object_states.append(self.to_object_state(lcl_smp.ego_agent))

        # 2. 更新其他智能体的状态
        updated_agent_ids = ['AV']  # 用于记录已更新的智能体ID
        for agent in lcl_smp.exo_agents:
            if agent.id not in self.agent_obs:
                # 如果其他智能体尚未在观察列表中，则初始化它
                self.agent_obs[agent.id] = Track(agent.id, [self.to_object_state(agent)], agent.type,
                                                 TrackCategory.TRACK_FRAGMENT)
            else:
                # 如果其他智能体已在观察列表中，则追加其新的状态
                self.agent_obs[agent.id].object_states.append(self.to_object_state(agent))
            updated_agent_ids.append(agent.id)

        # 3. 为未被观察到的智能体分配虚拟状态
        for agent in self.agent_obs.values():
            if agent.track_id not in updated_agent_ids:
                # 如果某个智能体在当前样本中没有被观察到，为其分配一个虚拟状态
                agent.object_states.append(ObjectState(False, agent.object_states[-1].timestep,
                                                       agent.object_states[-1].position,
                                                       agent.object_states[-1].heading,
                                                       agent.object_states[-1].velocity))

        # 保持观察序列的长度不超过预设的最大值
        for agent in self.agent_obs.values():
            if len(agent.object_states) > self.obs_len:
                # 如果某个智能体的状态序列超出了设定的长度，移除最旧的状态
                agent.object_states.pop(0)

    def update_state_ctrl(self, state, ctrl):
        self.state = state
        self.ctrl = ctrl

    def update_target_lane(self, gt_tgt_lane):  # 目标车道线（gt_tgt_lane）
        self.gt_tgt_lane = gt_tgt_lane

    def plan(self, lcl_smp):
        """
        规划函数，用于计算最优轨迹和控制指令。

        :param lcl_smp: 当前样本，包含环境和车辆信息
        :return: 布尔值表示规划是否成功，最优控制指令，以及规划结果的调试信息
        """
        t0 = time.time()
        # 重置场景树生成器
        self.scen_tree_gen.reset()
        # 高层命令：重新采样目标车道
        resample_target_lane, resample_target_lane_info = self.resample_target_lane(lcl_smp)

        # 设置目标车道
        self.scen_tree_gen.set_target_lane(resample_target_lane, resample_target_lane_info)

        # 生成场景树
        scen_trees = self.scen_tree_gen.branch_aime(lcl_smp, self.agent_obs)

        # 如果生成的场景树数量小于0，返回失败
        if len(scen_trees) < 0:
            return False, None, None

        print("scenario expand done in {} secs".format(time.time() - t0))
        traj_trees = []
        debug_info = []
        # 对每个场景树生成对应的轨迹树
        for scen_tree in scen_trees:
            traj_tree, debug = self.get_traj_tree(scen_tree, lcl_smp)
            # traj_trees.append(self.get_traj_tree(scen_tree, lcl_smp))
            traj_trees.append(traj_tree)
            debug_info.append(debug)


        # use multi-threading to speed up
        # n_proc = len(scen_trees)
        # traj_trees = Parallel(n_jobs=n_proc)(
        #     delayed(self.get_traj_tree)(scen_tree, lcl_smp) for scen_tree in scen_trees)

        print("mind planning done in {} secs".format(time.time() - t0))

        # import matplotlib.pyplot as plt
        # for idx, (traj_tree, scen_tree) in enumerate(zip(traj_trees, scen_trees)):
        #     _, ax = plt.subplots(figsize=(12, 12))
        #     ax.axis('equal')
        #     debug_info[idx].plot(ax)
        #     ax.plot(lcl_smp.ego_agent.state[0], lcl_smp.ego_agent.state[1], 'b*')
        #
        #     l_node = scen_tree.get_leaf_nodes()[0]
        #     tgt_pts = l_node.data[-1]
        #
        #     # ax.plot(lcl_smp.target_lane[:, 0], lcl_smp.target_lane[:, 1], 'r-', marker='o')
        #     ax.plot(tgt_pts[:, 0], tgt_pts[:, 1], 'r-', marker='o')
        #     for node in traj_tree.nodes.values():
        #         node_ctr = node.data[0][:2]
        #         ax.plot(node_ctr[0], node_ctr[1], marker='o', alpha=0.5, color='g')
        #         ax.text(node_ctr[0], node_ctr[1], str(node.key) + "_v:{:.2}".format(node.data[0][2]) +
        #                 "_q:{:.2}".format(node.data[0][3]) + "_a:{:.2}".format(node.data[0][4]) +
        #                 "_theta:{:.2}".format(node.data[0][4]) +
        #                 "_da:{:.2}".format(node.data[1][0]) + "_dtheta:{:.2}".format(node.data[1][1]), fontsize=8)
        #         if node.parent_key is not None:
        #             parent_node = traj_tree.get_node(node.parent_key)
        #             pts = np.array([parent_node.data[0][:2], node.data[0][:2]])
        #             ax.plot(pts[:, 0], pts[:, 1], color='g', alpha=0.25)
        #     plt.show()

        # 选择最优轨迹
        best_traj_idx = None
        min_cost = np.inf
        for idx, traj_tree in enumerate(traj_trees):
            cost = self.evaluate_traj_tree(lcl_smp, traj_tree)
            if cost < min_cost:
                min_cost = cost
                best_traj_idx = idx

        # 获取最优轨迹树和下一个节点的控制指令
        opt_traj_tree = traj_trees[best_traj_idx]
        next_node = opt_traj_tree.get_node(opt_traj_tree.get_root().children_keys[0])
        ret_ctrl = next_node.data[0][-2:]

        # 返回规划结果
        return True, ret_ctrl, [[scen_trees[best_traj_idx]], [traj_trees[best_traj_idx]]]

    def resample_target_lane(self, lcl_smp):
        """
        以 1.0 米的间隔重新采样目标车道及其相关信息。

        参数:
        lcl_smp: 包含要重新采样的目标车道和目标车道信息。

        返回:
        resample_target_lane: 一个包含重新采样车道点的 NumPy 数组。
        resample_target_lane_info: 一个包含重新采样车道信息的 NumPy 数组列表。
        """
        # 初始化用于存储重新采样车道点和信息的列表
        resample_target_lane = []
        resample_target_lane_info = [[] for _ in range(6)]

        # 遍历目标车道中的每一段
        for i in range(len(lcl_smp.target_lane) - 1):
            # 获取目标车道段
            lane_segment = lcl_smp.target_lane[i:i + 2]
            # 计算目标车道段的长度
            lane_segment_len = np.linalg.norm(lane_segment[0] - lane_segment[1])
            # 计算目标车道段所需的采样点数
            num_sample = int(np.ceil(lane_segment_len / 1.0))
            # 重新采样目标车道段
            for j in range(num_sample):
                alpha = j / num_sample
                resample_target_lane.append(lane_segment[0] + alpha * (lane_segment[1] - lane_segment[0]))  # 啥意思？一阶滤波？
                # 重新采样对应的车道信息
                for k, info in enumerate(lcl_smp.target_lane_info):
                    resample_target_lane_info[k].append(info[i])

        # 确保最后一个点及其信息被添加
        resample_target_lane.append(lcl_smp.target_lane[-1])
        for k, info in enumerate(lcl_smp.target_lane_info):
            resample_target_lane_info[k].append(info[-1])

        # 转换为 NumPy 数组以提高效率
        resample_target_lane = np.array(resample_target_lane)
        for i in range(len(resample_target_lane_info)):
            resample_target_lane_info[i] = np.array(resample_target_lane_info[i])

        return resample_target_lane, resample_target_lane_info


    def get_traj_tree(self, scen_tree, lcl_smp):
        # 初始化带有热启动的成本树，并设置目标速度。这个warm start如何理解？
        self.traj_tree_opt.init_warm_start_cost_tree(scen_tree, self.state, self.ctrl, self.gt_tgt_lane, lcl_smp.target_velocity)

        # 使用热启动方法求解轨迹和控制输入
        xs, us = self.traj_tree_opt.warm_start_solve()

        # 重新初始化成本树，不带热启动状态。
        self.traj_tree_opt.init_cost_tree(scen_tree, self.state, self.ctrl, self.gt_tgt_lane, lcl_smp.target_velocity)

        # 返回最终的轨迹和调试信息。
        # return self.traj_tree_opt.solve(us)
        return self.traj_tree_opt.solve(us), self.traj_tree_opt.debug

    def evaluate_traj_tree(self, lcl_smp, traj_tree):
        # we use cost function here, instead of the reward function in the paper, but reward functions can work as well
        # simplified cost function

        # 定义成本函数中的权重和初始化成本
        comfort_acc_weight = .1
        comfort_str_weight = 5.
        comfort_cost = 0.0
        efficiency_weight = .01
        efficiency_cost = 0.0
        target_weight = .01
        target_cost = 0.0

        # 遍历轨迹树中的所有节点
        n_nodes = len(traj_tree.nodes)
        for node in traj_tree.nodes.values():
            # 提取节点中的状态和控制数据
            state = node.data[0]
            ctrl = node.data[1]
            
            # 计算舒适性成本，包括加速度和转向的舒适性
            comfort_cost += comfort_acc_weight * ctrl[0] ** 2
            comfort_cost += comfort_str_weight * ctrl[1] ** 2

            # 计算效率成本，即与目标速度的偏差
            efficiency_cost += efficiency_weight * (lcl_smp.target_velocity - state[2]) ** 2

            # 计算目标成本，即到目标车道的距离
            target_cost += target_weight * self.get_dist_to_target_lane(lcl_smp, state)
        print(
            "n_nodes: {}, comfort cost: {}, efficiency cost: {}, target cost: {}".format(n_nodes, comfort_cost,
                                                                                         efficiency_cost, target_cost))
        
        # 返回平均成本，作为轨迹树优劣的评估
        return (comfort_cost + efficiency_cost + target_cost) / n_nodes

    def get_dist_to_target_lane(self, lcl_smp, state):
        #  project the state to the target lane
        proj_state, _, _ = project_point_on_polyline(state[:2], lcl_smp.target_lane)
        #  get the distance
        dist = np.linalg.norm(proj_state - state[:2])
        return dist

    def get_interpolated_state(self, tree, timestep):
        """
        根据给定的时间步，从树结构中获取插值状态。

        首先检查时间步是否小于根节点的时间，如果是，则返回根节点的状态和控制信号。
        否则，通过遍历树中的节点，找到包含目标时间步的节点，进行线性插值以估计该时间步的状态。
        插值是基于找到的包含目标时间步的节点及其前一个节点进行的。

        参数:
        - tree: 一棵存储按时间顺序排列的节点的树，每个节点包含状态和时间信息。
        - timestep: 我们想要获取状态信息的特定时间步。

        返回:
        - interp_state: 在给定时间步的插值状态。
        - ctrl: 当前节点的控制信号。
        """
        # 获取树的根节点
        root_node = tree.get_node(0)
        # 如果时间步小于根节点的时间，直接返回根节点的状态和控制信号
        if timestep < root_node.data.t:
            return root_node.data.state, root_node.data.ctrl
        else:
            # 初始化当前节点为根节点
            node = root_node
            # 遍历树，直到找到包含目标时间步的节点
            while node.data.t <= timestep:
                node = tree.get_node(node.children_keys[0])
            # 在找到的节点和其父节点之间进行线性插值
            # 获取父节点
            prev_node = tree.get_node(node.parent_key)
            # 父节点和当前节点的状态
            prev_state = prev_node.data.state
            next_state = node.data.state
            # 父节点和当前节点的时间
            prev_time = prev_node.data.t
            next_time = node.data.t
            # 计算插值系数
            alpha = (timestep - prev_time) / (next_time - prev_time)
            # 根据插值系数计算插值状态
            interp_state = prev_state + alpha * (next_state - prev_state)
            # 返回插值状态和当前节点的控制信号
            return interp_state, node.data.ctrl
