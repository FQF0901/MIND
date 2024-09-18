import copy
import torch
import numpy as np
from planners.basic.tree import Tree, Node
from planners.mind.utils import gpu, from_numpy, get_max_covariance, get_origin_rotation, get_new_lane_graph, \
    get_rpe, get_angle, collate_fn, get_agent_trajectories, update_lane_graph_from_argo, \
    get_distance_to_polyline


class ScenarioData:
    def __init__(self, data, obs_data, branch_flag=False, end_flag=False, terminate_flag=False):
        self.data = data
        self.obs_data = obs_data
        self.branch_flag = branch_flag
        self.end_flag = end_flag
        self.terminate_flag = terminate_flag


class ScenarioTreeGenerator:
    def __init__(self, device, network, obs_len=50, pred_len=60, config=None):
        self.device = device
        self.network = network
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = self.obs_len + self.pred_len
        self.config = config
        self.tree = Tree()
        self.lane_graph = None
        self.target_lane = None
        self.target_lane_info = None
        self.ego_idx = 0
        self.branch_depth = 0

    def reset(self):
        self.branch_depth = 0
        self.tree = Tree()

    def branch_aime(self, lcl_smp, agent_obs):
        """
        执行AIME算法，通过迭代预测、剪枝、合并和分支构建场景树。

        参数:
        - lcl_smp: 本地样本信息，用于数据处理。
        - agent_obs: 环境中代理的观察信息，作为场景预测的输入。

        返回:
        - 构建好的场景树，旨在通过预测和分析场景来提供决策依据。
        """

        # Initialization phase:处理输入数据并初始化场景树
        data = self.process_data(lcl_smp, agent_obs)
        self.init_scenario_tree(data)

        # AIME iteration:持续执行直到没有节点可分支
        branch_nodes = self.get_branch_set()
        while branch_nodes:
            # Batch Scenario Prediction: 收集当前分支节点的观察数据进行批量处理和预测
            data_batch = collate_fn([node.data.obs_data for node in branch_nodes])  # data_batch包含了agent、lane、和target信息
            pred_batch = self.predict_scenes(data_batch)    # 通过网络模型进行场景预测，这里得到的是预测每个agent的res_cls, res_reg, res_aux

            # Pruning & Merging: 根据预测结果，剪枝不可能的场景并合并相似的场景
            pred_bar = self.prune_merge(data_batch, pred_batch) # 两个输入：当前信息data_batch 和 预测结果pred_batch

            # Create New Nodes: 根据剪枝和合并后的预测结果，在场景树中创建新节点
            self.create_nodes(pred_bar)

            # Branching Decision: 对新添加的节点进行分支决策，扩展场景树
            self.decide_branch()

            # Update Branch Set: 更新待分支的节点集，准备下一轮迭代
            branch_nodes = self.get_branch_set()

        # 确保场景树至少有一个结束节点，表示场景发展完成
        assert len(self.get_end_set()) > 0, "No end node found in the scenario tree."
        return self.get_scenario_tree()

    def init_scenario_tree(self, data):
        # prepossess the observation data and map data
        root_data = self.prepare_root_data(data)
        self.tree.add_node(Node('root', None, ScenarioData(None, root_data, branch_flag=True)))
        pred_batch = self.predict_scenes(root_data)
        pred_bar = self.prune_merge(root_data, pred_batch)
        self.create_nodes(pred_bar)
        self.decide_branch()

    def predict_scenes(self, data):
        data_in = self.network.pre_process(data)    # 对输入数据进行预处理
        return self.network(data_in)    # 通过网络模型进行场景预测

    def create_nodes(self, pred_bar):
        for pred in pred_bar:
            parent_id = pred["PARENT_ID"]
            node_id = pred["SCEN_ID"]
            # Create new node
            new_node = Node(node_id, parent_id, ScenarioData(pred, None))
            # Attach to the tree
            self.tree.add_node(new_node)

    def decide_branch(self):
        # iterate over the leaf nodes
        for l in self.tree.get_leaf_nodes():    # 获取树中的所有叶节点，意味着要检查每个节点的状态以决定其在树中的角色

            # 1. 更新分支和终止标志。如果叶节点已经是一个分支节点，则将其标记为终止。这反映了对树结构动态管理的需求，避免重复分支
            if l.data.branch_flag:
                l.data.branch_flag = False
                l.data.terminate_flag = True

            # 2. 深度检查。限制树的高度以控制复杂度。达到最大深度时，节点标记为终止，确保树的形状不会无限扩展
            elif not l.data.end_flag:
                if l.depth >= self.config.max_depth:
                    l.data.terminate_flag = True
                else:
                    # 3. 计算该节点的分支时间，评估是否应该继续向下分支，反映了时间对决策的影响
                    t_b = self.get_branch_time(l.data.data)
                    if t_b < self.pred_len:
                        # Update the observation data. 更新节点的数据，可能反映出新的观测结果，从而影响后续的决策过程
                        l.data.obs_data, l.data.data = self.update_obser(l.data.data)
                        # Add node to branch set
                        l.data.branch_flag = True   # 如果节点的分支时间小于预测长度，则将其标记为分支，反之则标记为结束。这反映出时间和状态变化对决策的影响
                    else:
                        # Add node the end set
                        l.data.end_flag = True

    def get_branch_set(self):
        branch_set = []
        for l in self.tree.get_leaf_nodes():
            if l.data.branch_flag:
                branch_set.append(l)
        self.branch_depth += 1
        return branch_set

    def set_target_lane(self, target_lane, target_lane_info):
        # 这个 gpu 函数可以递归处理不同类型的数据结构，并将其中的所有张量移动到指定的 GPU 设备上
        self.target_lane = gpu(torch.from_numpy(np.array(target_lane)), self.device)

        # unsqueeze用于在指定的位置向张量中添加一个新的维度。例如一个形状为 [N] 的一维张量，使用 unsqueeze(1) 会将其转换为 [N, 1] 的二维张量
        # torch.cat 是 PyTorch 中的一个函数，用于将多个张量沿着最后一个维度（dim=-1）连接起来
        self.target_lane_info = torch.cat([torch.from_numpy(target_lane_info[0]).unsqueeze(1),
                                           torch.from_numpy(target_lane_info[1]),
                                           torch.from_numpy(target_lane_info[2]),
                                           torch.from_numpy(target_lane_info[3]),
                                           torch.from_numpy(target_lane_info[4]).unsqueeze(1),
                                           torch.from_numpy(target_lane_info[5]).unsqueeze(1)],
                                          dim=-1)  # [N_{lane}, 16, F]

        self.target_lane_info = gpu(self.target_lane_info, self.device)

    def process_data(self, lcl_smp, agent_obs):
        """
        处理agent观察数据和局部样本数据。

        参数:
        lcl_smp : LocalSample
            包含环境数据的局部样本。
        agent_obs : AgentObservation
            agent的观察数据。

        返回:
        处理后的数据，供系统使用。
        """
        # 获取agent轨迹数据
        trajs_pos, trajs_ang, trajs_vel, trajs_type, has_flags, trajs_tid, trajs_cat = get_agent_trajectories(
            agent_obs, self.device)

        # 获取当前ego agent的速度
        cur_vel = lcl_smp.ego_agent.state[2]

        # 获取第一个轨迹的目标位置和旋转序列
        orig_seq, rot_seq, theta_seq = get_origin_rotation(trajs_pos[0], trajs_ang[0], self.device)

        # 更新车道图
        lane_graph = update_lane_graph_from_argo(lcl_smp.map_data, orig_seq.cpu().numpy(), rot_seq.cpu().numpy())
        lane_graph = gpu(from_numpy(lane_graph), self.device)

        # 对轨迹进行场景归一化
        trajs_pos = torch.matmul(trajs_pos - orig_seq, rot_seq)
        trajs_ang = trajs_ang - theta_seq
        trajs_vel = torch.matmul(trajs_vel, rot_seq)

        # 归一化轨迹数据
        trajs_pos_norm = []
        trajs_ang_norm = []
        trajs_vel_norm = []
        trajs_ctrs = []
        trajs_vecs = []
        for traj_pos, traj_ang, traj_vel in zip(trajs_pos, trajs_ang, trajs_vel):
            orig_act, rot_act, theta_act = get_origin_rotation(traj_pos, traj_ang, self.device)
            trajs_pos_norm.append(torch.matmul(traj_pos - orig_act, rot_act))
            trajs_ang_norm.append(traj_ang - theta_act)
            trajs_vel_norm.append(torch.matmul(traj_vel, rot_act))
            trajs_ctrs.append(orig_act)
            trajs_vecs.append(torch.tensor([torch.cos(theta_act), torch.sin(theta_act)]))

        # 将归一化的轨迹数据堆叠起来
        trajs_pos = torch.stack(trajs_pos_norm)  # [N, 110(50), 2]
        trajs_ang = torch.stack(trajs_ang_norm)  # [N, 110(50)]
        trajs_vel = torch.stack(trajs_vel_norm)  # [N, 110(50), 2]
        trajs_ctrs = torch.stack(trajs_ctrs).to(self.device)  # [N, 2]
        trajs_vecs = torch.stack(trajs_vecs).to(self.device)  # [N, 2]

        # 创建一个字典来存储轨迹数据
        trajs = dict()

        trajs["TRAJS_POS_OBS"] = trajs_pos
        trajs["TRAJS_ANG_OBS"] = torch.stack([torch.cos(trajs_ang), torch.sin(trajs_ang)], axis=-1)
        trajs["TRAJS_VEL_OBS"] = trajs_vel
        trajs["TRAJS_TYPE"] = trajs_type
        trajs["PAD_OBS"] = has_flags
        # anchor ctrs & vecs
        trajs["TRAJS_CTRS"] = trajs_ctrs
        trajs["TRAJS_VECS"] = trajs_vecs
        # track id & category
        trajs["TRAJS_TID"] = trajs_tid  # List[str]
        trajs["TRAJS_CAT"] = trajs_cat  # List[str]

        # 获取高层次命令
        tgt_pts, tgt_nodes, tgt_anch = self.get_high_level_command(orig_seq, rot_seq, cur_vel)

        # 计算相对位置误差
        lane_ctrs = lane_graph['lane_ctrs']
        lane_vecs = lane_graph['lane_vecs']
        # ~ calc rpe
        rpes = dict()

        scene_ctrs = torch.cat([trajs_ctrs, lane_ctrs], dim=0)
        scene_vecs = torch.cat([trajs_vecs, lane_vecs], dim=0)
        rpes['scene'], rpes['scene_mask'] = get_rpe(scene_ctrs, scene_vecs)

        # ~ calc rpe for tgt
        tgt_ctr, tgt_vec = tgt_anch
        tgt_ctrs = torch.cat([tgt_ctr.unsqueeze(0), trajs_ctrs[0].unsqueeze(0)])
        tgt_vecs = torch.cat([tgt_vec.unsqueeze(0), trajs_vecs[0].unsqueeze(0)])
        tgt_rpe, _ = get_rpe(tgt_ctrs, tgt_vecs)

        # prepare data
        data = {}
        data["ORIG"] = orig_seq
        data["ROT"] = rot_seq
        data["TRAJS"] = trajs
        data["LANE_GRAPH"] = lane_graph
        data["TGT_PTS"] = tgt_pts
        data["TGT_NODES"] = tgt_nodes
        data["TGT_ANCH"] = tgt_anch
        data['RPE'] = rpes
        data['TGT_NODES'] = tgt_nodes
        data['TGT_ANCH'] = tgt_anch
        data['TGT_RPE'] = tgt_rpe

        self.lane_graph = copy.deepcopy(data["LANE_GRAPH"])
        return gpu(collate_fn([data]), self.device)

    def get_scenario_tree(self):
        data_tree = Tree()
        root_node = self.tree.get_root()
        data_tree.add_node(Node(root_node.key, None, [1.0]))

        # label the branch that actually finished
        for node in self.get_end_set():
            while node.parent_key is not None:
                node.data.end_flag = True
                node = self.tree.get_node(node.parent_key)

        # construct the data_tree recursively and add normalized probability
        for key in root_node.children_keys:
            node = self.tree.get_node(key)
            if not node.data.end_flag:
                continue
            data_tree.add_node(Node(node.key, root_node.key, [1.0]))
            queue = [node]
            while queue:
                cur_node = queue.pop(0)
                parent_prob = data_tree.get_node(cur_node.key).data[0]
                total_prob = 0.0
                for child_key in cur_node.children_keys:
                    child_node = self.tree.get_node(child_key)
                    if child_node.data.end_flag:
                        total_prob += child_node.data.data["SCEN_PROB"].cpu().numpy()

                for child_key in cur_node.children_keys:
                    child_node = self.tree.get_node(child_key)
                    if child_node.data.end_flag:
                        data_tree.add_node(Node(child_node.key, cur_node.key,
                                                [child_node.data.data[
                                                     "SCEN_PROB"].cpu().numpy() / total_prob * parent_prob]))
                        queue.append(child_node)

        # add traj, cov, tgt_lane to the data_tree
        for node in self.get_end_set():
            while node.parent_key is not None:
                duration = node.data.data["END_T"] - node.data.data["CUR_T"]
                data_node = data_tree.get_node(node.key)
                if len(data_node.data) == 1:
                    data_node.data += [
                        node.data.data["TRAJS_POS_HIST"][:, self.obs_len: self.obs_len + duration, :].cpu().numpy(),
                        node.data.data["TRAJS_COV_HIST"][:, self.obs_len: self.obs_len + duration, :].cpu().numpy(),
                        node.data.data["TGT_PTS"].cpu().numpy()]
                node = self.tree.get_node(node.parent_key)

        #  separate the data_tree into trajectory trees from the root
        scenario_trees = []

        for key in data_tree.get_root().children_keys:
            scenario_tree = Tree()
            node = data_tree.get_node(key)
            scenario_tree.add_node(Node(node.key, None, node.data))
            #  add the children nodes recursively and add normalized probability
            queue = [node]
            while queue:
                cur_node = queue.pop(0)
                for child_key in cur_node.children_keys:
                    child_node = data_tree.get_node(child_key)
                    scenario_tree.add_node(Node(child_node.key, cur_node.key, child_node.data))
                    queue.append(child_node)
            scenario_trees.append(scenario_tree)

        return scenario_trees

    def get_end_set(self):
        end_nodes = []
        for node in self.tree.get_leaf_nodes():
            if node.data.end_flag:
                end_nodes.append(node)
        return end_nodes

    def prune_merge(self, data, out):
        data_interact = []
        batch_size = len(data['ORIG'])
        res_cls_batch, res_reg_batch, res_aux_batch = out

        for idx in range(batch_size):
            orig = data['ORIG'][idx]
            rot = data['ROT'][idx]
            trajs_ctrs = data['TRAJS'][idx]['TRAJS_CTRS']   # Trajectories Centers
            trajs_vecs = data['TRAJS'][idx]['TRAJS_VECS']   # Trajectories Vectors
            trajs_type = data['TRAJS'][idx]["TRAJS_TYPE"]   # Trajectories Types
            trajs_tid = data['TRAJS'][idx]["TRAJS_TID"] # Trajectories ID
            trajs_cat = data['TRAJS'][idx]["TRAJS_CAT"] # Trajectories Category

            # items in global frame
            theta_global = torch.atan2(rot[1, 0], rot[0, 0])

            trajs_pos_hist = data['TRAJS_POS_HIST'][idx]    # Trajectories Position History
            trajs_ang_hist = data['TRAJS_ANG_HIST'][idx]    # Trajectories Angle History
            trajs_vel_hist = data['TRAJS_VEL_HIST'][idx]    # Trajectories Velocity History
            trajs_cov_hist = data['TRAJS_COV_HIST'][idx]    # Trajectories Coverage History

            parent_id = data['SCEN_ID'][idx]
            parent_prob = data['SCEN_PROB'][idx]    # Scenario Probability
            cur_t = data['CUR_T'][idx]  # Current Time
            end_t = data['END_T'][idx]  # End Time

            # 提取并分离当前索引的回归、分类和速度结果（这3个都是预测网给出的）
            res_reg = res_reg_batch[idx].detach()   # 表示对于每个预测模式的具体轨迹参数。这些参数描述了预测轨迹的具体形状或位置信息，例如位置坐标、速度等
            res_cls = res_cls_batch[idx].detach()   # 表示对于每个模式（mode）的分类概率。具体来说，它包含了模型对不同未来轨迹模式的概率分布估计
            res_vel = res_aux_batch[idx][0].detach()    # 提供了额外的信息来辅助理解或评估回归结果。具体来说，它包含了预测轨迹的速度（vel）、协方差（cov_vel）等
            # 计算速度向量的角度
            res_ang = get_angle(res_vel)

            # sort the scene by the probability. 根据概率对场景进行排序
            scene_idcs = torch.argsort(res_cls, dim=1, descending=True)[0]

            # 初始化数据候选列表
            data_candidates = []

            # 遍历排序后的场景索引
            for scene_id in scene_idcs:
                # 提取当前场景的概率
                scene_prob = res_cls[0, scene_id]

                # 构造场景ID
                scen_id = "{}_{}_{}".format(self.branch_depth, idx, scene_id)

                # 提取当前位置、协方差和速度的预测结果
                trajs_pos_pred = res_reg[:, scene_id, :, :2]
                # trajs_cov_pred = get_covariance_matrix(res_reg[:, scene_id, :, 2:])
                trajs_cov_pred = get_max_covariance(res_reg[:, scene_id, :, 2:])  # use the max sigma
                trajs_vel_pred = res_vel[:, scene_id]

                # 计算轨迹的角度和旋转矩阵
                trajs_theta = torch.atan2(trajs_vecs[:, 1], trajs_vecs[:, 0])
                trajs_rots = torch.stack([torch.cos(trajs_theta), -torch.sin(trajs_theta),
                                          torch.sin(trajs_theta), torch.cos(trajs_theta)], dim=1).view(-1, 2, 2)

                # 对每个轨迹进行坐标系对齐和位置偏移
                for i in range(len(trajs_pos_pred)):
                    trajs_pos_pred[i] = torch.matmul(trajs_pos_pred[i], trajs_rots[i].transpose(-1, -2)) + trajs_ctrs[i]
                    trajs_vel_pred[i] = torch.matmul(trajs_vel_pred[i], trajs_rots[i].transpose(-1, -2))
                    # trajs_cov_pred[i] = torch.matmul(trajs_rots[i],
                    #                                  torch.matmul(trajs_cov_pred[i], trajs_rots[i].transpose(-1, -2)))

                # 将预测结果转换到全局坐标系
                trajs_pos_pred = torch.matmul(trajs_pos_pred, rot.T) + orig
                trajs_vel_pred = torch.matmul(trajs_vel_pred, rot.T)
                # trajs_cov_pred = torch.matmul(rot, torch.matmul(trajs_cov_pred, rot.T))

                # 更新预测的角度
                trajs_ang_pred = res_ang[:, scene_id] + trajs_theta.unsqueeze(1) + theta_global

                trajs_cov_pred += trajs_cov_hist[:, -1].unsqueeze(1)

                # 将预测结果与历史数据合并
                trajs_pos_hist_new = torch.cat([trajs_pos_hist, trajs_pos_pred], dim=1)[:, :self.seq_len]
                trajs_cov_hist_new = torch.cat([trajs_cov_hist, trajs_cov_pred], dim=1)[:, :self.seq_len]
                trajs_ang_hist_new = torch.cat([trajs_ang_hist, trajs_ang_pred], dim=1)[:, :self.seq_len]
                trajs_vel_hist_new = torch.cat([trajs_vel_hist, trajs_vel_pred], dim=1)[:, :self.seq_len]

                # 构造当前轨迹数据字典
                cur_traj_data = dict()
                cur_traj_data["TRAJS_TYPE"] = trajs_type
                cur_traj_data["TRAJS_TID"] = trajs_tid
                cur_traj_data["TRAJS_CAT"] = trajs_cat

                # 构造当前场景数据字典
                cur_data = {}
                cur_data["SCEN_PROB"] = scene_prob * parent_prob
                cur_data["CUR_T"] = cur_t
                cur_data["END_T"] = end_t
                cur_data["PARENT_ID"] = parent_id
                cur_data["SCEN_ID"] = scen_id
                cur_data["TRAJS"] = cur_traj_data
                cur_data['TRAJS_POS_HIST'] = trajs_pos_hist_new
                cur_data['TRAJS_COV_HIST'] = trajs_cov_hist_new
                cur_data['TRAJS_ANG_HIST'] = trajs_ang_hist_new
                cur_data['TRAJS_VEL_HIST'] = trajs_vel_hist_new
                cur_data['TGT_PTS'] = data['TGT_PTS'][idx]

                # 1. prune if the scene is not likely. 如果场景概率过低，则忽略
                if cur_data["SCEN_PROB"] < 0.001:
                    continue

                # 2. prune if the ego decision is not likely to follow the target lane. 
                # 如果自车决策没有遵循目标路线，则进行修剪。这段代码帮助确保自车在控制和导航时保持在目标车道附近，但可能有点不合适，因为貌似难以处理倒车避让场景
                if self.target_lane is not None and self.ego_idx is not None:   # 确保存在目标车道和自车的索引
                    # 提取自车当前的平均位置（ego_mean）和不确定性（ego_cov），反映自车状态的精确度和稳定性
                    ego_mean = cur_data['TRAJS_POS_HIST'][self.ego_idx][-1] 
                    ego_cov = cur_data['TRAJS_COV_HIST'][self.ego_idx][-1]

                    dis = get_distance_to_polyline(self.target_lane, ego_mean)  # 计算自车位置到目标车道的最短距离，帮助判断自车与车道的关系
                    if dis - ego_cov > self.config.tar_dist_thres:  # 判断自车到车道的距离是否超过了一个阈值
                        continue

                # 3.1 al the topo cum change for merging. 计算拓扑累积变化用于合并
                topos = torch.zeros(len(trajs_pos_pred) - 1)    # 用于存储每个轨迹段之间的累计角度变化，帮助后续分析车辆的转向行为
                for iii, traj in enumerate(trajs_pos_pred[1:]): # 遍历每个预测的轨迹，从第二个轨迹开始，目的是分析每段轨迹与起始轨迹之间的关系
                    # cal the cum angle change of the vector pointing from ego to the exo. 计算从 ego 到 exogenous 的向量的角度累积变化
                    vec = traj - trajs_pos_pred[0]
                    vec = vec / torch.norm(vec, dim=-1, keepdim=True)   # 将向量归一化，使其长度为1，便于只关注方向，而不受距离影响
                    ang = torch.atan2(vec[:, 1], vec[:, 0]) # 通过计算向量的角度，获取其与x轴的夹角，表示自车相对于目标的方向
                    ang_diff = ang[1:] - ang[:-1]   # 计算相邻向量之间的角度差，反映车辆在行驶过程中转向的变化
                    # normalize the angle diff
                    ang_diff = torch.atan2(torch.sin(ang_diff), torch.cos(ang_diff))    # 归一化角度差，确保角度在 -π 到 π 之间，避免跨越180度时出现错误
                    # cal the cum angle change of the vector pointing from ego to the exo
                    topos[iii] = torch.sum(ang_diff)    # 将每段轨迹的角度变化累加，表示自车在行驶过程中总的转向变化，帮助分析其运动模式和稳定性

                # 将当前数据候选添加到列表中
                data_candidates.append([cur_data, scene_prob, topos])

            # 3.2 merge the similar scenes. 合并相似场景
            selected_data = []
            min_topo_change = torch.pi / 6  # delta 设置一个阈值（30度），用于限制所选择数据之间的角度变化，确保选出的数据在方向上较为一致

            while len(data_candidates) > 0: # 在候选数据仍有剩余时，不断进行筛选，以获得一组满足条件的数据
                select_data, select_prob, select_topos = data_candidates[0]
                selected_data.append(select_data)   # 从候选数据中选取第一个数据并将其添加到已选择的数据集中
                data_candidates_tmp = []
                for data_candidate in data_candidates[1:]:
                    _, _, res_topos = data_candidate

                    # 1. 计算已选择数据与其他候选数据之间的拓扑差异，归一化角度变化，确保其在有效范围内，反映出两者的相对方向变化
                    topos_diff = select_topos - res_topos
                    topos_diff = torch.atan2(torch.sin(topos_diff), torch.cos(topos_diff))

                    # 2. 只保留那些与已选择数据的拓扑变化超过阈值的数据，从而确保选出的数据集在方向上保持一致
                    if torch.sum((torch.abs(topos_diff) - min_topo_change) > 0) > 0:
                        data_candidates_tmp.append(data_candidate)

                data_candidates = data_candidates_tmp
            data_interact += selected_data

        return data_interact

    def prepare_root_data(self, data):
        batch_size = len(data['ORIG'])
        data['TRAJS_POS_HIST'] = [[] for _ in range(batch_size)]
        data['TRAJS_ANG_HIST'] = [[] for _ in range(batch_size)]
        data['TRAJS_VEL_HIST'] = [[] for _ in range(batch_size)]
        data['TRAJS_COV_HIST'] = [[] for _ in range(batch_size)]
        data['SCEN_PROB'] = [1.0 for _ in range(batch_size)]
        data['SCEN_ID'] = ["root" for _ in range(batch_size)]
        data['PARENT_ID'] = [None for _ in range(batch_size)]
        data['CUR_T'] = [0 for _ in range(batch_size)]
        data['END_T'] = [self.pred_len for _ in range(batch_size)]
        for idx in range(batch_size):
            orig = data['ORIG'][idx]
            rot = data['ROT'][idx]
            trajs_ctrs = data['TRAJS'][idx]['TRAJS_CTRS']
            trajs_vecs = data['TRAJS'][idx]['TRAJS_VECS']
            theta_global = torch.atan2(rot[1, 0], rot[0, 0])

            trajs_pos_obs = data['TRAJS'][idx]['TRAJS_POS_OBS']
            trajs_vel_obs = data['TRAJS'][idx]['TRAJS_VEL_OBS']
            trajs_ang_obs = get_angle(data['TRAJS'][idx]['TRAJS_ANG_OBS'])

            trajs_theta = torch.atan2(trajs_vecs[:, 1], trajs_vecs[:, 0])
            trajs_rots = torch.stack([torch.cos(trajs_theta), -torch.sin(trajs_theta),
                                      torch.sin(trajs_theta), torch.cos(trajs_theta)], dim=1).view(-1, 2, 2).to(
                self.device)

            trajs_pos_hist = torch.empty_like(trajs_pos_obs)
            trajs_vel_hist = torch.empty_like(trajs_vel_obs)
            # trajs_cov_hist = 1e-5 * torch.eye(2).unsqueeze(0).unsqueeze(0).repeat(len(trajs_pos_obs),
            #                                                                       len(trajs_pos_obs[0]), 1, 1).to(
            #     self.device)
            # print("trajs_cov_hist: ", trajs_cov_hist.shape)
            trajs_cov_hist = 1e-5 * torch.ones((1,)).repeat(len(trajs_pos_obs), len(trajs_pos_obs[0]), 1).to(
                self.device)
            for i in range(len(trajs_pos_obs)):
                trajs_pos_hist[i] = torch.matmul(trajs_pos_obs[i], trajs_rots[i].transpose(-1, -2)) + trajs_ctrs[i]
                trajs_vel_hist[i] = torch.matmul(trajs_vel_obs[i], trajs_rots[i].transpose(-1, -2))
                # trajs_cov_hist[i] = torch.matmul(trajs_rots[i],
                #                                  torch.matmul(trajs_cov_hist[i], trajs_rots[i].transpose(-1, -2)))

            trajs_pos_hist = torch.matmul(trajs_pos_hist, rot.T) + orig
            trajs_vel_hist = torch.matmul(trajs_vel_hist, rot.T)
            # trajs_cov_hist = torch.matmul(rot, torch.matmul(trajs_cov_hist, rot.T))
            trajs_ang_hist = trajs_ang_obs + trajs_theta.unsqueeze(1) + theta_global

            # items in global frame
            data['TRAJS_POS_HIST'][idx] = trajs_pos_hist  # [N, 50, 2]
            data['TRAJS_ANG_HIST'][idx] = trajs_ang_hist  # [N, 50, 2]
            data['TRAJS_VEL_HIST'][idx] = trajs_vel_hist  # [N, 50, 2]
            data['TRAJS_COV_HIST'][idx] = trajs_cov_hist  # [N, 50, 1]
        return data

    def update_obser(self, cur_data):
        end_t = cur_data["END_T"]
        cur_t = cur_data["CUR_T"]
        duration = end_t - cur_t
        cur_data['TRAJS_POS_HIST'] = cur_data['TRAJS_POS_HIST'][:, :self.obs_len + duration]
        cur_data['TRAJS_COV_HIST'] = cur_data['TRAJS_COV_HIST'][:, :self.obs_len + duration]
        cur_data['TRAJS_ANG_HIST'] = cur_data['TRAJS_ANG_HIST'][:, :self.obs_len + duration]
        cur_data['TRAJS_VEL_HIST'] = cur_data['TRAJS_VEL_HIST'][:, :self.obs_len + duration]

        data = copy.deepcopy(cur_data)
        data['CUR_T'] = end_t
        data['END_T'] = self.pred_len
        data['TRAJS_POS_HIST'] = data['TRAJS_POS_HIST'][:, -self.obs_len:]
        data['TRAJS_COV_HIST'] = data['TRAJS_COV_HIST'][:, -self.obs_len:]
        data['TRAJS_ANG_HIST'] = data['TRAJS_ANG_HIST'][:, -self.obs_len:]
        data['TRAJS_VEL_HIST'] = data['TRAJS_VEL_HIST'][:, -self.obs_len:]

        trajs_pos = data['TRAJS_POS_HIST']
        trajs_cov = data['TRAJS_COV_HIST']
        trajs_ang = data['TRAJS_ANG_HIST']
        trajs_vel = data['TRAJS_VEL_HIST']

        has_flags = torch.ones_like(trajs_ang)

        trajs_type = data['TRAJS']["TRAJS_TYPE"]
        trajs_tid = data['TRAJS']["TRAJS_TID"]
        trajs_cat = data['TRAJS']["TRAJS_CAT"]

        # ~ get origin and rot
        orig_seq, rot_seq, theta_seq = get_origin_rotation(trajs_pos[0], trajs_ang[0], self.device)  # * target-centric

        # ~ normalize w.r.t. scene
        trajs_pos = torch.matmul(trajs_pos - orig_seq, rot_seq)
        trajs_ang = trajs_ang - theta_seq
        trajs_vel = torch.matmul(trajs_vel, rot_seq)

        # ~ normalize trajs
        trajs_pos_norm = []
        trajs_ang_norm = []
        trajs_vel_norm = []
        trajs_ctrs = []
        trajs_vecs = []
        for traj_pos, traj_ang, traj_vel in zip(trajs_pos, trajs_ang, trajs_vel):
            orig_act, rot_act, theta_act = get_origin_rotation(traj_pos, traj_ang, self.device)
            trajs_pos_norm.append(torch.matmul(traj_pos - orig_act, rot_act))
            trajs_ang_norm.append(traj_ang - theta_act)
            trajs_vel_norm.append(torch.matmul(traj_vel, rot_act))
            trajs_ctrs.append(orig_act)
            trajs_vecs.append(torch.tensor([torch.cos(theta_act), torch.sin(theta_act)]))

        trajs_pos_obs = torch.stack(trajs_pos_norm)  # [N, 110(50), 2]
        trajs_ang_obs = torch.stack(trajs_ang_norm)  # [N, 110(50)]
        trajs_vel_obs = torch.stack(trajs_vel_norm)  # [N, 110(50), 2]
        trajs_ctrs = torch.stack(trajs_ctrs).to(self.device)  # [N, 2]
        trajs_vecs = torch.stack(trajs_vecs).to(self.device)  # [N, 2]

        trajs = dict()
        # observation
        trajs["TRAJS_POS_OBS"] = trajs_pos_obs
        trajs["TRAJS_ANG_OBS"] = torch.stack([torch.cos(trajs_ang_obs), torch.sin(trajs_ang_obs)], dim=-1)
        trajs["TRAJS_VEL_OBS"] = trajs_vel_obs
        trajs["TRAJS_TYPE"] = trajs_type
        trajs["PAD_OBS"] = has_flags[:, :self.obs_len]

        # anchor ctrs & vecs
        trajs["TRAJS_CTRS"] = trajs_ctrs
        trajs["TRAJS_VECS"] = trajs_vecs
        # track id & category
        trajs["TRAJS_TID"] = trajs_tid  # List[str]
        trajs["TRAJS_CAT"] = trajs_cat  # List[str]

        # ~ get lane graph
        lane_graph = get_new_lane_graph(self.lane_graph, orig_seq, rot_seq, self.device)

        # ~ calc rpe
        rpes = dict()
        lane_ctrs = lane_graph['lane_ctrs']
        lane_vecs = lane_graph['lane_vecs']
        scene_ctrs = torch.cat([trajs_ctrs, lane_ctrs], dim=0)
        scene_vecs = torch.cat([trajs_vecs, lane_vecs], dim=0)
        rpes['scene'], rpes['scene_mask'] = get_rpe(scene_ctrs, scene_vecs)

        # ~ get target lane
        tgt_pts, tgt_nodes, tgt_anch = self.get_high_level_command(orig_seq, rot_seq, trajs_vel_obs[0, -1].norm())
        # ~ calc rpe for tgt
        tgt_ctr, tgt_vec = tgt_anch
        tgt_ctrs = torch.cat([tgt_ctr.unsqueeze(0), trajs_ctrs[0].unsqueeze(0)])
        tgt_vecs = torch.cat([tgt_vec.unsqueeze(0), trajs_vecs[0].unsqueeze(0)])
        tgt_rpe, _ = get_rpe(tgt_ctrs, tgt_vecs)

        data["ORIG"] = orig_seq
        data["ROT"] = rot_seq
        data['TRAJS'] = trajs
        data["LANE_GRAPH"] = lane_graph
        data['RPE'] = rpes
        data['TGT_PTS'] = tgt_pts
        data['TGT_NODES'] = tgt_nodes
        data['TGT_ANCH'] = tgt_anch
        data['TGT_RPE'] = tgt_rpe

        return data, cur_data

    def is_condition_met(self, data):
        cov_change_rate = 9
        trajs_cov = data["TRAJS_COV_HIST"]
        cur_t = data["CUR_T"]
        end_t = data["END_T"]
        compare_t = self.obs_len + cur_t

        if cur_t == 0:
            compare_t += 1

        for t in range(cur_t + 1, end_t):
            # only check even time step
            if t % 2 == 1:
                continue

            # check if the covariance is changing too fast
            # for max sigma
            if torch.sum(trajs_cov[:, self.obs_len + t] / trajs_cov[:, compare_t] > cov_change_rate) > 0:
                data["END_T"] = t
                return False

        return True

    def get_branch_time(self, pred_data):
        cov_change_rate = 9
        trajs_cov = pred_data["TRAJS_COV_HIST"]
        cur_t = pred_data["CUR_T"]
        end_t = pred_data["END_T"]
        compare_t = self.obs_len + cur_t

        if cur_t == 0:
            compare_t += 1

        for t in range(cur_t + 1, end_t):
            # only check even time step to save computation
            if t % 2 == 1:
                continue

            # check if the covariance is changing too fast for max sigma
            if torch.sum(trajs_cov[:, self.obs_len + t] / trajs_cov[:, compare_t] > cov_change_rate) > 0:
                pred_data["END_T"] = t
                return t
        return end_t

    def get_high_level_command(self, orig, rot, cur_vel, min_vel=0.5):
        """
        根据当前状态和目标车道信息，生成高级行为命令。

        参数:
        - orig: 起始点坐标
        - rot: 旋转矩阵，用于坐标转换
        - cur_vel: 当前速度
        - min_vel: 最小速度，默认值为0.5

        返回:
        - tgt_pts: 目标车道点列表
        - tgt_nodes: 目标节点信息，包含中心点、向量和车道信息
        - tgt_anch: 目标锚点和向量
        """

        # get tgt lane: 计算目标车道上每一点到起始点的距离
        dists = torch.norm(self.target_lane - orig, dim=-1)
        # get the closest target lane point: 找到距离最近的目标车道点索引
        closest_idx = torch.argmin(dists)
        # 根据当前速度和预测时间计算将要行驶的距离
        travel_dist = max(cur_vel, min_vel) * self.config.tar_time_ahead
        # get approximation of the future area idx: 初始化目标点索引为最近点索引
        target_idx = closest_idx
        # 根据行驶距离调整目标点索引
        while target_idx < len(self.target_lane) - 1 and travel_dist > 0:
            target_idx += 1
            travel_dist -= torch.norm(self.target_lane[target_idx] - self.target_lane[target_idx - 1])

        # 确保目标点索引不在车道的最后五个点内
        if target_idx == len(self.target_lane) - 1:
            target_idx -= 1
        target_idx = max(5, min(target_idx, len(self.target_lane) - 6))

        # 选取目标车道点及其信息
        selected_idx = torch.arange(target_idx - 5, target_idx + 6)
        target_lane_pts = self.target_lane[selected_idx]
        target_lane_info = self.target_lane_info[selected_idx][1:]

        tgt_pts = copy.deepcopy(target_lane_pts)
        assert len(target_lane_pts) == 11

        # 复制并转换目标车道点到局部坐标系
        ctrln = copy.deepcopy(target_lane_pts)  # [num_sub_segs + 1, 2]
        ctrln = torch.matmul(ctrln - orig, rot) # to local frame

        # 计算锚点位置和向量
        anch_pos = torch.mean(ctrln, dim=0)
        anch_vec = (ctrln[-1] - ctrln[0]) / torch.norm(ctrln[-1] - ctrln[0])
        anch_rot = torch.tensor([[anch_vec[0], -anch_vec[1]],
                                 [anch_vec[1], anch_vec[0]]]).to(self.device)
        ctrln = torch.matmul(ctrln - anch_pos, anch_rot)    # to instance frame

        # 计算中心点和向量
        ctrs = (ctrln[:-1] + ctrln[1:]) / 2.0
        vecs = ctrln[1:] - ctrln[:-1]
        tgt_anch = [anch_pos, anch_vec]

        # convert to tensor: 合并目标节点信息
        # ~ calc tgt feat
        tgt_nodes = torch.cat([ctrs, vecs, target_lane_info], dim=-1)   # [N_{lane}, 16, F]
        return tgt_pts, tgt_nodes, tgt_anch
