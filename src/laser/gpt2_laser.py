import torch
import numpy as np
from copy import deepcopy
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class GPT2Laser(AbstractLaser):

    def __init__(self):
        super().__init__()  # 修正：调用父类初始化

    @staticmethod
    def convert_name(name):
        """
        将通用层名转换为GPT-2特定的层名
        """
        if name == "k_proj":
            return "attn.c_attn.weight"  # GPT-2中k_proj包含在c_attn中
        elif name == "q_proj":
            return "attn.c_attn.weight"  # GPT-2中q_proj包含在c_attn中
        elif name == "v_proj":
            return "attn.c_attn.weight"  # GPT-2中v_proj包含在c_attn中
        elif name == "out_proj":
            return "attn.c_proj.weight"
        elif name == "fc_in":
            return "mlp.c_fc.weight"
        elif name == "fc_out":
            return "mlp.c_proj.weight"
        elif name == "None":
            return "None"
        elif name == "mlp":
            return ["mlp.c_fc.weight", "mlp.c_proj.weight"]
        elif name == "attn":
            return ["attn.c_attn.weight", "attn.c_proj.weight"]
        elif name == "all":
            return ["attn.c_attn.weight", "attn.c_proj.weight", 
                    "mlp.c_fc.weight", "mlp.c_proj.weight"]
        else:
            raise AssertionError(f"未处理的层名: {name}")

    @staticmethod
    def _modify_layer(name, lnum_to_modify, lname_to_modify, converted_names):
        """检查参数是否需要修改"""
        # 检查层号匹配
        if lnum_to_modify != -1 and not name.startswith(f"transformer.h.{lnum_to_modify}."):
            return False

        # 检查层类型匹配
        if isinstance(converted_names, list):
            return any(name.endswith(cn) for cn in converted_names)
        elif isinstance(converted_names, str):
            return name.endswith(converted_names)
        else:
            raise AssertionError(f"转换后的名称类型应为list或str，实际为{type(converted_names)}")

    @staticmethod
    def _get_qkv_slice(lname, param_shape):
        """获取q_proj/k_proj/v_proj在c_attn中的切片位置"""
        hidden_size = param_shape[1] // 3
        if lname == "q_proj":
            return slice(0, hidden_size)
        elif lname == "k_proj":
            return slice(hidden_size, 2 * hidden_size)
        elif lname == "v_proj":
            return slice(2 * hidden_size, 3 * hidden_size)
        else:
            return None

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):
        """获取编辑后的模型"""
        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            if logger:
                logger.log("不进行任何干预")
            return model_edit

        converted_names = GPT2Laser.convert_name(lname)
        num_update = 0

        for name, param in model.named_parameters():
            modify_flag = GPT2Laser._modify_layer(
                name=name,
                lnum_to_modify=lnum,
                lname_to_modify=lname,
                converted_names=converted_names
            )

            if not modify_flag:
                continue

            if logger:
                logger.log(f"更新层: {name}")
            print(f"更新层: {name}")

            # 获取参数切片（针对q/k/v_proj）
            param_slice = None
            if lname in ["q_proj", "k_proj", "v_proj"] and "c_attn" in name:
                slice_idx = GPT2Laser._get_qkv_slice(lname, param.shape)
                param_slice = param[:, slice_idx]
            else:
                param_slice = param

            # 应用干预
            if intervention == 'dropout':
                mat_analysis = param_slice.detach().cpu().numpy().copy()
                mat_sort = sorted_mat(mat_analysis)
                mat_analysis = prune(mat_analysis, mat_sort, rate)
                mat_analysis = torch.from_numpy(mat_analysis).to(param.device)

            elif intervention == 'rank-reduction':
                # 确保在CPU上执行SVD以提高稳定性
                device = param_slice.device
                mat_analysis = do_low_rank(param_slice.cpu().float(), (10 - rate) * 0.1).to(device)

            elif intervention == 'zero':
                mat_analysis = torch.zeros_like(param_slice)

            else:
                raise AssertionError(f"未处理的干预类型: {intervention}")

            # 更新模型参数
            with torch.no_grad():
                if lname in ["q_proj", "k_proj", "v_proj"] and "c_attn" in name:
                    param[:, slice_idx] = mat_analysis
                else:
                    param.data = mat_analysis

            num_update += 1

        assert num_update > 0, f"必须更新一些参数: {lnum}, {lname}"

        if logger:
            logger.log(f"总共更新了 {num_update} 个参数")

        if lnum != -1 and lname not in ["all", "mlp", "attn"]:
            assert num_update == 1, f"应该只更新1个参数，但实际更新了 {num_update} 个"

        return model_edit