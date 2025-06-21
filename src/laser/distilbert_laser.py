from copy import deepcopy
import torch
from laser.abstract_laser import AbstractLaser
from laser.matrix_utils import do_low_rank, sorted_mat, prune


class DistilBERTLaser(AbstractLaser):

    def __init__(self):
        pass

    @staticmethod
    def convert_name(name):
        if name == "k_proj":
            return "attention.key.weight"
        elif name == "q_proj":
            return "attention.query.weight"
        elif name == "v_proj":
            return "attention.value.weight"
        elif name == "out_proj":
            return "attention.out_proj.weight"
        elif name == "fc_in":
            return "ffn.lin1.weight"
        elif name == "fc_out":
            return "ffn.lin2.weight"
        elif name == "None":
            return "None"
        else:
            raise AssertionError(f"Unhandled name {name}")

    @staticmethod
    def get_edited_model(model, lname, lnum, rate, intervention="rank-reduction", logger=None, in_place=True):
        if in_place:
            model_edit = model
        else:
            model_edit = deepcopy(model)

        if lname == "dont":
            print("Not intervening at all")
            return model_edit

        num_update = 0
        for name, param in model.named_parameters():
            if not name.startswith(f"distilbert.transformer.layer.{lnum}"):
                continue

            converted_name = DistilBERTLaser.convert_name(lname)
            if lname != "None" and not name.endswith(converted_name):
                continue

            if logger:
                logger.log(f"Updating Layer: distilbert.transformer.layer.{lnum}.{converted_name}")
            print(f"Updating Layer: distilbert.transformer.layer.{lnum}.{converted_name}")

            mat_analysis_tensor = param.data.clone().float()

            if intervention == 'dropout':
                mat_sort = sorted_mat(mat_analysis_tensor.numpy())
                mat_analysis = prune(mat_analysis_tensor.numpy(), mat_sort, rate)
                mat_analysis = torch.from_numpy(mat_analysis)
            elif intervention == 'rank-reduction':
                mat_analysis = do_low_rank(mat_analysis_tensor, (10 - rate) * 0.1, niter=20)
            elif intervention == 'zero':
                mat_analysis = torch.zeros_like(mat_analysis_tensor)
            else:
                raise AssertionError(f"Unhandled intervention type {intervention}")

            DistilBERTLaser.update_model(model_edit, name, mat_analysis)
            num_update += 1

        assert num_update == 1, f"Expected 1 update but made {num_update}"

        return model_edit

    @staticmethod
    def update_model(model, name, mat_analysis):
        # 直接更新模型参数
        parts = name.split('.')
        module = model
        for part in parts[:-1]:
            module = getattr(module, part)
        module_weight = getattr(module, parts[-1])
        module_weight.data = torch.nn.Parameter(mat_analysis.to(module_weight.device))