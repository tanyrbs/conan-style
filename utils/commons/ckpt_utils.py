import glob
import os
import re

import torch


_CHECKPOINT_STEP_RE = re.compile(r".*steps_(\d+)\.ckpt$")


def _checkpoint_step_from_path(path: str) -> int:
    match = _CHECKPOINT_STEP_RE.match(str(path))
    if match is None:
        return -1
    return int(match.group(1))


def _raise_or_log_missing_ckpt(base_dir: str, *, force: bool):
    e_msg = f"| ckpt not found in {base_dir}."
    if force:
        raise FileNotFoundError(e_msg)
    print(e_msg)


def get_last_checkpoint(work_dir, steps=None):
    checkpoint = None
    last_ckpt_path = None
    ckpt_paths = get_all_ckpts(work_dir, steps)
    if len(ckpt_paths) > 0:
        last_ckpt_path = ckpt_paths[0]
        checkpoint = torch.load(last_ckpt_path, map_location="cpu")
    return checkpoint, last_ckpt_path


def get_all_ckpts(work_dir, steps=None):
    if steps is None:
        ckpt_path_pattern = f"{work_dir}/model_ckpt_steps_*.ckpt"
    else:
        ckpt_path_pattern = f"{work_dir}/model_ckpt_steps_{steps}.ckpt"
    ckpt_paths = glob.glob(ckpt_path_pattern)
    return sorted(
        ckpt_paths,
        key=lambda x: (-_checkpoint_step_from_path(x), str(x)),
    )


def load_ckpt(cur_model, ckpt_base_dir, model_name="model", force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location="cpu")
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if len([k for k in state_dict.keys() if "." in k]) > 0:
            state_dict = {
                k[len(model_name) + 1:]: v
                for k, v in state_dict.items()
                if k.startswith(f"{model_name}.")
            }
        else:
            if "." not in model_name:
                state_dict = state_dict[model_name]
            else:
                base_model_name = model_name.split(".")[0]
                rest_model_name = model_name[len(base_model_name) + 1:]
                state_dict = {
                    k[len(rest_model_name) + 1:]: v
                    for k, v in state_dict[base_model_name].items()
                    if k.startswith(f"{rest_model_name}.")
                }
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        _raise_or_log_missing_ckpt(base_dir, force=force)


def load_ckpt_emformer(cur_model, ckpt_base_dir, model_name="model", force=True, strict=True):
    if os.path.isfile(ckpt_base_dir):
        base_dir = os.path.dirname(ckpt_base_dir)
        ckpt_path = ckpt_base_dir
        checkpoint = torch.load(ckpt_base_dir, map_location="cpu")
    else:
        base_dir = ckpt_base_dir
        checkpoint, ckpt_path = get_last_checkpoint(ckpt_base_dir)
    if checkpoint is not None:
        state_dict = checkpoint["state_dict"]
        if not strict:
            cur_model_state_dict = cur_model.state_dict()
            unmatched_keys = []
            for key, param in state_dict.items():
                if key in cur_model_state_dict:
                    new_param = cur_model_state_dict[key]
                    if new_param.shape != param.shape:
                        unmatched_keys.append(key)
                        print("| Unmatched keys: ", key, new_param.shape, param.shape)
            for key in unmatched_keys:
                del state_dict[key]
        cur_model.load_state_dict(state_dict, strict=strict)
        print(f"| load '{model_name}' from '{ckpt_path}'.")
    else:
        _raise_or_log_missing_ckpt(base_dir, force=force)
