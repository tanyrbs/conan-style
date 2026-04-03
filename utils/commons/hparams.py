import argparse
import os
import yaml

from utils.os_utils import remove_file

global_print_hparams = True
hparams = {}


class Args:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__setattr__(k, v)


def override_config(old_config: dict, new_config: dict):
    for k, v in new_config.items():
        if isinstance(v, dict) and k in old_config:
            override_config(old_config[k], new_config[k])
        else:
            old_config[k] = v


def load_config_recursive(config_fn, loaded_config=None, config_chains=None):
    # deep first inheritance and avoid the second visit of one node
    if not config_fn or not os.path.exists(config_fn):
        return {}
    config_fn = os.path.normpath(config_fn)
    if loaded_config is None:
        loaded_config = set()
    if config_chains is None:
        config_chains = []
    if config_fn in loaded_config:
        return {}
    with open(config_fn, encoding='utf-8') as f:
        hparams_ = yaml.safe_load(f) or {}
    loaded_config.add(config_fn)
    if 'base_config' in hparams_:
        ret_hparams = {}
        if not isinstance(hparams_['base_config'], list):
            hparams_['base_config'] = [hparams_['base_config']]
        for c in hparams_['base_config']:
            if c.startswith('.'):
                c = f'{os.path.dirname(config_fn)}/{c}'
                c = os.path.normpath(c)
            if c not in loaded_config:
                override_config(ret_hparams, load_config_recursive(
                    c, loaded_config=loaded_config, config_chains=config_chains))
        override_config(ret_hparams, hparams_)
    else:
        ret_hparams = hparams_
    config_chains.append(config_fn)
    return ret_hparams


def set_hparams(config='', exp_name='', hparams_str='', print_hparams=True, global_hparams=True):
    if config == '' and exp_name == '':
        parser = argparse.ArgumentParser(description='')
        parser.add_argument('--config', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--exp_name', type=str, default='', help='exp_name')
        parser.add_argument('-hp', '--hparams', type=str, default='',
                            help='location of the data corpus')
        parser.add_argument('--infer', action='store_true', help='infer')
        parser.add_argument('--validate', action='store_true', help='validate')
        parser.add_argument('--reset', action='store_true', help='reset hparams')
        parser.add_argument('--remove', action='store_true', help='remove old ckpt')
        parser.add_argument('--debug', action='store_true', help='debug')
        args, unknown = parser.parse_known_args()
        print("| Unknown hparams: ", unknown)
    else:
        args = Args(config=config, exp_name=exp_name, hparams=hparams_str,
                    infer=False, validate=False, reset=False, debug=False, remove=False)
    global hparams
    assert args.config != '' or args.exp_name != ''
    if args.config != '':
        assert os.path.exists(args.config)

    config_chains = []
    saved_hparams = {}
    args_work_dir = ''
    ckpt_config_path = ''
    if args.exp_name != '':
        args_work_dir = f'checkpoints/{args.exp_name}'
        ckpt_config_path = f'{args_work_dir}/config.yaml'
        if os.path.exists(ckpt_config_path):
            saved_hparams_ = load_config_recursive(
                ckpt_config_path, loaded_config=set(), config_chains=config_chains)
            if saved_hparams_ is not None:
                saved_hparams.update(saved_hparams_)
    hparams_ = {}
    if not args.reset:
        override_config(hparams_, saved_hparams)
    if args.config != '':
        override_config(hparams_, load_config_recursive(
            args.config, loaded_config=set(), config_chains=config_chains))
    if args_work_dir != '':
        hparams_['work_dir'] = args_work_dir
    else:
        hparams_.setdefault('work_dir', '')

    # Support config overriding in command line. Support list type config overriding.
    # Examples: --hparams="a=1,b.c=2,d=[1 1 1]"
    if args.hparams != "":
        for new_hparam in args.hparams.split(","):
            k, v = new_hparam.split("=")
            v = v.strip("\'\" ")
            config_node = hparams_
            key_path = k.split(".")
            traversed = []
            for k_ in key_path[:-1]:
                traversed.append(k_)
                if k_ not in config_node or not isinstance(config_node[k_], dict):
                    current_path = ".".join(traversed)
                    raise KeyError(
                        f"Unknown hparam override path '{k}'. "
                        f"Missing intermediate key '{current_path}'."
                    )
                config_node = config_node[k_]
            k = key_path[-1]
            if k not in config_node:
                available_keys = ", ".join(sorted(map(str, config_node.keys())))
                raise KeyError(
                    f"Unknown hparam override key '{k}' in '{new_hparam}'. "
                    f"Available keys at this level: [{available_keys}]"
                )
            if v in ['True', 'False'] or type(config_node[k]) in [bool, list, dict]:
                if type(config_node[k]) == list:
                    v = v.replace(" ", ",")
                config_node[k] = eval(v)
            else:
                config_node[k] = type(config_node[k])(v)
    if args_work_dir != '' and args.remove:
        answer = input("REMOVE old checkpoint? Y/N [Default: N]: ")
        if answer.lower() == "y":
            remove_file(args_work_dir)
    if args_work_dir != '' and (not os.path.exists(ckpt_config_path) or args.reset) and not args.infer:
        os.makedirs(hparams_['work_dir'], exist_ok=True)
        with open(ckpt_config_path, 'w') as f:
            yaml.safe_dump(hparams_, f)

    hparams_['infer'] = args.infer
    hparams_['debug'] = args.debug
    hparams_['validate'] = args.validate
    hparams_['exp_name'] = args.exp_name
    global global_print_hparams
    if global_hparams:
        hparams.clear()
        hparams.update(hparams_)
    if print_hparams and global_print_hparams and global_hparams:
        print('| Hparams chains: ', config_chains)
        print('| Hparams: ')
        for i, (k, v) in enumerate(sorted(hparams_.items())):
            print(f"\033[;33;m{k}\033[0m: {v}, ", end="\n" if i % 5 == 4 else "")
        print("")
        global_print_hparams = False
    return hparams_
