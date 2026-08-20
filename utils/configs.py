"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

functions for managing configs
"""

import os

import omegaconf


def assign_default(target, key, value):
    """
    assign a default value to the target if it does not exist.
    """
    if key not in target:
        target[key] = value
    return target


def init_agentnet(config):
    """
    initialise the AgentNetwork part of the configuration.
    """
    config = assign_default(config, "agentnet", dict())
    cfg_net = config["agentnet"]
    cfg_net = assign_default(cfg_net, "name", "complete")
    # cfg_net = assign_default(cfg_net, "modifications", dict())

    if "modifications" in cfg_net:
        cfg_mod = cfg_net["modifications"]
        cfg_mod = assign_default(cfg_mod, "modify_step", 100)
        cfg_mod = assign_default(cfg_mod, "randomly_choose_agent", False)
        cfg_mod = assign_default(cfg_mod, "preference_criterion", "degree")
        # cfg_mod = assign_default(cfg_mod, "preferential_detachment", dict())
        # cfg_mod = assign_default(cfg_mod, "preferential_attachment", dict())

        if "preferential_detachment" in cfg_mod:
            cfg_pdt = config["agentnet"]["modifications"]["preferential_detachment"]
            cfg_pdt = assign_default(cfg_pdt, "select_high", True)
            cfg_pdt = assign_default(cfg_pdt, "select_prob", False)
            cfg_pdt = assign_default(cfg_pdt, "min_neighbours", 1)
        if "preferential_attachment" in cfg_mod:
            cfg_pat = config["agentnet"]["modifications"]["preferential_attachment"]
            cfg_pat = assign_default(cfg_pat, "select_high", False)
            cfg_pat = assign_default(cfg_pat, "select_prob", False)

    cfg_net = assign_default(cfg_net, "intermediate_plot", False)
    cfg_net = assign_default(cfg_net, "frequency_plot", False)

    if config["agentnet"]["name"] == "ER":
        cfg_net = assign_default(cfg_net, "no_isolate", True)

    cfg_net = assign_default(cfg_net, "growing", False)
    cfg_net = assign_default(cfg_net, "growing_schedule", 0.5)

    cfg_net = assign_default(cfg_net, "netsave_step", config["record_step"] * 10)
    if (cfg_net["netsave_step"] < config["record_step"]) or (
        cfg_net["netsave_step"] % config["record_step"] != 0
    ):
        cfg_net["netsave_step"] = config["record_step"] * 10

    return config


def init_config(config):
    """
    assign default values for keys that are not present in the config.
    arguments:
        config
    return:
        config with missing(but unnecessary) keys filled with default values
    """
    if "device" not in config:
        config["device"] = "cpu"
    if "profile_time" not in config:
        config["profile_time"] = False

    # `record_step` must be defaulted before init_agentnet() runs, because
    # init_agentnet() derives `agentnet.netsave_step` from it.
    config = assign_default(config, "record_step", 1000)

    config = assign_default(config, "iteration_modifier", 0)
    config = assign_default(config, "select_partner", False)
    config = assign_default(config, "select_partner2", False)
    config = assign_default(config, "freq_rewire", -1)
    config = assign_default(config, "selected_history", False)
    config = assign_default(config, "update_frequency", 1)

    # agent parallelisation
    if "parallelisation" not in config:
        config["parallelisation"] = dict()
    if "time_sleep" not in config["parallelisation"]:
        config["parallelisation"]["time_sleep"] = 0.001

    if "save_models" not in config:
        config["save_models"] = True
    if "print_loss" not in config:
        config["print_loss"] = False

    if "random_seeds" not in config:
        config["random_seeds"] = dict()
    if "experiment" not in config["random_seeds"]:
        config["random_seeds"]["experiment"] = 2329
    if "agent" not in config["random_seeds"]:
        config["random_seeds"]["agent"] = 3247
    if "selector" not in config["random_seeds"]:
        config["random_seeds"]["selector"] = 8761
    if "network" not in config["random_seeds"]:
        config["random_seeds"]["network"] = 4167
    if "state" not in config["random_seeds"]:
        config["random_seeds"]["state"] = 5397

    if "multiplier" not in config["matgame"]:
        config["matgame"]["multiplier"] = 1

    if "state" not in config:
        config["state"] = dict()
    if "len_history" not in config["state"]:
        config["state"]["len_history"] = 1
    if "history_type" not in config["state"]:
        config["state"]["history_type"] = "single"
    if "agent_id" not in config["state"]:
        config["state"]["agent_id"] = "none"
    if "include_opponent_history" not in config["state"]:
        config["state"]["include_opponent_history"] = False

    if "decay_epsilon" not in config:
        config["decay_epsilon"] = False
    elif "decay_epsilon_schedule" not in config:
        config["decay_epsilon_schedule"] = 0.9

    config = init_agentnet(config)

    if "profile_qvalues" not in config:
        config["profile_qvalues"] = False

    return config


def overwrite_config(d1, d2):
    """
    overwrite contents of d2 to d1 if overlapping.
    """
    if isinstance(d2, omegaconf.dictconfig.DictConfig):
        for k in d2:
            if k in d1:
                d1[k] = overwrite_config(d1[k], d2[k])
            else:
                d1[k] = d2[k]
        return d1
    else:
        return d2


def prepare_config(d_set, config_setting):
    """
    if common.yaml exists, use it as the basic format
    and overwrite with the given config
    arguments:
        d_set
        config
    return:
        overwritten config
    """
    path_common = f"configs/{d_set}common.yaml"
    if d_set != "" and os.path.exists(path_common):
        config = init_config(omegaconf.OmegaConf.load(path_common))
        config = overwrite_config(config, config_setting)
    else:
        config = init_config(config_setting)

    return config
