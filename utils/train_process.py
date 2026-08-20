"""
author: Seongho Son (seong.son.22@ucl.ac.uk)

Process used to hold an agent (and selector) model.
"""

import time


def game_process(
    game_id,
    qin,
    qin_response,
    qouts_agent,
    qouts_train_response,
    game,
    time_sleep=0.001,
):
    while True:
        qget = qin.get()
        cmd = qget["cmd"]
        if cmd == "abort":
            break
        if cmd == "step":
            me_id = qget["me_id"]
            op_id = qget["op_id"]
            me_state = qget["me_state"]
            op_state = qget["op_state"]
            me_action = None
            op_action = None

            qouts_agent[me_id].put({"cmd": "act", "role": "me", "state": me_state})
            qouts_agent[op_id].put({"cmd": "act", "role": "op", "state": op_state})

            for i in range(2):
                qget_response = qin_response.get()
                if qget_response["role"] == "me":
                    me_action = qget_response["action"]
                elif qget_response["role"] == "op":
                    op_action = qget_response["action"]

            rewards, new_states = game.step((me_action, op_action))

            qouts_train_response[me_id].put({"reward": rewards[0], "action": op_action})
        time.sleep(time_sleep)


def agent_act_process(agent_id, qin, qout_train, qouts, model, time_sleep=0.001):
    while True:
        qget = qin.get()
        cmd = qget["cmd"]
        if cmd == "abort":
            break
        if cmd == "act":
            game_id = qget["game_id"]
            state = qget["state"]
            action = model.act(state)
            # use experience for training
            if qget["role"] == "me":
                qout_train.put(
                    {
                        "cmd": "train",
                        "state": state,
                        "action": action,
                    }
                )
            qouts[game_id].put(
                {
                    "cmd": "response",
                    "role": qget["role"],
                    "agent_id": agent_id,
                    "action": action,
                }
            )

        time.sleep(time_sleep)


def agent_train_process(qin, qin_response, model, time_sleep=0.001):
    while True:
        qget = qin.get()
        cmd = qget["cmd"]
        if cmd == "abort":
            break
        if cmd == "train":
            state = cmd["state"]
            action = cmd["action"]
            qget_response = qin_response.get()
            reward = qget_response["reward"]
            actions = [action, qget_response["action"]]

            history = model.history
            old_state = history["states"][-1]
            old_reward = history["rewards"][-1]
            old_action = history["actions"][-1][0]
            model.store("states", state)
            model.store("rewards", reward)
            model.store("actions", actions)

            model.step_train(old_state, old_action, old_reward, history["states"][-1])

        time.sleep(time_sleep)


def selector_act_process(qin, qout, model, game, time_sleep=0.001):
    while True:
        qget = qin.get()
        cmd = qget["cmd"]
        if cmd == "abort":
            break

        time.sleep(time_sleep)


def selector_train_process(qin, qout, model, game, time_sleep=0.001):
    while True:
        qget = qin.get()
        cmd = qget["cmd"]
        if cmd == "abort":
            break

        time.sleep(time_sleep)
