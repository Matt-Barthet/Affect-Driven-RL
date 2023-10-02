def no_reward(agent_trace, human_trace):
    """

    :param agent_trace:
    :param human_trace:
    :return:
    """
    return 0


def regression_imitation(agent_trace, human_trace):
    """

    :param agent_trace:
    :param human_trace:
    :return:
    """
    return 0


def confidence_imitation(agent_trace, human_trace):
    """

    :param agent_trace:
    :param human_trace:
    :return:
    """
    return 0


def uncertainty_imitation(agent_trace, human_trace):
    """

    :param agent_trace:
    :param human_trace:
    :return:
    """
    return 0


reward_functions = {"No_Reward": no_reward,
                    "Regression": regression_imitation,
                    "Confidence": confidence_imitation,
                    "Uncertainty": uncertainty_imitation}
