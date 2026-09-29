def get_model(usr_args):
    if str(usr_args.get("server_mode", "false")).lower() in {"1", "true", "yes"}:
        from .client_policy import get_model as client_get_model
        return client_get_model(usr_args)
    from .deploy_policy import get_model as local_get_model
    return local_get_model(usr_args)


def reset_model(model):
    return model.reset()


def eval(task_env, model, observation):
    return model.step(task_env, observation)
