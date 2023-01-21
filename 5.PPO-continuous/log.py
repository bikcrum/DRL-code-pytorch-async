from collections import OrderedDict
import hashlib, os, pickle


class color:
    BOLD   = '\033[1m\033[48m'
    END    = '\033[0m'
    ORANGE = '\033[38;5;202m'
    BLACK  = '\033[38;5;240m'

# Logger stores in trained_models by default
def create_logger(args):
    from torch.utils.tensorboard import SummaryWriter
    """Use hyperparms to set a directory to output diagnostic files."""

    arg_dict = args.__dict__
    assert "seed" in arg_dict, \
    "You must provide a 'seed' key in your command line arguments"
    assert "logdir" in arg_dict, \
    "You must provide a 'logdir' key in your command line arguments."
    assert "env" in arg_dict, \
    "You must provide a 'env' key in your command line arguments."

    # sort the keys so the same hyperparameters will always have the same hash
    arg_dict = OrderedDict(sorted(arg_dict.items(), key=lambda t: t[0]))

    # remove seed so it doesn't get hashed, store value for filename
    # same for logging directory
    run_name = arg_dict.pop('run_name')
    seed = str(arg_dict.pop("seed"))
    logdir = str(arg_dict.pop('logdir'))
    env_name = str(arg_dict['env'])

    # see if this run has a unique name, if so then that is going to be the name of the folder
    if run_name is not None:
        logdir = os.path.join(logdir, env_name)
        output_dir = os.path.join(logdir, run_name)
        # Check if policy name already exists. If it does, increment filename
        index = ''
        while os.path.exists(output_dir + index):
            if index:
                index = '_(' + str(int(index[2:-1]) + 1) + ')'
            else:
                index = '_(1)'
        output_dir += index
    else:
        # see if we are resuming a previous run, if we are mark as continued
        if hasattr(args, 'previous') and args.previous is not None:
            if args.exchange_reward is not None:
                output_dir = args.previous[0:-1] + "_NEW-" + args.reward
            else:
                print(args.previous[0:-1])
                output_dir = args.previous[0:-1] + '-cont'
        else:
            # get a unique hash for the hyperparameter settings, truncated at 10 chars
            arg_hash   = hashlib.md5(str(arg_dict).encode('ascii')).hexdigest()[0:6] + '-seed' + seed
            logdir     = os.path.join(logdir, env_name)
            output_dir = os.path.join(logdir, arg_hash)

    # create a directory with the hyperparm hash as its name, if it doesn't
    # already exist.
    os.makedirs(output_dir, exist_ok=True)

    # Create a file with all the hyperparam settings in human-readable plaintext,
    # also pickle file for resuming training easily
    info_path = os.path.join(output_dir, "experiment.info")
    pkl_path = os.path.join(output_dir, "experiment.pkl")
    with open(pkl_path, 'wb') as file:
        pickle.dump(args, file)
    with open(info_path, 'w') as file:
        for key, val in arg_dict.items():
            file.write("%s: %s" % (key, val))
            file.write('\n')

    logger = SummaryWriter(output_dir, flush_secs=0.1) # flush_secs=0.1 actually slows down quite a bit, even on parallelized set ups
    print("Logging to " + color.BOLD + color.ORANGE + str(output_dir) + color.END)

    logger.dir = output_dir
    return logger

# Rule for curriculum learning is that env observation space should be the same (so attributes like env.clock_based or env.state_est shouldn't be different and are forced to be same here)
# deal with loading hyperparameters of previous run continuation
def parse_previous(args):
    if hasattr(args, 'previous') and args.previous is not None:
        run_args = pickle.load(open(args.previous + "experiment.pkl", "rb"))
        args.simrate = run_args.simrate
        args.impedance = run_args.impedance
        args.arch = run_args.arch
        args.history = run_args.history
        args.env = run_args.env
        if args.exchange_reward is None:
            args.reward = run_args.reward
            args.run_name = run_args.run_name + "--cont"
        else:
            args.reward = args.exchange_reward
            args.run_name = run_args.run_name + "_NEW-" + args.reward

    return args
