from launch_utils import get_argument_parser,set_seeds, get_log_param_dict, create_log_dirs
from DECIBL import learner, psstgcnn
from DECIBL.train_eval import get_device
import os
from scenarios.benchmark import get_continual_scenario_benchmark


if __name__ == '__main__':
    args = get_argument_parser()
    log_params = get_log_param_dict(args)
    create_log_dirs(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.deterministic: set_seeds(args.seed)
    scenario = get_continual_scenario_benchmark(args)
    
    network = psstgcnn.progressive_social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                      output_feat=args.output_size, seq_len=args.obs_seq_len,
                      kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)

    net_learner = learner.Learner(args, network, scenario, log_params)
    network = net_learner.learn_tasks()
