import os

from launch_utils import get_argument_parser, create_log_dirs, save_args
from scenarios.benchmark import get_continual_scenario_benchmark, get_joint_scenario_benchmark
from models import social_stgcnn, social_stgcnn_pnn, social_stgcnn_dem
from learners import pnn_learner, vanilla_learner, multiple_learner, joint_learner, dem_learner, gsm_learner
from utils.utils import set_seeds


if __name__ == '__main__':
    args = get_argument_parser()
    create_log_dirs(args)
    save_args(args)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    if args.deterministic: set_seeds(args.seed)
    
    if args.train_method == "Joint":
        # train
        joint_scenario = get_joint_scenario_benchmark(args)
        continual_scenarios = get_continual_scenario_benchmark(args)
        model = social_stgcnn.social_stgcnn(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
        learner = joint_learner.JointLearner(model, joint_scenario, continual_scenarios, args)
    else:
        scenarios = get_continual_scenario_benchmark(args)
        # determine train method
        if args.train_method == "DEM":
            model_ = social_stgcnn_dem.social_stgcnn_dem
            learner_ = dem_learner.DEMLearner
        if args.train_method == "Vanilla":
            model_ = social_stgcnn.social_stgcnn
            learner_ = vanilla_learner.VanillaLearner
        if args.train_method == "GSM":
            model_ = social_stgcnn.social_stgcnn
            learner_ = gsm_learner.GSMLearner
        if args.train_method == "Multiple":
            model_ = social_stgcnn.social_stgcnn
            learner_ = multiple_learner.MultipleLearner
        if args.train_method == "PNN":
            model_ = social_stgcnn_pnn.social_stgcnn_pnn
            learner_ = pnn_learner.PNNLearner
        
        model = model_(n_stgcnn=args.n_stgcnn, n_txpcnn=args.n_txpcnn,
                        output_feat=args.output_size, seq_len=args.obs_seq_len,
                        kernel_size=args.kernel_size, pred_seq_len=args.pred_seq_len)
        learner = learner_(model, scenarios, args)
        
    learner.learn_tasks()
