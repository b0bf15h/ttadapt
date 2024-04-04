import os
import torch
import logging
import numpy as np
import methods
import time
import torch.nn.functional as F
import json
import foolbox as fb
from autoattack import AutoAttack
from models.model import get_model
from utils.eval_utils import get_accuracy, eval_domain_dict
from robustbench.data import load_cifar10
from robustbench.data import load_cifar100
from robustbench.data import load_imagenet
from utils.registry import ADAPTATION_REGISTRY
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, ckpt_path_to_domain_seq
import pickle
from copy import deepcopy

logger = logging.getLogger(__name__)


def linf_attack(data,m, epsilons, epsilons_s, domain_name, severity, output_dir, architecture, adaptation):
  x_test = data[0]
  y_test = data[1]
  prefix = str(severity)+'_'+domain_name 
  keys = epsilons_s
  if hasattr(m, 'model'):
    m = m.model
  else:
    m = m
  m.eval()
  m.to('cuda:0')
  accuracy = torch.zeros(6).to('cuda:0')
  fmodel = fb.PyTorchModel(m, bounds=(0, 1))
  for i in range(int(len(x_test)/100)):
      lower = i*100
      upper = (i+1)*100
      x = x_test[lower:upper]
      y = y_test[lower:upper]
      _,_, is_adv =  fb.attacks.LinfPGD()(fmodel, x.to('cuda:0'), y.to('cuda:0'), epsilons=epsilons)
      robust_accuracy = 1 - is_adv.float().mean(axis=-1)
      accuracy+=robust_accuracy
  #model.train()
  vals = accuracy.tolist()
  values = [round(v,2)/5 for v in vals]
  dic = dict(zip(keys, values))
  file_path = architecture+'_'+adaptation+'_'+prefix+'.json'
  full_path = os.path.join(output_dir, file_path)
  with open(full_path, 'w') as file:
    json.dump(dic, file)
  return dic
def linf_aa(model,data,eps, output_dir,domain):
    dic = {}
    # dom = '_'.join(domain.split('_')[1:])
    # ada = domain.split('_')[0]
    # if ada=='source' and dom!='gaussian_noise':
    #     logger.info('Not running attack, source model does not change')
    #     return
    for e in eps:
        epsilon = eval(e)
        if epsilon == 0.0:
            acc = get_model_accuracy(model,data[0],data[1])
        else:
            acc = aa_partial(model,data,epsilon)
        dic[e] = acc
    file_name = domain+'.json'
    output_path = os.path.join(output_dir, file_name)
    with open(output_path, 'w') as file:
        json.dump(dic,file)
    
def aa_partial(model, data, epsilon):
    x_test, y_test = data
    adversary = AutoAttack(model, norm='Linf', seed=42, eps=epsilon, version='custom', attacks_to_run=['apgd-ce', 'apgd-dlr'])
    adversary.apgd.n_restarts = 1
    _,acc = adversary.run_standard_evaluation(x_test, y_test)
    return acc
def get_model_accuracy(model,data,label):
    output = model.forward(data.to('cuda:0'))  # input_data is your input to the model
    probabilities = F.softmax(output, dim=1)    
    predicted_labels = torch.argmax(probabilities, dim=1)
    predicted_labels = predicted_labels.cpu().numpy()
    orig_labels = label.cpu().numpy()
    errors = sum(1 for x, y in zip(predicted_labels, orig_labels) if x != y)
    return (100-errors)/100
def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "mixed_domains",              # consecutive test samples are likely to originate from different domains
                      "correlated",                 # sorted by class label
                      "mixed_domains_correlated",   # mixed domains + sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)

    # get the base model and its corresponding input pre-processing (if available)
    base_model, model_preprocess = get_model(cfg, num_classes, device)
    # append the input pre-processing to the base model
    base_model.model_preprocess = model_preprocess
    logger.info(f'BASE MODEL STATE DICT NUM KEYS {len(list(base_model.state_dict().keys()))}')
    # setup test-time adaptation method
    available_adaptations = ADAPTATION_REGISTRY.registered_names()
    assert cfg.MODEL.ADAPTATION in available_adaptations, \
        f"The adaptation '{cfg.MODEL.ADAPTATION}' is not supported! Choose from: {available_adaptations}"
    model = ADAPTATION_REGISTRY.get(cfg.MODEL.ADAPTATION)(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f'BASE MODEL STATE DICT NUM KEYS {len(list(base_model.state_dict().keys()))}')

    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION}")
    

    # get the test sequence containing the corruptions or domain names
    if cfg.CORRUPTION.DATASET == "domainnet126":
        # extract the domain sequence for a specific checkpoint.
        domain_sequence = ckpt_path_to_domain_seq(ckpt_path=cfg.MODEL.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in ["imagenet_d", "imagenet_d109"] and not cfg.CORRUPTION.TYPE[0]:
        # domain_sequence = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        domain_sequence = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        domain_sequence = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {domain_sequence}")

    # prevent iterating multiple times over the same data in the mixed_domains setting
    domain_seq_loop = ["mixed"] if "mixed_domains" in cfg.SETTING else domain_sequence

    # setup the severities for the gradual setting
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in ["cifar10_c", "cifar100_c", "imagenet_c"] and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    errs = []
    errs_5 = []
    domain_dict = {}
    #data = load_cifar10(n_examples= 100)
    data = load_cifar100(n_examples= 100)
    epsilons = ['0/255', '2/255', '4/255', '6/255', '8/255', '10/255']
    start_time = time.time()
    # start evaluation
    for i_dom, domain_name in enumerate(domain_seq_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except AttributeError:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                               adaptation=cfg.MODEL.ADAPTATION,
                                               dataset_name=cfg.CORRUPTION.DATASET,
                                               preprocess=model_preprocess,
                                               data_root_dir=cfg.DATA_DIR,
                                               domain_name=domain_name,
                                               domain_names_all=domain_sequence,
                                               severity=severity,
                                               num_examples=cfg.CORRUPTION.NUM_EX,
                                               rng_seed=cfg.RNG_SEED,
                                               delta_dirichlet=cfg.TEST.DELTA_DIRICHLET,
                                               batch_size=cfg.TEST.BATCH_SIZE,
                                               shuffle=False,
                                               workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            #evaluate the model
            logger.info(f'{len(list(model.state_dict().keys()))}')
            acc, domain_dict, num_samples = get_accuracy(model,
                                                         data_loader=test_data_loader,
                                                         dataset_name=cfg.CORRUPTION.DATASET,
                                                         domain_name=domain_name,
                                                         setting=cfg.SETTING,
                                                         domain_dict=domain_dict,
                                                         print_every=cfg.PRINT_EVERY,
                                                         device=device)
            #logger.info(f'{model.state_dict().keys()}')
            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)
            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={num_samples}]: {err:.2%}")

            json_name = cfg.MODEL.ADAPTATION +'_'+domain_name
    if cfg.MODEL.ADAPTATION =='rotta':
        m= deepcopy(model.model_ema)
        logger.info('USING TEACHER MODEL')
    else:
        logger.info('USING NORMAL MODEL')
        m=deepcopy(model.model)
    
    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time / 3600)
    minutes = int((run_time - hours * 3600) / 60)
    seconds = int(run_time - hours * 3600 - minutes * 60)
    logger.info(f"total run time: {hours}h {minutes}m {seconds}s")
    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

    if "mixed_domains" in cfg.SETTING and len(domain_dict.values()) > 0:
        # print detailed results for each domain
        eval_domain_dict(domain_dict, domain_seq=domain_sequence)
    linf_aa(model = m, data = data, eps = epsilons, output_dir = cfg.SAVE_DIR, domain = json_name)

if __name__ == '__main__':
    evaluate('"Evaluation.')
