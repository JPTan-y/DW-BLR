from copy import deepcopy
import numpy as np

def weight_share(n_train=None):
    if n_train is None:
        return '50/n'
    if isinstance(n_train, (bool, np.bool_)) or int(n_train) != n_train or n_train < 2:
        raise ValueError('n_train must be an integer of at least 2')
    return min(1.0, 50.0 / int(n_train))

def density_parameters(n_train=None):
    return dict(hidden_layer_sizes=(16, 8), activation='relu', solver='adam', l2=0.05, dropout=0.1, learning_rate=0.001, batch_size=200, max_epochs=300, early_stopping=True, validation_fraction=0.2, patience=20, tol=0.0001, gradient_clip=5.0, probability_clip=1e-06, weight_clip_quantile=None, max_weight_share=weight_share(n_train), sensitive_variables=None, random_state=9527, device='cpu', verbose=False)

def srdo_parameters(n_train=None):
    params = density_parameters(n_train)
    params.update(curvature_floor=1e-06, ignore_high_confidence_correct=False, corrected_weight_clip_quantile=None, logistic_optimizer='lbfgs', logistic_max_iter=500, logistic_tolerance=1e-08)
    return params

def svi_parameters(n_train=None):
    return dict(selector_hidden_layer_sizes=(4, 2), selector_activation='relu', selector_dropout=0.0, gate_sigma=0.5, gate_mu_init=0.5, selector_gate_lr=0.001, selector_predictor_lr=0.001, selector_pretrain_epochs=30, selector_max_epochs=2000, selector_min_epochs=50, selector_batch_size=256, selector_weight_decay=0.0001, selector_tol=1e-05, selector_patience=30, selector_gradient_clip=5.0, regularizer_normalization='sum', selection_threshold=0.5, min_features=1, C=0.03, penalty_dimension=30, lambda_stg=None, final_l2=0.0, final_use_srdo_weights=False, final_tol=1e-12, random_state=9527, device='cpu', verbose=False, srdo_params=density_parameters(n_train))

def dwr_parameters():
    return dict(dwr_eta=0.03, dwr_max_iter=1000, dwr_tol=1e-07, logistic_optimizer='lbfgs', logistic_max_iter=500, logistic_tolerance=1e-08, curvature_floor=1e-06, ignore_high_confidence_correct=False, corrected_weight_clip_quantile=None, probability_clip=1e-08, random_state=2026, device='cpu', verbose=False)

def vblr_parameters(n_features=None):
    return dict(tau=0.5, pi0_init=0.5, alpha_init=None if n_features is None else [0.5] * int(n_features), prior_inclusion_prob=None, sigma0_sq_init=1.0, sigma0_sq_max=None, sigma_b_sq=10.0, empirical_bayes=True, max_iter=500, tol=1e-06, delta_c=1e-08, delta_h=1e-06, verbose=False, log_every=20)

def dw_blr_parameters(n_features=None):
    return dict(auto_prior=True, optimization_mode='feature_em', prior_update_scope='outer', prior_inclusion_prob=None if n_features is None else [0.5] * int(n_features), pi0_init=0.5, n_subsamples=100, subsample_fraction=0.9, subsample_stratify=True, subsample_replace=False, random_state=999, pilot_gamma_w=0.05, pilot_warmup_sweeps=5, pilot_max_rounds=200, pilot_min_rounds=5, pilot_stable_rounds=5, pilot_outer_tol=1e-05, frequency_tau=0.5, prior_floor=0.1, frequency_mix=0.8, prior_clip=0.01, gamma_w=0.05, warmup_sweeps=5, unprotected_rounds=0, T_max=1000, tau=0.5, weight_damping=0.6, lambda_damping=0.6, prior_damping=0.6, sigma0_sq_init=1.0, slab_update='outer_eb', slab_damping=0.6, sigma_b_sq=10.0, output_dir=None)

def penalized_parameters(method):
    result = dict(lambda_=0.02, max_iter=5000, tol=1e-07, random_state=2026, verbose=False)
    if method == 'Elastic Net':
        result['alpha'] = 0.8
    elif method == 'SCAD':
        result.update(a=5.0, nonconvex_init='lasso', nonconvex_multi_start=False)
    elif method != 'Lasso':
        raise ValueError('Unknown penalized method')
    return result

def all_parameters(n_train=None, n_features=30):
    return deepcopy({'Lasso': penalized_parameters('Lasso'), 'Elastic Net': penalized_parameters('Elastic Net'), 'SCAD': penalized_parameters('SCAD'), 'SRDO': srdo_parameters(n_train), 'SVI': svi_parameters(n_train), 'DWR': dwr_parameters(), 'VBLR': vblr_parameters(n_features), 'DW-BLR': dw_blr_parameters(n_features)})
