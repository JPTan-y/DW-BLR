from copy import deepcopy

SEED = 9527
N_TRAIN = 466
N_FEATURES = 100
METHODS = ("Lasso", "Elastic Net", "SCAD", "SRDO", "SVI", "DWR", "VBLR", "DW-BLR")
DWBLR_FINAL_PIP_THRESHOLD = 0.5


def weight_share(n_train=None):
    return 0.005


def density_parameters(n_train=None):
    return dict(
        hidden_layer_sizes=(16, 8), activation="relu", dropout=0.1,
        solver="adam", learning_rate=0.001, l2=0.05, batch_size=200,
        max_epochs=300, tol=1e-4, patience=20, early_stopping=True,
        validation_fraction=0.2, gradient_clip=5.0, probability_clip=1e-6,
        weight_clip_quantile=None, max_weight_share=weight_share(n_train),
        sensitive_variables=None, standardize=True, random_state=SEED,
        device="cpu", verbose=True, log_every=50,
    )


def _internal_weighted_fit():
    return dict(
        lambda_l2=0.1, intercept_penalized=False,
        weights_mean_one=True,
        optimizer=dict(method="trust-exact", gtol=1e-10, maxiter=2000),
        independent_check=dict(method="BFGS", gtol=1e-10, maxiter=4000),
        verification=dict(gradient_inf_lt=1e-7, coefficient_max_difference_lt=1e-5),
    )


def dwr_parameters():
    return dict(
        weight_learning=dict(
            eta=0.1, max_iter=1000, tol=1e-7, min_iter=5,
            initial_step=1.0, backtracking_factor=0.5, max_backtracking=50,
            step_growth=1.2, standardize=True, verbose=True, log_every=50,
        ),
        curvature=dict(
            pilot_probability="in_sample_training_probability",
            probability_clip=1e-8, curvature_floor=1e-6,
            ignore_high_confidence_correct=False,
            corrected_weight_clip_quantile=None, final_max_weight_share=None,
            normalize_mean_one=True,
        ),
        pilot_fit=_internal_weighted_fit(),
        final_weighted_fit=_internal_weighted_fit(),
    )


def srdo_parameters(n_train=None):
    return dict(
        density=density_parameters(n_train),
        curvature=dict(
            base="raw_density_ratio_before_standalone_cap",
            pilot_probability="in_sample_training_probability",
            probability_clip=1e-6, curvature_floor=1e-6,
            ignore_high_confidence_correct=False,
            corrected_weight_clip_quantile=None,
            final_max_weight_share=weight_share(n_train), normalize_mean_one=True,
        ),
        pilot_fit=_internal_weighted_fit(),
        final_weighted_fit=_internal_weighted_fit(),
    )


def svi_parameters(n_train=None):
    return dict(
        srdo_hidden_layer_sizes=(16, 8), srdo_activation="relu", srdo_dropout=0.1,
        srdo_solver="adam", srdo_alpha=0.05, srdo_learning_rate_init=0.001,
        srdo_max_iter=300, srdo_tol=1e-4, srdo_batch_size=200,
        srdo_early_stopping=True, srdo_validation_fraction=0.2,
        srdo_n_iter_no_change=20, srdo_gradient_clip=5.0, srdo_cross_fit_folds=1,
        srdo_probability_clip=1e-6, srdo_weight_clip_quantile=None,
        srdo_max_weight_share=weight_share(n_train),
        precomputed_srdo_weights=None, precomputed_srdo_diagnostics=None,
        selector_hidden_layer_sizes=(64, 32), selector_activation="relu",
        selector_dropout=0.0, gate_sigma=0.5, gate_mu_init=0.0,
        lambda_stg=0.008, regularizer_normalization="sum",
        selection_threshold=0.5, n_features_to_select=None, min_features=1,
        selector_pretrain_epochs=100, selector_max_epochs=10000,
        selector_min_epochs=100, selector_batch_size=256,
        selector_predictor_lr=0.001, selector_gate_lr=0.005,
        selector_weight_decay=1e-4, selector_tol=1e-5, selector_patience=30,
        selector_gradient_clip=5.0, selector_log_every=100,
        final_use_srdo_weights=False, final_l2=0.1,
        final_max_iter=2000, final_tol=1e-9,
        standardize=True, random_state=SEED, device="cpu", verbose=True,
    )


def vblr_parameters(n_features=N_FEATURES):
    p = int(n_features)
    return dict(
        tau=0.5, pi0_init=0.5, sigma0_sq_init=1.0, sigma_b_sq=100.0,
        empirical_bayes=True, prior_inclusion_prob=[0.5] * p,
        feature_prior_update="em", alpha_init=None, sigma0_sq_max=None,
        standardize=True, max_iter=5000, tol=1e-6, delta_c=1e-8,
        delta_h=1e-6, verbose=True, log_every=100,
    )


def dw_blr_parameters(n_features=N_FEATURES):
    p = int(n_features)
    return dict(
        auto_prior=True, optimization_mode="feature_em", prior_update_scope="outer",
        prior_inclusion_prob=[0.5] * p, pi0_init=0.5,
        n_subsamples=100, subsample_fraction=0.85, subsample_stratify=True,
        subsample_replace=False,
        random_state=SEED, pilot_gamma_w=0.2,
        pilot_warmup_sweeps=5, pilot_max_rounds=200, pilot_min_rounds=5,
        pilot_stable_rounds=5, pilot_outer_tol=1e-5, frequency_tau=0.5,
        prior_floor=0.1, frequency_mix=0.8, prior_clip=0.01,
        gamma_w=0.1, decorrelation_scale=1.0, covariance_mode="normalized",
        use_curvature_decorrelation=True,
        T_max=1000, tau=DWBLR_FINAL_PIP_THRESHOLD,
        warmup_sweeps=5, unprotected_rounds=0,
        outer_tol=1e-5, min_outer_iter=5, stable_rounds=5, stability_window=10,
        inner_max_iter=200, inner_tol=1e-6, warm_start_inner=True,
        weight_damping=0.8, lambda_damping=0.8, prior_damping=0.8, slab_damping=0.8,
        nu_start=3.0, nu_end=1.0, nu_schedule="geometric", nu_values=None,
        anneal_iters=20, lambda_floor=0.1, use_selective_protection=True,
        sigma0_sq_init=2.0, sigma0_sq_max=None, sigma_b_sq=100.0, alpha_init=None,
        prior_update_interval=1, prior_center=0.5, update_prior_unprotected=False,
        slab_update="outer_eb", delta_c=1e-8, delta_h=1e-6,
        attacker_max_iter=3500, attacker_tol=1e-6, attacker_initial_lr=1.0,
        attacker_backtrack_factor=0.5, attacker_max_backtracking=30,
        attacker_scale_step_by_n=True, smooth_protection_start=True,
        record_inner_trace=False, verbose=True, output_dir=None,
    )


def penalized_parameters(method):
    if method not in ("Lasso", "Elastic Net", "SCAD"):
        raise ValueError(f"Unknown penalized method: {method}")
    cfg = dict(
        family="binomial", lambda_=0.035, intercept=True,
        external_standardization=dict(fit_on="training", ddof=0),
        nonzero_threshold=1e-8,
    )
    if method == "SCAD":
        cfg.update(
            backend="R/ncvreg", package_version="3.16.0", penalty="SCAD",
            gamma=5.0, alpha=1.0, eps=1e-10, max_iter_path_total=1000000,
            crosscheck_eps=1e-8,
            path=dict(start=0.5, stop=0.001, log_points=100,
                      keep="lambda > 0.035", append_exact_lambda=0.035),
        )
    else:
        cfg.update(
            backend="R/glmnet", package_version="4.1-10",
            alpha=1.0 if method == "Lasso" else 0.5, standardize=False,
            thresh=1e-14, maxit=2000000, crosscheck_thresh=1e-12,
            glmnet_control=dict(fdev=0.0, devmax=1.0),
            path=dict(start=1.0, stop=0.035, log_points=80,
                      include_exact_endpoint=True),
        )
    return cfg


def all_parameters(n_train=N_TRAIN, n_features=N_FEATURES):
    return deepcopy({
        "Lasso": penalized_parameters("Lasso"),
        "Elastic Net": penalized_parameters("Elastic Net"),
        "SCAD": penalized_parameters("SCAD"),
        "SRDO": srdo_parameters(n_train),
        "SVI": svi_parameters(n_train),
        "DWR": dwr_parameters(),
        "VBLR": vblr_parameters(n_features),
        "DW-BLR": dw_blr_parameters(n_features),
    })


def topk_refit_parameters():
    return dict(
        sample_weights="ones", clinical_covariates=False,
        intercept_penalized=False, training_scaler_ddof=0, refit_on_test=False,
        lambda_by_method={m: {"Top3": 0.1, "Top5": 0.1, "Top10": 0.1} for m in METHODS},
        optimizer=dict(method="trust-exact", gtol=1e-9, maxiter=2000),
        gradient_inf_lt=1e-8,
        independent_check=dict(solver="sklearn/lbfgs", penalty="l2",
                               C="1/(n_train*lambda_l2)", tol=1e-12,
                               max_iter=20000, max_difference=1e-6),
    )
