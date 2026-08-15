import tinygp
import smolgp
import jax
import jax.numpy as jnp

SAMPLE_KEY = jax.random.PRNGKey(0)
SAMPLE_SHAPE = (10,)
# Prior draws have no measurement error; a small shared jitter keeps the
# dense Cholesky well conditioned and is applied identically to all three.
PRIOR_JITTER = 1e-6

def unpack_idata(data):
    t_train = data[0, :]
    y_train = data[1, :]
    yerr = data[2, :]
    texp = data[3, :]
    instid = data[4, :].astype(jnp.int64)
    X_train = (t_train, texp, instid)
    return X_train, y_train, yerr

def unpack_data(data):
    t_train = data[0, :]
    y_train = data[1, :]
    yerr = data[2, :]
    return t_train, y_train, yerr

######################################## INSTANTANEOUS DATA FUNCTIONS ########################################
#################### LIKELIHOOD ####################
def ss_llh(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_ss = smolgp.GaussianProcess(kernel, t_train, noise=yerr**2)
    return gp_ss.log_probability(y_train)

def qs_llh(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_qs = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    return gp_qs.log_probability(y_train)

def gp_llh(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_gp = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    return gp_gp.log_probability(y_train)

def pss_llh(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_ss = smolgp.GaussianProcess(
        kernel, t_train, noise=yerr**2, solver=smolgp.solvers.ParallelStateSpaceSolver
    )
    return gp_ss.log_probability(y_train)

def pqs_llh(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_qs = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2, 
                                   solver=tinygp.solvers.QuasisepSolver, parallel=True)
    return gp_qs.log_probability(y_train)

#################### CONDITION ####################
def ss_cond(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_ss = smolgp.GaussianProcess(kernel, t_train, noise=yerr**2)
    llh, condGP_ss = gp_ss.condition(y_train)
    return jnp.array([condGP_ss.loc, condGP_ss.variance])

def qs_cond(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_qs = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    llh, condGP_qs = gp_qs.condition(y_train)
    return jnp.array([condGP_qs.loc, condGP_qs.variance])

def gp_cond(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_gp = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2)
    llh, condGP_gp = gp_gp.condition(y_train)
    return jnp.array([condGP_gp.loc, condGP_gp.variance])

def pss_cond(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_ss = smolgp.GaussianProcess(
        kernel, t_train, noise=yerr**2, solver=smolgp.solvers.ParallelStateSpaceSolver
    )
    llh, condGP_ss = gp_ss.condition(y_train)
    return jnp.array([condGP_ss.loc, condGP_ss.variance])

def pqs_cond(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_qs = tinygp.GaussianProcess(kernel, t_train, diag=yerr**2, 
                                   solver=tinygp.solvers.QuasisepSolver, parallel=True)
    llh, condGP_qs = gp_qs.condition(y_train)
    return jnp.array([condGP_qs.loc, condGP_qs.variance])

#################### PREDICTION ####################
## TODO?: Only time the actual prediction part (use condGP here)
def ss_pred(t_test, gp_ss, y_train):
    mu, var = gp_ss.predict(t_test, y_train, return_var=True)
    return jnp.array([mu, var])

def qs_pred(t_test, gp_qs, y_train):
    mu, var = gp_qs.predict(y_train, t_test, return_var=True)
    return jnp.array([mu, var])

def gp_pred(t_test, gp_gp, y_train):
    mu, var = gp_gp.predict(y_train, t_test, return_var=True)
    return jnp.array([mu, var])

#################### SAMPLE ####################
#################### SAMPLE (PRIOR) ####################
# A prior draw has no training data, so the only size parameter is M, the
# number of coordinates the realization is drawn at. These take the sample
# coordinates directly rather than a (t, y, yerr) dataset.
def ss_sample_prior(t_sample, kernel):
    gp_ss = smolgp.GaussianProcess(kernel, t_sample, noise=PRIOR_JITTER)
    return gp_ss.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)

def qs_sample_prior(t_sample, kernel):
    gp_qs = tinygp.GaussianProcess(kernel, t_sample, diag=PRIOR_JITTER)
    return gp_qs.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)

def gp_sample_prior(t_sample, kernel):
    gp_gp = tinygp.GaussianProcess(kernel, t_sample, diag=PRIOR_JITTER)
    return gp_gp.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)

#################### SAMPLE (POSTERIOR) ####################
# Mirrors the predict benchmark exactly: condition on N training points, then
# draw at M = 100N test coordinates. Same signature as the *_pred funcs so
# run_pred_benchmark can drive these unchanged.
def ss_sample_post(t_test, gp_ss, y_train):
    _llh, condGP_ss = gp_ss.condition(y_train)
    return condGP_ss.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE, X_test=t_test)

def qs_sample_post(t_test, gp_qs, y_train):
    _llh, condGP_qs = gp_qs.condition(y_train, t_test)
    return condGP_qs.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)

def gp_sample_post(t_test, gp_gp, y_train):
    _llh, condGP_gp = gp_gp.condition(y_train, t_test)
    return condGP_gp.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)



######################################## INTEGRATED DATA FUNCTIONS ########################################
#################### LIKELIHOOD ####################
def iss_llh(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_ss = smolgp.GaussianProcess(kernel, X_train, noise=yerr**2)
    return gp_ss.log_probability(y_train)

def igp_llh(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_gp = tinygp.GaussianProcess(kernel, X_train, diag=yerr**2)
    return gp_gp.log_probability(y_train)

def ipss_llh(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_ss = smolgp.GaussianProcess(
        kernel, X_train, noise=yerr**2, solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver
    )
    return gp_ss.log_probability(y_train)

#################### CONDITION ####################
def iss_cond(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_ss = smolgp.GaussianProcess(kernel, X_train, noise=yerr**2)
    llh, condGP_ss = gp_ss.condition(y_train)
    return jnp.array([condGP_ss.loc, condGP_ss.variance])

def igp_cond(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_gp = tinygp.GaussianProcess(kernel, X_train, diag=yerr**2)
    llh, condGP_gp = gp_gp.condition(y_train)
    return jnp.array([condGP_gp.loc, condGP_gp.variance])

def ipss_cond(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_ss = smolgp.GaussianProcess(
        kernel, X_train, noise=yerr**2, solver=smolgp.solvers.ParallelIntegratedStateSpaceSolver
    )
    llh, condGP_ss = gp_ss.condition(y_train)
    return jnp.array([condGP_ss.loc, condGP_ss.variance])

#################### PREDICTION ####################
def iss_pred(t_test, gp_ss, y_train):
    X_test = (t_test, jnp.zeros_like(t_test), jnp.zeros_like(t_test).astype(int))
    mu, var = gp_ss.predict(X_test, y_train, return_var=True)
    return jnp.array([mu, var])

def igp_pred(t_test, gp_gp, y_train):
    X_test = (t_test, jnp.zeros_like(t_test), jnp.zeros_like(t_test).astype(int))
    mu, var = gp_gp.predict(y_train, X_test, return_var=True)
    return jnp.array([mu, var])

#################### SAMPLE ####################
#################### SAMPLE (PRIOR), INTEGRATED ####################
# The sample coordinates arrive as a full (t, texp, instid) tuple, so the draw
# represents what an exposure-integrating instrument would have recorded.
def iss_sample_prior(X_sample, kernel):
    gp_ss = smolgp.GaussianProcess(kernel, X_sample, noise=PRIOR_JITTER)
    return gp_ss.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)

def igp_sample_prior(X_sample, kernel):
    gp_gp = tinygp.GaussianProcess(kernel, X_sample, diag=PRIOR_JITTER)
    return gp_gp.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)

#################### SAMPLE (POSTERIOR), INTEGRATED ####################
def iss_sample_post(t_test, gp_ss, y_train):
    X_test = (t_test, jnp.zeros_like(t_test), jnp.zeros_like(t_test).astype(int))
    _llh, condGP_ss = gp_ss.condition(y_train)
    return condGP_ss.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE, X_test=X_test)

def igp_sample_post(t_test, gp_gp, y_train):
    X_test = (t_test, jnp.zeros_like(t_test), jnp.zeros_like(t_test).astype(int))
    _llh, condGP_gp = gp_gp.condition(y_train, X_test)
    return condGP_gp.sample(SAMPLE_KEY, shape=SAMPLE_SHAPE)
