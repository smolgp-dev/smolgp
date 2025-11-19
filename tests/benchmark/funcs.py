import tinygp
import smolgp
import jax.numpy as jnp

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
    gp_ss = smolgp.GaussianProcess(kernel, t_train, diag=yerr**2)
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
        kernel, t_train, diag=yerr**2, solver=smolgp.solvers.ParallelStateSpaceSolver
    )
    return gp_ss.log_probability(y_train)

#################### CONDITION ####################
def ss_cond(data, kernel):
    t_train, y_train, yerr = unpack_data(data)
    gp_ss = smolgp.GaussianProcess(kernel, t_train, diag=yerr**2)
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
        kernel, t_train, diag=yerr**2, solver=smolgp.solvers.ParallelStateSpaceSolver
    )
    llh, condGP_ss = gp_ss.condition(y_train)
    return jnp.array([condGP_ss.loc, condGP_ss.variance])

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



######################################## INTEGRATED DATA FUNCTIONS ########################################
#################### LIKELIHOOD ####################
def iss_llh(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_ss = smolgp.GaussianProcess(kernel, X_train, diag=yerr**2)
    return gp_ss.log_probability(y_train)

def igp_llh(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_gp = tinygp.GaussianProcess(kernel, X_train, diag=yerr**2)
    return gp_gp.log_probability(y_train)

def ipss_llh(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_ss = smolgp.GaussianProcess(
        kernel, X_train, diag=yerr**2, solver=smolgp.solvers.ParallelStateSpaceSolver
    )
    return gp_ss.log_probability(y_train)

#################### CONDITION ####################
def iss_cond(data, kernel):
    X_train, y_train, yerr = unpack_idata(data)
    gp_ss = smolgp.GaussianProcess(kernel, X_train, diag=yerr**2)
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
        kernel, X_train, diag=yerr**2, solver=smolgp.solvers.ParallelStateSpaceSolver
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
