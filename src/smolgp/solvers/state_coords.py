from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
from tinygp.helpers import JAXArray


class StateCoords(eqx.Module):
    r"""The state-level bookkeeping shared by every solver, filter, smoother,
    and sampler in ``smolgp``.

    A state space model is solved by stepping through a sorted timeline of
    *states*. For an instantaneous kernel there is exactly one state per
    observation, but for an integrated (exposure-averaged) kernel each
    observation contributes *two* states: the start and end of its exposure
    window. Therefore, the number of states ``K`` and the number of observations
    ``N`` are different, and the mapping between them has to be carried
    around explicitly. Additionally, the integrated solver needs to know for
    a given state what observation it belongs to and whether that state was
    an exposure start or end, as these are handled differently.
    That mapping is what this object holds.

    Attributes:
        t_states: shape ``(K,)``. The sortable coordinate (e.g. time) of each
            state, in ascending order. Ties are broken so that exposure *ends*
            (``stateid==1``) precede exposure *starts* (``stateid==0``) at the
            same instant, via a ``jnp.lexsort((-stateid, t_states))``.
        instid: shape ``(N,)`` **per observation, not per state**. Which
            instrument recorded each observation. Index it through ``obsid``
            (or use :meth:`instid_per_state`) to get a state's instrument.
        obsid: shape ``(K,)``. Which observation ``0..N-1`` each state belongs
            to. For an integrated kernel, the start and end states of one
            exposure share an ``obsid``.
        stateid: shape ``(K,)``. ``0`` for an exposure-start state (where
            :meth:`~smolgp.kernels.base.StateSpaceModel.reset_matrix` is
            applied), ``1`` for an exposure-end state or a plain instantaneous
            observation (i.e. the states that carry data).

    For an instantaneous kernel the "integrated" fields degenerate to a trivial
    convention: ``K == N``, ``obsid == arange(N)``, ``stateid == 1``
    everywhere (every state carries data). A single instrument dataset likewise
    trivially has ``instid == 0`` everywhere.
    """

    t_states: JAXArray
    instid: JAXArray
    obsid: JAXArray
    stateid: JAXArray

    @classmethod
    def instantaneous(cls, t_states: JAXArray) -> StateCoords:
        """The degenerate ``StateCoords`` for an instantaneous kernel: one
        state per observation, all carrying data, all on one instrument."""
        K = t_states.shape[0]
        return cls(
            t_states=t_states,
            instid=jnp.zeros(K, dtype=int),
            obsid=jnp.arange(K, dtype=int),
            stateid=jnp.ones(K, dtype=int),
        )

    @property
    def num_states(self) -> int:
        """``K``, the number of states in the timeline."""
        return self.t_states.shape[0]

    @property
    def num_obs(self) -> int:
        """``N``, the number of observations."""
        return self.instid.shape[0]

    def instid_per_state(self) -> JAXArray:
        """``instid`` gathered to shape ``(K,)``: the instrument id of the
        observation each state belongs to."""
        return self.instid[self.obsid]
