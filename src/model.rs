//! Surrogate models: cheap approximations of an expensive function, fitted to its evaluations.
//!
//! [`gp`] has Gaussian process regression, the model of
//! [Bayesian optimization](crate::algorithm::bo), which can also be fitted and queried on its own.

pub mod gp;
