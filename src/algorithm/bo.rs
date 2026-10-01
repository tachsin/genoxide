//! Bayesian optimization: a surrogate model of an expensive function, and acquisition functions
//! that pick where to evaluate it next.
//!
//! [`acquisition`] has the acquisition functions, from a model's predictive mean and standard
//! deviation at a point: they work with any surrogate model that gives those.

pub mod acquisition;
