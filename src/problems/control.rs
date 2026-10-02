//! Control tasks: poles to balance on a cart, driven by a [`Policy`] such as a neural network of
//! [`nn`](crate::nn), for neuroevolution.
//!
//! A task is a simulated physical system and a goal. At each step, the policy observes the
//! system's state and chooses an action, a force on the cart; the task moves the system on by
//! 0.02 s, and the episode ends when the goal can't be met any more (a pole has fallen, or the cart
//! has left the track). A fitness function runs an episode with the policy that a genome encodes
//! and scores it, e.g. by the steps it lasted.
//!
//! | Task | State | Observed | Solved |
//! |---|---|---|---|
//! | [`CartPole`] | cart, one pole | all 4 variables | balanced for 100,000 steps |
//! | [`DoublePole::new`] | cart, a long and a short pole | all 6 variables | balanced for 100,000 steps |
//! | [`DoublePole::without_velocities`] | the same | the cart's position and the poles' angles: 3 | balanced for 100,000 steps, and for 1000 steps from at least 200 of 625 other starts |
//!
//! ```
//! use genoxide::nn::{Activation, Mlp};
//! use genoxide::prelude::*;
//! use genoxide::problems::control::{CartPole, SUCCESS_STEPS};
//!
//! // a 4-2-1 network without biases balances the pole
//! let mlp = Mlp::new([4, 2, 1], Activation::Tanh)?
//!     .output_activation(Activation::Tanh)
//!     .bias(false);
//! let task = CartPole::new();
//! let steps = |weights: &Reals| -> Option<f64> {
//!     let mut network = mlp.with(weights).ok()?;
//!     Some(f64::from(task.run(&mut network, SUCCESS_STEPS)))
//! };
//! let cmaes = Cmaes::builder(mlp.representation(-1.0..=1.0)?).seed(1).build()?;
//! let outcome = Engine::new(cmaes, steps)
//!     .stop_when(Stop::target(f64::from(SUCCESS_STEPS)).or(Stop::evaluations(10_000)))
//!     .run()?;
//! assert_eq!(outcome.stop_reason(), StopReason::Target);
//! # Ok::<(), genoxide::Error>(())
//! ```
//!
//! # The system
//!
//! A cart of 1 kg on a track from −2.4 m to 2.4 m carries poles hinged to it, which move in the
//! plane of the track. The state is the cart's position `x` and velocity `ẋ`, and each pole's angle
//! from the vertical `θᵢ` and angular velocity `θ̇ᵢ`, in m, m/s, rad and rad/s; the positive
//! directions of `x` and `θᵢ` are the same, so a pole leaning to positive angles falls further
//! that way unless the cart moves under it. The action is a number in [−1, 1] (clamped), for a
//! force of up to 10 N on the cart: forces smaller than 10/256 N are raised to that size, keeping
//! their sign (a zero action pushes to positive `x`), so that a policy can't hold the system in its
//! unstable equilibrium by doing nothing (Wieland 1991, as described by Gomez et al. 2008).
//!
//! | Setting | Cart-pole | Double pole |
//! |---|---|---|
//! | Pole masses | 0.1 kg | 0.1 kg and 0.01 kg |
//! | Pole half-lengths | 0.5 m | 0.5 m and 0.05 m |
//! | Friction of the cart on the track, `μc` | 0.0005 | 0.0005 |
//! | Friction of each pole's hinge, `μp` | 0.000002 N m s | 0.000002 N m s |
//! | Gravity | 9.8 m/s² | 9.8 m/s² |
//! | Failure: a pole beyond, or the cart beyond | 12°, 2.4 m | 36°, 2.4 m |
//! | Initial state | `θ` = 4°, the rest 0 | `θ₁` = 4°, the rest 0 |
//!
//! The equations of motion are Wieland's (1991) for any number of poles, as given in Igel (2003)
//! and Gomez et al. (2008), with Florian's (2007) corrections: gravity positive, and the cart's
//! friction `μc N sgn(N ẋ)` proportional to the normal force `N` of the track on the cart, which
//! the poles' motion changes, instead of `μc sgn(ẋ)`, whose dimension isn't a force. With
//! `aᵢ = g sin θᵢ − μp θ̇ᵢ / (mᵢ lᵢ)`, `F̃ᵢ = mᵢ lᵢ θ̇ᵢ² sin θᵢ − ¾ mᵢ aᵢ cos θᵢ` (the pole's
//! effective force on the cart) and `m̃ᵢ = mᵢ (1 − ¾ cos² θᵢ)` (its effective mass):
//!
//! ```text
//! N  = (M + Σ mᵢ) g − Σ mᵢ lᵢ (θ̈ᵢ sin θᵢ + θ̇ᵢ² cos θᵢ)
//! ẍ  = (F − μc N sgn(N ẋ) + Σ F̃ᵢ) / (M + Σ m̃ᵢ)
//! θ̈ᵢ = 3 / (4 lᵢ) (aᵢ − ẍ cos θᵢ)
//! ```
//!
//! `N` depends on `ẍ` through `θ̈ᵢ`, linearly, so the three are solved together, with the sign of
//! `N` of the previous evaluation (positive at the start), and again with the other sign if `N`
//! changes sign, as Florian (2007) describes. With one pole and without friction these are the
//! frictionless equations of Florian (2007) and Barto et al. (1983), with `g` positive.
//!
//! Each step holds the force for 0.02 s, in two steps of fourth-order Runge-Kutta integration of
//! 0.01 s (Gomez et al. 2008; Stanley and Miikkulainen 2002), with [`math::sin_cos`], so that an
//! episode is the same bits on every platform.
//!
//! # Observations
//!
//! A policy observes the state scaled to about [−1, 1]: the cart's position divided by 2.4 m, the
//! angles by the failure angle, and the velocities (m/s and rad/s) by 2, in the order `x`, `ẋ`,
//! `θ₁`, `θ̇₁`, `θ₂`, `θ̇₂`; without velocities, `x`, `θ₁`, `θ₂`. Gomez et al. (2008) scale the
//! inputs to [−1, 1] without giving the ranges; these are genoxide's. [`state`](CartPole::state)
//! gives the state unscaled.
//!
//! # Solving the tasks
//!
//! A task is solved, in all the papers above, when a policy balances the poles for
//! [`SUCCESS_STEPS`] steps, 100,000, "over 30 minutes of simulated time" (Gomez et al. 2008).
//!
//! Without velocities, a policy must compute them, which needs a recurrent network, such as an
//! [`Elman`](crate::nn::Elman) network. Gruau, Whitley and Pyeatt (1996) made this task harder to
//! pass by chance in two ways, which Stanley and Miikkulainen (2002), Igel (2003) and Gomez et al.
//! (2008) kept:
//!
//! - The fitness is [`damping_fitness`](DoublePole::damping_fitness) over 1000 steps, which
//!   rewards policies that bring the cart and the long pole to rest over those that keep the poles
//!   up by jiggling the cart.
//! - A policy that balances the poles for 100,000 steps must also pass the
//!   [`generalization`](DoublePole::generalization) test: balance them for 1000 steps from at
//!   least 200 of 625 starts spread over the state space.
//!
//! [`DoublePole::solved`] applies both criteria that apply to the task.
//!
//! # References
//!
//! - Barto, A. G., Sutton, R. S. and Anderson, C. W. (1983). Neuronlike adaptive elements that can
//!   solve difficult learning control problems. *IEEE Transactions on Systems, Man, and
//!   Cybernetics* 13(5): 834-846. doi:10.1109/TSMC.1983.6313077 (the cart-pole system, its
//!   masses, lengths and friction coefficients)
//! - Wieland, A. P. (1991). Evolving neural network controllers for unstable systems. IJCNN 1991,
//!   vol. 2: 667-673. doi:10.1109/IJCNN.1991.155416 (poles side by side on one cart, and their
//!   equations of motion)
//! - Gruau, F., Whitley, D. and Pyeatt, L. (1996). A comparison between cellular encoding and
//!   direct encoding for genetic neural networks. Genetic Programming 1996: 81-89 (the double
//!   pole without velocities, its damping fitness and its generalization test)
//! - Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting
//!   topologies. *Evolutionary Computation* 10(2): 99-127. doi:10.1162/106365602320169811
//!   (sections 4.3.1 to 4.3.3)
//! - Igel, C. (2003). Neuroevolution for reinforcement learning using evolution strategies.
//!   CEC 2003: 2588-2595. doi:10.1109/CEC.2003.1299414 (the equations for any number of poles,
//!   and the 625 starts of the generalization test)
//! - Florian, R. V. (2007). Correct equations for the dynamics of the cart-pole system. Technical
//!   report, Center for Cognitive and Neural Studies (Coneural), Romania
//! - Gomez, F., Schmidhuber, J. and Miikkulainen, R. (2008). Accelerated neural evolution through
//!   cooperatively coevolved synapses. *Journal of Machine Learning Research* 9: 937-965 (section
//!   5.3, the settings above; appendix A)

use crate::math;
use crate::nn::{ElmanNetwork, MlpNetwork};

/// Steps balanced to solve a task: 100,000, of 0.02 s each.
pub const SUCCESS_STEPS: u32 = 100_000;

/// The steps of an episode of [`DoublePole::damping_fitness`], and of each start of
/// [`DoublePole::generalization`]: 1000.
pub const DAMPING_STEPS: u32 = 1000;

/// The starts of [`DoublePole::generalization`] that a policy must balance for
/// [`DAMPING_STEPS`] steps: 200 of the 625.
pub const GENERALIZATION_THRESHOLD: u32 = 200;

// the seconds of a step of the integration, and the integration steps per step of a task
const TAU: f64 = 0.01;
const SUBSTEPS: usize = 2;
const GRAVITY: f64 = 9.8;
const CART_MASS: f64 = 1.0;
const CART_FRICTION: f64 = 0.0005;
const POLE_FRICTION: f64 = 0.000_002;
const FORCE: f64 = 10.0;
const MIN_FORCE: f64 = FORCE / 256.0;
const TRACK: f64 = 2.4;
// the scale of the velocities in an observation
const VELOCITY_SCALE: f64 = 2.0;

fn degrees(angle: f64) -> f64 {
    angle * std::f64::consts::PI / 180.0
}

/// How a controller chooses actions: a [`Policy`] observes a task's state, scaled, and writes its
/// action.
///
/// Implemented by the networks of [`nn`](crate::nn) (their outputs are the action) and by
/// closures `FnMut(&[f64], &mut [f64])`:
///
/// ```
/// use genoxide::problems::control::CartPole;
///
/// // push the cart towards where the pole leans, harder the faster it falls
/// let mut push = |observation: &[f64], action: &mut [f64]| {
///     let [x, velocity, angle, angular_velocity] = observation else { unreachable!() };
///     action[0] = 2.0 * angle + angular_velocity + 0.1 * x + 0.2 * velocity;
/// };
/// assert_eq!(CartPole::new().run(&mut push, 1000), 1000);
/// ```
pub trait Policy {
    /// Writes the action for `observation` into `action`: one force for the tasks of this module.
    fn act(&mut self, observation: &[f64], action: &mut [f64]);

    /// Forgets the previous episode, e.g. a recurrent network's context. Nothing by default.
    fn reset(&mut self) {}
}

impl<F: FnMut(&[f64], &mut [f64])> Policy for F {
    fn act(&mut self, observation: &[f64], action: &mut [f64]) {
        self(observation, action);
    }
}

impl Policy for MlpNetwork<'_> {
    fn act(&mut self, observation: &[f64], action: &mut [f64]) {
        self.forward(observation, action);
    }
}

/// A NEAT network's outputs are the actions: with NEAT's sigmoid, in (0, 1), only pushes one way;
/// map them to [−1, 1] in a closure, or use a symmetric activation such as tanh.
impl Policy for crate::neat::FeedForward {
    fn act(&mut self, observation: &[f64], action: &mut [f64]) {
        self.activate(observation, action);
    }
}

/// As for [`FeedForward`](crate::neat::FeedForward); [`reset`](Policy::reset) clears the
/// network's state between episodes.
impl Policy for crate::neat::Recurrent {
    fn act(&mut self, observation: &[f64], action: &mut [f64]) {
        self.activate(observation, action);
    }

    fn reset(&mut self) {
        crate::neat::Recurrent::reset(self);
    }
}

impl Policy for ElmanNetwork<'_> {
    fn act(&mut self, observation: &[f64], action: &mut [f64]) {
        self.forward(observation, action);
    }

    fn reset(&mut self) {
        ElmanNetwork::reset(self);
    }
}

// the state of a cart with `N` poles: the cart's position and velocity, the poles' angles and
// angular velocities; also the derivative of a state
#[derive(Clone, Copy, Debug, PartialEq)]
struct State<const N: usize> {
    x: f64,
    velocity: f64,
    angles: [f64; N],
    angular_velocities: [f64; N],
}

impl<const N: usize> State<N> {
    const ZERO: Self = Self {
        x: 0.0,
        velocity: 0.0,
        angles: [0.0; N],
        angular_velocities: [0.0; N],
    };

    // self + h k
    fn plus(&self, k: &Self, h: f64) -> Self {
        Self {
            x: self.x + h * k.x,
            velocity: self.velocity + h * k.velocity,
            angles: std::array::from_fn(|i| self.angles[i] + h * k.angles[i]),
            angular_velocities: std::array::from_fn(|i| {
                self.angular_velocities[i] + h * k.angular_velocities[i]
            }),
        }
    }

    // self + h / 6 (k1 + 2 k2 + 2 k3 + k4), fourth-order Runge-Kutta's step
    fn runge_kutta(&self, [k1, k2, k3, k4]: [&Self; 4], h: f64) -> Self {
        let step = |value: f64, a: f64, b: f64, c: f64, d: f64| {
            value + h / 6.0 * (a + 2.0 * b + 2.0 * c + d)
        };
        Self {
            x: step(self.x, k1.x, k2.x, k3.x, k4.x),
            velocity: step(
                self.velocity,
                k1.velocity,
                k2.velocity,
                k3.velocity,
                k4.velocity,
            ),
            angles: std::array::from_fn(|i| {
                step(
                    self.angles[i],
                    k1.angles[i],
                    k2.angles[i],
                    k3.angles[i],
                    k4.angles[i],
                )
            }),
            angular_velocities: std::array::from_fn(|i| {
                let [a, b, c, d] = [k1, k2, k3, k4].map(|k| k.angular_velocities[i]);
                step(self.angular_velocities[i], a, b, c, d)
            }),
        }
    }

    // [x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂, ...]
    fn write(&self, out: &mut [f64]) {
        out[0] = self.x;
        out[1] = self.velocity;
        for i in 0..N {
            out[2 + 2 * i] = self.angles[i];
            out[3 + 2 * i] = self.angular_velocities[i];
        }
    }

    fn read(values: &[f64]) -> Self {
        Self {
            x: values[0],
            velocity: values[1],
            angles: std::array::from_fn(|i| values[2 + 2 * i]),
            angular_velocities: std::array::from_fn(|i| values[3 + 2 * i]),
        }
    }
}

// the sign of `x`, 0 for 0
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

// the force for an action: in [−10, 10] N, at least 10/256 N in size; NaN for NaN
fn force(action: f64) -> f64 {
    let force = FORCE * action.clamp(-1.0, 1.0);
    if force.abs() < MIN_FORCE {
        if force < 0.0 { -MIN_FORCE } else { MIN_FORCE }
    } else {
        force
    }
}

// a cart with `N` poles: its parameters, its initial and current states, and the sign of the
// normal force of the last evaluation of the equations
#[derive(Clone, Copy, Debug, PartialEq)]
struct Cart<const N: usize> {
    masses: [f64; N],
    half_lengths: [f64; N],
    cart_friction: f64,
    pole_friction: f64,
    failure_angle: f64,
    start: State<N>,
    state: State<N>,
    normal_sign: f64,
}

impl<const N: usize> Cart<N> {
    fn new(masses: [f64; N], half_lengths: [f64; N], failure_angle: f64) -> Self {
        let mut start = State::ZERO;
        start.angles[0] = degrees(4.0);
        Self {
            masses,
            half_lengths,
            cart_friction: CART_FRICTION,
            pole_friction: POLE_FRICTION,
            failure_angle,
            start,
            state: start,
            normal_sign: 1.0,
        }
    }

    // the derivative of `state` under `force`, and the normal force of the track on the cart;
    // `normal_sign` is the sign of the normal force assumed, updated if it changed
    fn derivative(&self, state: &State<N>, force: f64, normal_sign: &mut f64) -> (State<N>, f64) {
        let mut total_mass = CART_MASS;
        for mass in self.masses {
            total_mass += mass;
        }
        // with N = c0 + c1 ẍ: ẍ (M + Σ m̃ᵢ + μc s c1) = F − μc s c0 + Σ F̃ᵢ, where s = sgn(N ẋ)
        let mut c0 = total_mass * GRAVITY;
        let mut c1 = 0.0;
        let mut effective_force = 0.0;
        let mut effective_mass = CART_MASS;
        let mut sines = [0.0; N];
        let mut cosines = [0.0; N];
        let mut a = [0.0; N];
        for i in 0..N {
            let (m, l) = (self.masses[i], self.half_lengths[i]);
            let omega = state.angular_velocities[i];
            let (sin, cos) = math::sin_cos(state.angles[i]);
            (sines[i], cosines[i]) = (sin, cos);
            a[i] = GRAVITY * sin - self.pole_friction * omega / (m * l);
            effective_force += m * l * omega * omega * sin - 0.75 * m * a[i] * cos;
            effective_mass += m * (1.0 - 0.75 * cos * cos);
            c0 -= m * (0.75 * a[i] * sin + l * omega * omega * cos);
            c1 += 0.75 * m * cos * sin;
        }
        let acceleration = |normal_sign: f64| {
            let s = self.cart_friction * normal_sign * sign(state.velocity);
            (force - s * c0 + effective_force) / (effective_mass + s * c1)
        };
        let mut x_acceleration = acceleration(*normal_sign);
        let mut normal = c0 + c1 * x_acceleration;
        if sign(normal) == -*normal_sign {
            *normal_sign = -*normal_sign;
            x_acceleration = acceleration(*normal_sign);
            normal = c0 + c1 * x_acceleration;
        }
        let derivative = State {
            x: state.velocity,
            velocity: x_acceleration,
            angles: state.angular_velocities,
            angular_velocities: std::array::from_fn(|i| {
                0.75 / self.half_lengths[i] * (a[i] - x_acceleration * cosines[i])
            }),
        };
        (derivative, normal)
    }

    // a step of 0.02 s under the force of `action`; whether the poles are still up and the cart
    // on the track
    fn step(&mut self, action: f64) -> bool {
        self.advance(force(action));
        self.balanced()
    }

    // 0.02 s under `force`, in newtons
    fn advance(&mut self, force: f64) {
        let mut sign = self.normal_sign;
        let mut state = self.state;
        for _ in 0..SUBSTEPS {
            let (k1, _) = self.derivative(&state, force, &mut sign);
            let (k2, _) = self.derivative(&state.plus(&k1, TAU / 2.0), force, &mut sign);
            let (k3, _) = self.derivative(&state.plus(&k2, TAU / 2.0), force, &mut sign);
            let (k4, _) = self.derivative(&state.plus(&k3, TAU), force, &mut sign);
            state = state.runge_kutta([&k1, &k2, &k3, &k4], TAU);
        }
        self.state = state;
        self.normal_sign = sign;
    }

    // whether the poles are within the failure angle and the cart on the track (false for NaN)
    fn balanced(&self) -> bool {
        let mut angles = self.state.angles.iter();
        self.state.x.abs() <= TRACK && angles.all(|a| a.abs() <= self.failure_angle)
    }

    fn reset(&mut self) {
        self.state = self.start;
        self.normal_sign = 1.0;
    }

    fn set_start(&mut self, start: State<N>) {
        self.start = start;
        self.reset();
    }

    // the scaled state, with or without the velocities
    fn observe(&self, velocities: bool, out: &mut [f64]) {
        let state = &self.state;
        if velocities {
            out[0] = state.x / TRACK;
            out[1] = state.velocity / VELOCITY_SCALE;
            for i in 0..N {
                out[2 + 2 * i] = state.angles[i] / self.failure_angle;
                out[3 + 2 * i] = state.angular_velocities[i] / VELOCITY_SCALE;
            }
        } else {
            out[0] = state.x / TRACK;
            for i in 0..N {
                out[1 + i] = state.angles[i] / self.failure_angle;
            }
        }
    }

    // an episode of at most `steps` steps from the start, on a copy; the steps balanced, and
    // `each` called with the state after each step balanced
    fn run<P: Policy + ?Sized>(
        &self,
        velocities: bool,
        policy: &mut P,
        steps: u32,
        mut each: impl FnMut(&State<N>),
    ) -> u32 {
        let mut cart = *self;
        cart.reset();
        policy.reset();
        let mut observation = [0.0; 6];
        let observed = if velocities { 2 + 2 * N } else { 1 + N };
        let mut action = [0.0];
        for step in 0..steps {
            cart.observe(velocities, &mut observation[..observed]);
            policy.act(&observation[..observed], &mut action);
            if !cart.step(action[0]) {
                return step;
            }
            each(&cart.state);
        }
        steps
    }
}

/// The cart-pole system of Barto, Sutton and Anderson (1983): one pole on a cart, balanced by
/// pushing the cart, with Florian's (2007) corrected equations and Gomez et al.'s (2008) settings.
/// See the [module](self) for the system and the references.
///
/// The state is `[x, ẋ, θ, θ̇]`, in m, m/s, rad and rad/s; the pole fails beyond 12°, the cart
/// beyond 2.4 m. The policy observes all four, scaled, and outputs one action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CartPole {
    cart: Cart<1>,
}

impl Default for CartPole {
    fn default() -> Self {
        Self::new()
    }
}

impl CartPole {
    /// The task, from its initial state: the pole at 4°, the rest 0.
    pub fn new() -> Self {
        Self {
            cart: Cart::new([0.1], [0.5], degrees(12.0)),
        }
    }

    /// The task from `state`, `[x, ẋ, θ, θ̇]`, instead of its initial state.
    #[must_use]
    pub fn with_state(mut self, state: [f64; 4]) -> Self {
        self.cart.set_start(State::read(&state));
        self
    }

    /// The task without friction, on the track or at the hinge: the system then conserves its
    /// energy and its horizontal momentum when no force acts on it.
    #[must_use]
    pub fn without_friction(mut self) -> Self {
        self.cart.cart_friction = 0.0;
        self.cart.pole_friction = 0.0;
        self
    }

    /// The current state, `[x, ẋ, θ, θ̇]`.
    pub fn state(&self) -> [f64; 4] {
        let mut state = [0.0; 4];
        self.cart.state.write(&mut state);
        state
    }

    /// Writes the policy's observation of the current state into `observation`: the four
    /// variables, scaled (see the [module](self)).
    ///
    /// # Panics
    ///
    /// If `observation` is shorter than 4.
    pub fn observe(&self, observation: &mut [f64]) {
        self.cart.observe(true, observation);
    }

    /// A step of 0.02 s with the force of `action` (in [−1, 1], for up to 10 N); whether the pole
    /// is still within 12° and the cart on the track.
    pub fn step(&mut self, action: f64) -> bool {
        self.cart.step(action)
    }

    /// Whether the pole is within 12° and the cart on the track.
    pub fn balanced(&self) -> bool {
        self.cart.balanced()
    }

    /// Goes back to the initial state.
    pub fn reset(&mut self) {
        self.cart.reset();
    }

    /// Runs an episode of at most `steps` steps from the initial state, on a copy of the task,
    /// after resetting `policy`: the steps before the pole fell or the cart left the track.
    pub fn run<P: Policy + ?Sized>(&self, policy: &mut P, steps: u32) -> u32 {
        self.cart.run(true, policy, steps, |_| ())
    }

    /// Whether `policy` solves the task: balances the pole for [`SUCCESS_STEPS`] steps.
    pub fn solved<P: Policy + ?Sized>(&self, policy: &mut P) -> bool {
        self.run(policy, SUCCESS_STEPS) == SUCCESS_STEPS
    }
}

/// Two poles side by side on a cart (Wieland 1991), a long one and a short one, balanced
/// together by pushing the cart; with Florian's (2007) corrected equations and Gomez et al.'s
/// (2008) settings. See the [module](self) for the system and the references.
///
/// The state is `[x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]`, the long pole first, in m, m/s, rad and rad/s; the
/// poles fail beyond 36°, the cart beyond 2.4 m. The policy observes the six variables, scaled
/// ([`new`](DoublePole::new)), or only `x`, `θ₁` and `θ₂`
/// ([`without_velocities`](DoublePole::without_velocities)), and outputs one action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoublePole {
    cart: Cart<2>,
    velocities: bool,
}

impl Default for DoublePole {
    fn default() -> Self {
        Self::new()
    }
}

impl DoublePole {
    /// The task with velocities, from its initial state: the long pole at 4°, the rest 0.
    pub fn new() -> Self {
        Self {
            cart: Cart::new([0.1, 0.01], [0.5, 0.05], degrees(36.0)),
            velocities: true,
        }
    }

    /// The task without velocities: the policy observes `x`, `θ₁` and `θ₂` only.
    pub fn without_velocities() -> Self {
        Self {
            velocities: false,
            ..Self::new()
        }
    }

    /// Whether the policy observes the velocities.
    pub fn has_velocities(&self) -> bool {
        self.velocities
    }

    /// The number of variables the policy observes: 6, or 3 without velocities.
    pub fn observations(&self) -> usize {
        if self.velocities { 6 } else { 3 }
    }

    /// The task from `state`, `[x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]`, instead of its initial state.
    #[must_use]
    pub fn with_state(mut self, state: [f64; 6]) -> Self {
        self.cart.set_start(State::read(&state));
        self
    }

    /// The task without friction, on the track or at the hinges: the system then conserves its
    /// energy and its horizontal momentum when no force acts on it.
    #[must_use]
    pub fn without_friction(mut self) -> Self {
        self.cart.cart_friction = 0.0;
        self.cart.pole_friction = 0.0;
        self
    }

    /// The current state, `[x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]`.
    pub fn state(&self) -> [f64; 6] {
        let mut state = [0.0; 6];
        self.cart.state.write(&mut state);
        state
    }

    /// Writes the policy's observation of the current state into `observation`: the six
    /// variables, or `x`, `θ₁` and `θ₂` without velocities, scaled (see the [module](self)).
    ///
    /// # Panics
    ///
    /// If `observation` is shorter than [`observations`](DoublePole::observations).
    pub fn observe(&self, observation: &mut [f64]) {
        self.cart.observe(self.velocities, observation);
    }

    /// A step of 0.02 s with the force of `action` (in [−1, 1], for up to 10 N); whether both
    /// poles are still within 36° and the cart on the track.
    pub fn step(&mut self, action: f64) -> bool {
        self.cart.step(action)
    }

    /// Whether both poles are within 36° and the cart on the track.
    pub fn balanced(&self) -> bool {
        self.cart.balanced()
    }

    /// Goes back to the initial state.
    pub fn reset(&mut self) {
        self.cart.reset();
    }

    /// Runs an episode of at most `steps` steps from the initial state, on a copy of the task,
    /// after resetting `policy`: the steps before a pole fell or the cart left the track.
    pub fn run<P: Policy + ?Sized>(&self, policy: &mut P, steps: u32) -> u32 {
        self.cart.run(self.velocities, policy, steps, |_| ())
    }

    /// Gruau, Whitley and Pyeatt's (1996) fitness, to maximize, of an episode of
    /// [`DAMPING_STEPS`] steps from the initial state: `0.1 f₁ + 0.9 f₂`, where `f₁ = t / 1000`
    /// for the `t` steps balanced, and `f₂ = 0.75 / Σ (|x| + |ẋ| + |θ₁| + |θ̇₁|)` over the states
    /// after the last 100 of them (m, m/s, rad and rad/s), or 0 for `t < 100`.
    ///
    /// `f₂` rewards a policy that brings the cart and the long pole to rest, and penalizes one
    /// that keeps the poles up by moving the cart back and forth, without computing the velocities
    /// that the task without velocities hides. As written by Stanley and Miikkulainen (2002) and
    /// Gomez et al. (2008), the sum runs from step `t − 100` to `t`, 101 states; Igel (2003) sums
    /// 100, which genoxide does: the last 100 steps.
    pub fn damping_fitness<P: Policy + ?Sized>(&self, policy: &mut P) -> f64 {
        const WINDOW: usize = 100;
        let mut offsets = [0.0; WINDOW];
        let mut steps = 0;
        let balanced = self
            .cart
            .run(self.velocities, policy, DAMPING_STEPS, |state| {
                let offset = state.x.abs()
                    + state.velocity.abs()
                    + state.angles[0].abs()
                    + state.angular_velocities[0].abs();
                offsets[steps % WINDOW] = offset;
                steps += 1;
            });
        let f1 = f64::from(balanced) / f64::from(DAMPING_STEPS);
        let f2 = if steps < WINDOW {
            0.0
        } else {
            // the last 100 in the order they came
            let oldest = steps % WINDOW;
            let mut sum = 0.0;
            for k in 0..WINDOW {
                sum += offsets[(oldest + k) % WINDOW];
            }
            0.75 / sum
        };
        0.1 * f1 + 0.9 * f2
    }

    /// The generalization test of Gruau, Whitley and Pyeatt (1996): of 625 starts, the number from
    /// which `policy` balances the poles for [`DAMPING_STEPS`] steps, reset before each.
    ///
    /// The starts give `x`, `ẋ`, `θ₁` and `θ̇₁` each of the values 0.05, 0.25, 0.5, 0.75 and 0.95
    /// of their ranges, ±2.16 m, ±1.35 m/s, ±3.6° and ±8.6°/s (5⁴ = 625), with the short pole at
    /// rest upright, as given by Igel (2003) and Stanley and Miikkulainen (2002). A policy passes
    /// with at least [`GENERALIZATION_THRESHOLD`], 200.
    pub fn generalization<P: Policy + ?Sized>(&self, policy: &mut P) -> u32 {
        let mut passed = 0;
        for start in Self::generalization_starts() {
            let task = self.with_state(start);
            if task.run(policy, DAMPING_STEPS) == DAMPING_STEPS {
                passed += 1;
            }
        }
        passed
    }

    /// The 625 starts of [`generalization`](DoublePole::generalization), `[x, ẋ, θ₁, θ̇₁, θ₂, θ̇₂]`,
    /// with `x` changing slowest.
    pub fn generalization_starts() -> impl Iterator<Item = [f64; 6]> {
        const FRACTIONS: [f64; 5] = [0.05, 0.25, 0.5, 0.75, 0.95];
        let value = |k: f64, range: f64| k * 2.0 * range - range;
        FRACTIONS.into_iter().flat_map(move |k1| {
            FRACTIONS.into_iter().flat_map(move |k2| {
                FRACTIONS.into_iter().flat_map(move |k3| {
                    FRACTIONS.into_iter().map(move |k4| {
                        [
                            value(k1, 2.16),
                            value(k2, 1.35),
                            value(k3, degrees(3.6)),
                            value(k4, degrees(8.6)),
                            0.0,
                            0.0,
                        ]
                    })
                })
            })
        })
    }

    /// Whether `policy` solves the task: balances the poles for [`SUCCESS_STEPS`] steps and,
    /// without velocities, also passes the [`generalization`](DoublePole::generalization) test,
    /// the criteria of Gruau, Whitley and Pyeatt (1996).
    pub fn solved<P: Policy + ?Sized>(&self, policy: &mut P) -> bool {
        self.run(policy, SUCCESS_STEPS) == SUCCESS_STEPS
            && (self.velocities || self.generalization(policy) >= GENERALIZATION_THRESHOLD)
    }
}

#[cfg(test)]
#[expect(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use crate::StreamRng;
    use crate::nn::{Activation, Elman, Mlp};
    use rand::RngExt;

    fn close(a: f64, b: f64, tolerance: f64) -> bool {
        (a - b).abs() <= tolerance * (1.0 + b.abs())
    }

    fn random_state<const N: usize>(rng: &mut StreamRng) -> State<N> {
        State {
            x: rng.random_range(-2.0..2.0),
            velocity: rng.random_range(-2.0..2.0),
            angles: std::array::from_fn(|_| rng.random_range(-1.0..1.0)),
            angular_velocities: std::array::from_fn(|_| rng.random_range(-3.0..3.0)),
        }
    }

    // Florian's (2007) equations 20 to 22 for one pole: θ̈ with the sign of N assumed, N from θ̈,
    // again with the other sign if N changed sign, then ẍ. Equation 21 (and 19) as printed has
    // `cos θ − μc sgn(N ẋ)` in its denominator: substituting his equation 18 into 16, as he says
    // he does, gives `cos θ − μc sgn(N ẋ) sin θ`, used here (and Newton's laws agree, below)
    fn florian(state: &State<1>, force: f64) -> (f64, f64) {
        let (mc, mp, l, g) = (CART_MASS, 0.1, 0.5, GRAVITY);
        let (muc, mup) = (CART_FRICTION, POLE_FRICTION);
        let (theta, omega, v) = (state.angles[0], state.angular_velocities[0], state.velocity);
        let (sin, cos) = (theta.sin(), theta.cos());
        let theta_acceleration = |normal_sign: f64| {
            let s = normal_sign * v.signum() * f64::from(u8::from(v != 0.0));
            let numerator = g * sin
                + cos
                    * ((-force - mp * l * omega * omega * (sin + muc * s * cos)) / (mc + mp)
                        + muc * g * s)
                - mup * omega / (mp * l);
            let denominator = l * (4.0 / 3.0 - mp * cos / (mc + mp) * (cos - muc * s * sin));
            numerator / denominator
        };
        let normal = |alpha: f64| (mc + mp) * g - mp * l * (alpha * sin + omega * omega * cos);
        let mut normal_sign = 1.0;
        let mut alpha = theta_acceleration(normal_sign);
        if normal(alpha) < 0.0 {
            normal_sign = -1.0;
            alpha = theta_acceleration(normal_sign);
        }
        let n = normal(alpha);
        let s = normal_sign * v.signum() * f64::from(u8::from(v != 0.0));
        let x_acceleration =
            (force + mp * l * (omega * omega * sin - alpha * cos) - muc * n * s) / (mc + mp);
        (x_acceleration, alpha)
    }

    #[test]
    fn one_pole_follows_florians_equations() {
        let cart = CartPole::new().cart;
        let mut rng = StreamRng::seed_from_u64(1);
        for _ in 0..1000 {
            let state = random_state::<1>(&mut rng);
            let force = rng.random_range(-10.0..10.0);
            let (derivative, _) = cart.derivative(&state, force, &mut 1.0);
            let (x_acceleration, alpha) = florian(&state, force);
            assert!(close(derivative.velocity, x_acceleration, 1e-12));
            assert!(close(derivative.angular_velocities[0], alpha, 1e-12));
            assert_eq!(derivative.x, state.velocity);
            assert_eq!(derivative.angles, state.angular_velocities);
        }
    }

    #[test]
    fn frictionless_equations_are_barto_and_wielands() {
        // without friction: Florian's equations 23 and 24, which are Barto et al.'s with g > 0
        let cart = CartPole::new().without_friction().cart;
        let mut rng = StreamRng::seed_from_u64(2);
        for _ in 0..100 {
            let state = random_state::<1>(&mut rng);
            let force = rng.random_range(-10.0..10.0);
            let (derivative, _) = cart.derivative(&state, force, &mut 1.0);
            let (mc, mp, l, g) = (1.0, 0.1, 0.5, 9.8);
            let (theta, omega) = (state.angles[0], state.angular_velocities[0]);
            let alpha = (g * theta.sin()
                + theta.cos() * (-force - mp * l * omega * omega * theta.sin()) / (mc + mp))
                / (l * (4.0 / 3.0 - mp * theta.cos() * theta.cos() / (mc + mp)));
            let x_acceleration =
                (force + mp * l * (omega * omega * theta.sin() - alpha * theta.cos())) / (mc + mp);
            assert!(close(derivative.velocity, x_acceleration, 1e-12));
            assert!(close(derivative.angular_velocities[0], alpha, 1e-12));
        }
    }

    // solves a x = b by Gaussian elimination with partial pivoting
    fn solve<const K: usize>(mut a: [[f64; K]; K], mut b: [f64; K]) -> [f64; K] {
        for column in 0..K {
            let pivot = (column..K)
                .max_by(|&i, &j| a[i][column].abs().total_cmp(&a[j][column].abs()))
                .unwrap();
            a.swap(column, pivot);
            b.swap(column, pivot);
            for row in column + 1..K {
                let factor = a[row][column] / a[column][column];
                for k in column..K {
                    a[row][k] -= factor * a[column][k];
                }
                b[row] -= factor * b[column];
            }
        }
        let mut x = [0.0; K];
        for row in (0..K).rev() {
            let mut sum = b[row];
            for k in row + 1..K {
                sum -= a[row][k] * x[k];
            }
            x[row] = sum / a[row][row];
        }
        x
    }

    // Newton's laws for the cart and two poles (Florian's equations 4, 5, 11, 12 and 14 for each
    // pole), solved for (ẍ, θ̈₁, θ̈₂, N) with the friction's sign s = sgn(N ẋ)
    fn newton(state: &State<2>, force: f64, s: f64) -> [f64; 4] {
        let (masses, lengths) = ([0.1, 0.01], [0.5, 0.05]);
        let (m_total, g) = (CART_MASS + 0.1 + 0.01, GRAVITY);
        let mut a = [[0.0; 4]; 4];
        let mut b = [0.0; 4];
        // the cart: (M + Σm) ẍ + Σ mᵢ lᵢ cos θᵢ θ̈ᵢ + μc s N = F + Σ mᵢ lᵢ θ̇ᵢ² sin θᵢ
        a[0][0] = m_total;
        a[0][3] = CART_FRICTION * s;
        b[0] = force;
        // the normal force: N + Σ mᵢ lᵢ sin θᵢ θ̈ᵢ = (M + Σm) g − Σ mᵢ lᵢ θ̇ᵢ² cos θᵢ
        a[3][3] = 1.0;
        b[3] = m_total * g;
        for i in 0..2 {
            let (m, l) = (masses[i], lengths[i]);
            let (theta, omega) = (state.angles[i], state.angular_velocities[i]);
            a[0][1 + i] = m * l * theta.cos();
            b[0] += m * l * omega * omega * theta.sin();
            a[3][1 + i] = m * l * theta.sin();
            b[3] -= m * l * omega * omega * theta.cos();
            // the pole's rotation: 4/3 m l² θ̈ + m l cos θ ẍ = m g l sin θ − μp θ̇
            a[1 + i][1 + i] = 4.0 / 3.0 * m * l * l;
            a[1 + i][0] = m * l * theta.cos();
            b[1 + i] = m * g * l * theta.sin() - POLE_FRICTION * omega;
        }
        solve(a, b)
    }

    #[test]
    fn two_poles_follow_newtons_laws() {
        let cart = DoublePole::new().cart;
        let mut rng = StreamRng::seed_from_u64(3);
        let mut negative_normal = 0;
        for trial in 0..2000 {
            let mut state = random_state::<2>(&mut rng);
            if trial % 2 == 0 {
                // fast enough to lift the cart: a negative normal force
                state.angular_velocities[0] = rng.random_range(-20.0..20.0);
            }
            let force = rng.random_range(-10.0..10.0);
            let (derivative, normal) = cart.derivative(&state, force, &mut 1.0);
            let v_sign = sign(state.velocity);
            let mut solution = newton(&state, force, v_sign);
            if solution[3] < 0.0 {
                solution = newton(&state, force, -v_sign);
                negative_normal += 1;
            }
            assert!(close(derivative.velocity, solution[0], 1e-10));
            assert!(close(derivative.angular_velocities[0], solution[1], 1e-10));
            assert!(close(derivative.angular_velocities[1], solution[2], 1e-10));
            assert!(close(normal, solution[3], 1e-10));
        }
        assert!(negative_normal > 0);
    }

    // the energy and the horizontal momentum of a cart with poles (Florian's pole: a rod of length
    // 2l, with a moment of inertia m l² / 3 about its centre)
    fn energy_and_momentum<const N: usize>(cart: &Cart<N>) -> (f64, f64) {
        let state = &cart.state;
        let mut energy = 0.5 * CART_MASS * state.velocity * state.velocity;
        let mut momentum = CART_MASS * state.velocity;
        for i in 0..N {
            let (m, l) = (cart.masses[i], cart.half_lengths[i]);
            let (theta, omega) = (state.angles[i], state.angular_velocities[i]);
            let horizontal = state.velocity + l * omega * theta.cos();
            let vertical = -l * omega * theta.sin();
            energy += 0.5 * m * (horizontal * horizontal + vertical * vertical)
                + 0.5 * m * l * l / 3.0 * omega * omega
                + m * GRAVITY * l * theta.cos();
            momentum += m * horizontal;
        }
        (energy, momentum)
    }

    #[test]
    fn a_frictionless_run_conserves_energy_and_momentum() {
        // falling poles, and hanging ones that swing for 100,000 steps
        let falling = [0.3, 1.0, -0.4, 0.0, 0.2, 0.5];
        let hanging = [0.0, 0.5, 3.0, 0.0, -3.0, 1.0];
        for (start, steps) in [(falling, 1000), (hanging, 100_000)] {
            let mut cart = DoublePole::new().without_friction().with_state(start).cart;
            let (energy, momentum) = energy_and_momentum(&cart);
            let mut largest = (0.0_f64, 0.0_f64);
            for _ in 0..steps {
                cart.advance(0.0);
                let (e, p) = energy_and_momentum(&cart);
                largest.0 = largest.0.max((e - energy).abs());
                largest.1 = largest.1.max((p - momentum).abs());
            }
            // Runge-Kutta's error only: a few 1e-7 of energies and momenta of order 1
            assert!(
                largest.0 < 1e-5,
                "{steps} steps: energy drift {}",
                largest.0
            );
            assert!(
                largest.1 < 1e-5,
                "{steps} steps: momentum drift {}",
                largest.1
            );
        }
        let mut cart = CartPole::new()
            .without_friction()
            .with_state([0.0, 0.3, 2.5, 0.0])
            .cart;
        let (energy, momentum) = energy_and_momentum(&cart);
        for _ in 0..100_000 {
            cart.advance(0.0);
        }
        let (e, p) = energy_and_momentum(&cart);
        assert!((e - energy).abs() < 1e-5 && (p - momentum).abs() < 1e-5);
    }

    #[test]
    fn friction_only_dissipates() {
        let mut cart = DoublePole::new()
            .with_state([0.0, 0.5, 3.0, 0.0, -3.0, 1.0])
            .cart;
        let (mut energy, _) = energy_and_momentum(&cart);
        let start = energy;
        for _ in 0..10_000 {
            cart.advance(0.0);
            let (e, _) = energy_and_momentum(&cart);
            assert!(e <= energy + 1e-9);
            energy = e;
        }
        assert!(energy < start);
    }

    #[test]
    fn a_step_is_two_runge_kutta_steps_of_florians_equations() {
        let mut task = CartPole::new().with_state([0.1, -0.2, 0.05, 0.3]);
        let florian_derivative = |s: &State<1>| {
            let (x_acceleration, alpha) = florian(s, 2.5);
            State {
                x: s.velocity,
                velocity: x_acceleration,
                angles: s.angular_velocities,
                angular_velocities: [alpha],
            }
        };
        let mut expected = task.cart.state;
        for _ in 0..2 {
            let k1 = florian_derivative(&expected);
            let k2 = florian_derivative(&expected.plus(&k1, 0.005));
            let k3 = florian_derivative(&expected.plus(&k2, 0.005));
            let k4 = florian_derivative(&expected.plus(&k3, 0.01));
            expected = expected.runge_kutta([&k1, &k2, &k3, &k4], 0.01);
        }
        assert!(task.step(0.25));
        let [x, v, theta, omega] = task.state();
        assert!(close(x, expected.x, 1e-13));
        assert!(close(v, expected.velocity, 1e-13));
        assert!(close(theta, expected.angles[0], 1e-13));
        assert!(close(omega, expected.angular_velocities[0], 1e-13));
    }

    #[test]
    fn forces() {
        assert_eq!(force(1.0), 10.0);
        assert_eq!(force(3.0), 10.0);
        assert_eq!(force(-7.0), -10.0);
        assert_eq!(force(0.5), 5.0);
        assert_eq!(force(0.0), 10.0 / 256.0);
        assert_eq!(force(-0.0), 10.0 / 256.0);
        assert_eq!(force(-0.001), -10.0 / 256.0);
        assert_eq!(force(0.003), 10.0 / 256.0);
        assert!(force(f64::NAN).is_nan());
    }

    #[test]
    fn a_leaning_pole_falls_and_pushing_moves_the_cart() {
        let mut task = CartPole::new();
        let mut steps = 0;
        // the smallest force, to positive x: the pole, at 4°, falls on
        while task.step(0.0) {
            steps += 1;
        }
        let [_, _, theta, _] = task.state();
        assert!(theta > degrees(12.0), "{:?}", task.state());
        assert!((10..100).contains(&steps), "{steps}");
        let mut task = CartPole::new().with_state([0.0; 4]);
        task.step(-1.0);
        let [_, v, theta, _] = task.state();
        // pushed to negative x, the cart leaves the pole leaning to positive angles
        assert!(v < 0.0 && theta > 0.0);
    }

    #[test]
    fn initial_states_and_observations() {
        assert_eq!(CartPole::new().state(), [0.0, 0.0, degrees(4.0), 0.0]);
        let task = DoublePole::new();
        assert_eq!(task.state(), [0.0, 0.0, degrees(4.0), 0.0, 0.0, 0.0]);
        let task = task.with_state([1.2, -1.0, degrees(18.0), 0.5, degrees(-9.0), 4.0]);
        let mut observation = [0.0; 6];
        task.observe(&mut observation);
        assert_eq!(observation, [0.5, -0.5, 0.5, 0.25, -0.25, 2.0]);
        let task = DoublePole::without_velocities().with_state(task.state());
        assert_eq!(task.observations(), 3);
        let mut observation = [0.0; 3];
        task.observe(&mut observation);
        assert_eq!(observation, [0.5, 0.5, -0.25]);
        // a reset goes back to the state given
        let mut task = task;
        task.step(1.0);
        task.reset();
        assert_eq!(task.state()[0], 1.2);
    }

    #[test]
    fn the_failure_bounds() {
        let up = |state| CartPole::new().with_state(state).balanced();
        assert!(up([2.4, 0.0, degrees(12.0), 0.0]));
        assert!(!up([2.41, 0.0, 0.0, 0.0]));
        assert!(!up([0.0, 0.0, degrees(-12.1), 0.0]));
        assert!(!up([f64::NAN, 0.0, 0.0, 0.0]));
        let up = |state| DoublePole::new().with_state(state).balanced();
        assert!(up([-2.4, 0.0, degrees(36.0), 0.0, degrees(-36.0), 0.0]));
        assert!(!up([0.0, 0.0, 0.0, 0.0, degrees(36.1), 0.0]));
        // a NaN action fails at once
        let mut nan = |_: &[f64], action: &mut [f64]| action[0] = f64::NAN;
        assert_eq!(DoublePole::new().run(&mut nan, 100), 0);
    }

    // a linear state feedback: pushes the cart under the pole, and back to the middle
    fn linear_controller(observation: &[f64], action: &mut [f64]) {
        let [x, v, theta, omega] = observation else {
            unreachable!()
        };
        action[0] = 2.0 * theta + omega + 0.1 * x + 0.2 * v;
    }

    #[test]
    fn a_linear_controller_solves_the_cart_pole() {
        let task = CartPole::new();
        let mut policy = linear_controller;
        assert!(task.solved(&mut policy));
        assert_eq!(task.run(&mut policy, SUCCESS_STEPS), SUCCESS_STEPS);
        // and the double pole isn't solved by pushing one way
        let mut push = |_: &[f64], action: &mut [f64]| action[0] = 1.0;
        assert!(!DoublePole::new().solved(&mut push));
        assert!(DoublePole::new().run(&mut push, SUCCESS_STEPS) < 100);
    }

    #[test]
    fn the_damping_fitness() {
        // falling before 100 steps: 0.1 t / 1000
        let mut push = |_: &[f64], action: &mut [f64]| action[0] = 1.0;
        let task = DoublePole::without_velocities();
        let t = task.run(&mut push, DAMPING_STEPS);
        assert!(t < 100);
        assert_eq!(task.damping_fitness(&mut push), 0.1 * f64::from(t) / 1000.0);

        // a policy balancing all 1000 steps: the offsets of the last 100 states, by hand
        let mut policy = |observation: &[f64], action: &mut [f64]| {
            let [x, v, theta, omega, _, _] = observation else {
                unreachable!()
            };
            action[0] = 2.0 * theta + omega + 0.1 * x + 0.2 * v;
        };
        let mut task = DoublePole::new().with_state([0.0; 6]);
        let fitness = task.damping_fitness(&mut policy);
        let mut offsets = Vec::new();
        let mut observation = [0.0; 6];
        let mut action = [0.0];
        let mut balanced = 0;
        for _ in 0..DAMPING_STEPS {
            task.observe(&mut observation);
            policy(&observation, &mut action);
            if !task.step(action[0]) {
                break;
            }
            balanced += 1;
            let [x, v, theta, omega, _, _] = task.state();
            offsets.push(x.abs() + v.abs() + theta.abs() + omega.abs());
        }
        let f1 = f64::from(balanced) / 1000.0;
        let f2 = if balanced < 100 {
            0.0
        } else {
            let mut sum = 0.0;
            for offset in &offsets[offsets.len() - 100..] {
                sum += offset;
            }
            0.75 / sum
        };
        assert_eq!(fitness, 0.1 * f1 + 0.9 * f2);
    }

    #[test]
    fn the_generalization_starts() {
        let starts: Vec<[f64; 6]> = DoublePole::generalization_starts().collect();
        assert_eq!(starts.len(), 625);
        let (first, last) = (starts[0], starts[624]);
        let expected = [-1.944, -1.215, degrees(-3.24), degrees(-7.74), 0.0, 0.0];
        for i in 0..6 {
            assert!(close(first[i], expected[i], 1e-12), "{first:?}");
            assert!(close(last[i], -expected[i], 1e-12), "{last:?}");
        }
        // the middle start is at rest
        assert_eq!(starts[312], [0.0; 6]);
        let mut sorted: Vec<[u64; 6]> = starts.iter().map(|s| s.map(f64::to_bits)).collect();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), 625);
        // a policy that pushes one way passes none
        let mut push = |_: &[f64], action: &mut [f64]| action[0] = 1.0;
        assert_eq!(
            DoublePole::without_velocities().generalization(&mut push),
            0
        );
    }

    #[test]
    fn networks_are_policies() {
        let mlp = Mlp::new([4, 1], Activation::Identity).unwrap().bias(false);
        // the linear controller as a network
        let weights = [0.1, 0.2, 2.0, 1.0];
        let mut network = mlp.with(&weights).unwrap();
        let task = CartPole::new();
        assert_eq!(task.run(&mut network, 1000), 1000);
        // an Elman network is reset before each episode
        let elman = Elman::new(3, 2, 1, Activation::Tanh).unwrap();
        let weights = vec![0.3; elman.parameters()];
        let mut network = elman.with(&weights).unwrap();
        let task = DoublePole::without_velocities();
        let first = task.damping_fitness(&mut network);
        assert_eq!(task.damping_fitness(&mut network), first);
    }

    #[test]
    fn episodes_are_the_same_bits_every_time() {
        let task = DoublePole::new();
        let mut rng = StreamRng::seed_from_u64(4);
        let actions: Vec<f64> = (0..200).map(|_| rng.random_range(-1.0..1.0)).collect();
        let run = || {
            let mut task = task;
            let mut states = Vec::new();
            for &action in &actions {
                task.step(action);
                states.push(task.state().map(f64::to_bits));
            }
            states
        };
        assert_eq!(run(), run());
    }
}
