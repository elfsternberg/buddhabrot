#![deny(missing_docs)]
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

//! Buddhabrot renderer
//!
//! The Buddhabrot (and the Nebulabrot) are variants of the Mandelbrot
//! set that explore "what's in the black heart" of the Mandelbrot.
//! The Mandelbrot takes a point on the complex plane and repeatedly
//! multiplies it by itself, measuring how quickly that number goes to
//! infinity.  This "velocity" is the number used to render the image.
//!
//! The black heart of the mandelbrot consists of points that have no
//! velocity, that is, they never go to infinity when iterated.
//! However, each iteration creates a new complex number that itself
//! may be used as a coordinate on the complex plane.  By mapping that
//! coordinate to the nearest integral pixel and incrementing that
//! pixel by one, we can plot the "orbit" of all the points accessible
//! within the heart, and most points within the heart create orbits
//! that stay within the heart.  The resulting image is called a
//! Buddhabrot.

extern crate crossbeam;
extern crate image;
extern crate itertools;
extern crate num;
extern crate num_cpus;

pub mod naive;
pub use naive::NaiveRenderer;
