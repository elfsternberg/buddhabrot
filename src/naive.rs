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

use itertools::iproduct;
use num::Complex;
use std::ops::Range;
use std::sync::{Arc, Mutex};

use planes::{Pixel, PlaneMapper};

type PixelType = Arc<Mutex<itertools::Product<Range<usize>, Range<usize>>>>;

/// Takes a plane and a limit (the number of iterations to conduct
/// per-point, and creates a "naive" buddhabrot out of it.  "Naive" in
/// this case means that it attempts to create a
/// pixel->point->orbit->pixels relationship, but individual pixels do
/// not map to enough points to create really interesting
/// high-resolution buddhabrots.  It also plots *all* of the points,
/// even those that are fairly well-known to be worthless.
pub struct NaiveRenderer {
    plane: PlaneMapper,
    limit: usize,
}

impl NaiveRenderer {
    /// Requires the width and height of the image, the left-lower and
    /// right-upper corners of the complex plane where the calculation
    /// will take place, and the number of iterations to perform on a
    /// per-orbit basis.
    pub fn new(
        width: usize,
        height: usize,
        leftlower: Complex<f64>,
        rightupper: Complex<f64>,
        limit: usize,
    ) -> Result<Self, String> {
        match PlaneMapper::new(width, height, leftlower, rightupper) {
            Ok(plane) => Ok(NaiveRenderer { plane, limit }),
            Err(u) => Err(u),
        }
    }

    /// This is the 'primary' helper function, in that its purpose is to
    /// take a point, a plane, and a buffer, and plot the orbit of
    /// that point up to some limit, either some max number, or the
    /// depth until transition outside the black heart.  We don't test
    /// for escape here because we assume the calling function already
    /// determined the escape limit.
    fn plot(&self, start: Complex<f64>, buffer: &mut [u32], limit: usize) {
        let mut z: Complex<f64> = Complex { re: 0.0, im: 0.0 };
        for _ in 0..limit {
            z = z * z + start;
            if let Some(offset) = self.plane.point_to_offset(&z) {
                buffer[offset] += 1;
            }
        }
    }

    /// The main function for single-threaded implementations.  This produces
    /// the naive buddhabrot.
    pub fn buddhabrot_single(&self) -> Result<Vec<u32>, String> {
        let mut buffer = vec![0 as u32; self.plane.len()];
        for column in 0..self.plane.integral_plane.0 {
            for row in 0..self.plane.integral_plane.1 {
                let mut z: Complex<f64> = Complex { re: 0.0, im: 0.0 };
                let p = Pixel(column, row);
                let c = self.plane.pixel_to_point(&p);
                for i in 0..self.limit {
                    z = z * z + c;
                    if z.norm_sqr() >= 4.0 {
                        self.plot(c, &mut buffer, i);
                        break;
                    }
                }
            }
        }
        Ok(buffer)
    }

    /// Given a collection of planes in a contiguous block, merge them
    /// all into a single plane.  This is a helper function to the
    /// threaded Buddhabrot function.
    fn render_merge(&self, regions: &[u32]) -> Vec<u32> {
        let mut ret = vec![0 as u32; self.plane.len()];
        let regions: Vec<&[u32]> = regions.chunks(self.plane.len()).collect();
        for i in 0..ret.len() {
            for region in &regions {
                ret[i] += region[i];
            }
        }
        ret
    }

    /// A multi-threaded version of the render function that takes a thread count
    /// as an option.
    pub fn buddhabrot(&self, threads: usize) -> Result<Vec<u32>, String> {
        let mut allocation = vec![0 as u32; self.plane.len() * threads];
        crossbeam::scope(|spawner| {
            let regions: Vec<&mut [u32]> = allocation.chunks_mut(self.plane.len()).collect();
            {
                let pixels: PixelType = Arc::new(Mutex::new(iproduct!(
                    0..self.plane.integral_plane.0,
                    0..self.plane.integral_plane.1
                )));
                for region in regions {
                    let pixels = pixels.clone();
                    spawner.spawn(move |_| loop {
                        let pixel = { pixels.lock().unwrap().next() };
                        match pixel {
                            Some(pixel) => {
                                let mut z: Complex<f64> = Complex { re: 0.0, im: 0.0 };
                                let point = self.plane.pixel_to_point(&Pixel(pixel.0, pixel.1));
                                for i in 0..self.limit {
                                    z = z * z + point;
                                    if z.norm_sqr() >= 4.0 {
                                        self.plot(point, region, i);
                                        break;
                                    }
                                }
                            }
                            None => {
                                break;
                            }
                        }
                    });
                }
            }
        })
        .unwrap();
        Ok(self.render_merge(&allocation))
    }
}
