//! The CupeRenderer is based on an algorithm by Johann Korndoerfer,
//! which was the easiest to read of all the publicly available
//! algorithms I could find despite its being written in Common Lisp.
//! It has a lot more knobs and dials than the NaiveRenderer.

extern crate crossbeam;

use crossbeam::thread::ScopedJoinHandle;
use num::complex::Complex;
use planes::PlaneMapper;
use rand::distributions::{Distribution, Uniform};
use rand::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

macro_rules! t {
    ($condition: expr, $_true: expr, $_false: expr) => {
        if $condition {
            $_true
        } else {
            $_false
        }
    };
}

/// Given a complex plane and a resolution, we try to find a square size
/// that we can break the region up into sub-regions, each of which
/// can then be tested for "interesting cells."

struct Cells {
    leftlower: Complex<f64>,
    rightupper: Complex<f64>,
    resolution: f64,
    re: f64,
    im: f64,
}

impl Cells {
    pub fn new(leftlower: Complex<f64>, rightupper: Complex<f64>, resolution: f64) -> Self {
        Cells {
            leftlower,
            rightupper,
            resolution,
            re: leftlower.re,
            im: leftlower.im,
        }
    }
}

impl Iterator for Cells {
    type Item = Complex<f64>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.im > self.rightupper.im {
            self.re += self.resolution;
            self.im = self.leftlower.im;
        }

        if self.re > self.rightupper.re {
            return None;
        }

        let val = Complex::new(self.re, self.im);
        self.im += self.resolution;
        Some(val)
    }
}

/// The CupeRenderer contains the parameters by which a buddhabrot is
/// generated.  Once set, this object should not be mutable.
pub struct CupeRenderer {
    // For the sample quality pass, the minimum number of iterations we
    // want to see (default: 1/5th the max)
    min_iterations: usize,
    // For all passes, the maximum number of iterations allowed.
    max_iterations: usize,
    // For the mapping pass, the number of cells in a given sample that
    // are discarded before plotting begins (default: 1/100th of the
    // max_iterations
    min_plot_count: usize,
    // For a given cell (see above), the number of samples of the cell
    // we randomly sample to determine its utility to the buddhabrot.
    // The default is a concrete 200.
    samples_per_cell: usize,
    // The size of the "space" around an absolute point in complex
    // space that we probe for "interestingness."  I'd like, somehow,
    // to synthesize this out of a relationship between the integral
    // plane, the complex plane, and an arbitrary "resolution" figure.
    // The default is a concrete 0.005. Multiplied by the
    // samples_per_cell is 1.0, which is still a meaningless and
    // arbitrary number, but at least it's something to work with.
    cell_size: f64,
    // The maximum number of iterations we should do for checking a single cell:
    max_scan: usize,
    // The two planes about which we care:
    pub planes: PlaneMapper,
}

impl CupeRenderer {
    /// TODO: Placeholder new.  We need something smarter than this.
    pub fn new(
        width: usize,
        height: usize,
        leftlower: Complex<f64>,
        rightupper: Complex<f64>,
    ) -> Result<Self, String> {
        match PlaneMapper::new(width, height, leftlower, rightupper) {
            Ok(planes) => Ok(CupeRenderer {
                min_iterations: 10_000,
                max_iterations: 100_000,
                min_plot_count: 500,
                samples_per_cell: 200,
                max_scan: 1000,
                cell_size: 0.0005,
                planes,
            }),
            Err(u) => Err(u),
        }
    }
}

struct CellPoint(Uniform<f64>, ThreadRng);

impl CellPoint {
    pub fn new(max: f64) -> Self {
        let u = Uniform::new(0.0_f64, max);
        CellPoint(u, rand::thread_rng())
    }
    pub fn get(&mut self) -> f64 {
        self.0.sample(&mut self.1)
    }
}

/// The nice thing about Buddhabrot is that, unlike the Mandelbrot
/// set, there's a very finite universe in which you're allowed to
/// play.  The bad thing about this is that the universe is
/// *absolute*; you can't zoom in on a part of the Buddhabrot
/// without rendering the whole damn thing every time.
///
/// This routine finds "interesting points" in the complex plane.
/// For our purposes, "interesting" means that there's a
/// probabilistic zone around the point through which the border
/// of the Mandelbrot set passes.  The important details here are
/// the size of the zone, which, like the Complex plane itself, is
/// an absolute number; in the default case, the universe fits in
/// a complex cartesian plane approximately 3.5x x 2.5iy, our
/// resolution is every 0.005, and we pick about 200 random points
/// in a square .005 from every point.  (After doing so, we also
/// increment to the next "cell" by that much in a fairly classic
/// raster scan.)
pub fn find_interesting_points(cupe: &CupeRenderer, threads: usize) -> Vec<Complex<f64>> {
    let cells = Cells::new(
        cupe.planes.complex_plane.0,
        cupe.planes.complex_plane.1,
        cupe.cell_size,
    );
    let cells = Arc::new(Mutex::new(cells));

    let mut points: Vec<Complex<f64>> = vec![];
    crossbeam::scope(|spawner| {
        let handles: Vec<ScopedJoinHandle<Vec<Complex<f64>>>> = (0..threads)
            .map(|_i| {
                let cells = cells.clone();
                let mut points: Vec<Complex<f64>> = vec![];
                spawner.spawn(move |_| {
                    let mut rng = CellPoint::new(cupe.cell_size);
                    loop {
                        let cell = { cells.lock().unwrap().next() };
                        match cell {
                            Some(cell) => {
                                let ur = Complex::new(
                                    cell.re + cupe.cell_size,
                                    cell.im + cupe.cell_size,
                                );
                                points.extend(find_interesting_points_inside(
                                    cupe, cell, ur, &mut rng,
                                ));
                            }
                            None => {
                                break;
                            }
                        }
                    }
                    points
                })
            })
            .collect();

        points = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .flatten()
            .collect()
    })
    .unwrap();
    points
}

fn find_interesting_points_inside(
    cupe: &CupeRenderer,
    leftlower: Complex<f64>,
    rightupper: Complex<f64>,
    rng: &mut CellPoint,
) -> Vec<Complex<f64>> {
    let mut points: Vec<Complex<f64>> = vec![];
    let mut re = leftlower.re;
    while re < rightupper.re {
        let mut im = leftlower.im;
        while im < rightupper.im {
            let point = Complex::new(re, im);
            if is_interesting_cell(cupe, point, rng) {
                points.push(point);
            }
            im = im + cupe.cell_size;
        }
        re = re + cupe.cell_size;
    }
    points
}

/// A "cell" is a region in the complex space, defined as a
/// rectangle.  This function samples the cell a small number of
/// times and tries to figure out if the cell really is
/// "interesting."  If it is, return its coordinates.
fn is_interesting_cell(cupe: &CupeRenderer, point: Complex<f64>, rng: &mut CellPoint) -> bool {
    let (mut seen_inside, mut seen_outside) = (false, false);
    for _ in 0..cupe.samples_per_cell {
        match iterate_random_sample(cupe, point, rng) {
            None => {
                seen_inside = true;
            }
            Some(_) => {
                seen_outside = true;
            }
        };
        if seen_inside && seen_outside {
            return true;
        }
    }
    false
}

/// A helper function that finds a random point within a cell and if
/// there's a chance the cell is "interesting" actually run the
/// expensive version of the function to make sure.
fn iterate_random_sample(
    cupe: &CupeRenderer,
    point: Complex<f64>,
    rng: &mut CellPoint,
) -> Option<usize> {
    let c = Complex {
        re: point.re + rng.get(),
        im: point.im + rng.get(),
    };
    if maybe_outside(c) {
        iterate_sample(&c, cupe.max_scan)
    } else {
        None
    }
}

const D4: f64 = 1.0 / 4.0;
const D16: f64 = D4 / 4.0;

/// As I understand it, the two halves of the `and` expression
/// represent false if the point is guaranteed to be inside the
/// mandelbrot set.  Is does *not* guarantee that a point will be
/// outside, however; it is at best a crude estimation, and there
/// are points inside the mandelbrot set for which this will
/// return true.  Those still have to be checked, but it
/// constrains our problem set a little bit, and that's what we
/// want.
pub fn maybe_outside(point: Complex<f64>) -> bool {
    let y = point.im.powf(2.0);
    let q = y + (point.re - D4).powf(2.0);
    q * (q + point.re - D4) > (y * D4) && (point.re + 1.0_f64).powf(2.0) + y > D16
}

/// This is our classic iterator function, which either returns the
/// number of iterations it took to escape the Mandelbrot set, or
/// it returns nothing at all.
pub fn iterate_sample(point: &Complex<f64>, max_iterations: usize) -> Option<usize> {
    let mut z = Complex {
        re: 0.0_f64,
        im: 0.0_f64,
    };
    for i in 0..max_iterations {
        z = z * z + point;
        if i % 8 == 0 && z.norm_sqr() >= 4.0_f64 {
            return Some(i);
        }
    }
    None
}

/// Given a collection of possibly interesting points, now revisit
/// them to find out which ones really are in the Buddhabrot and
/// when those leave the Mandelbrot set.
pub fn collect_samples(
    cupe: &CupeRenderer,
    points: &Vec<Complex<f64>>,
    threads: usize,
) -> Vec<(Complex<f64>, usize)> {
    let mut samples: Vec<(Complex<f64>, usize)> = vec![];
    let points = Arc::new(Mutex::new(points.into_iter()));
    let cell_size = cupe.cell_size;

    crossbeam::scope(|spawner| {
        let handles: Vec<ScopedJoinHandle<Vec<(Complex<f64>, usize)>>> = (0..threads)
            .map(|_| {
                let points = points.clone();
                spawner.spawn(move |_| {
                    let mut samples: Vec<(Complex<f64>, usize)> = vec![];
                    let mut rng = CellPoint::new(cell_size);
                    loop {
                        let point = { points.lock().unwrap().next() };
                        match point {
                            Some(point) => {
                                let c = Complex {
                                    re: point.re + rng.get(),
                                    im: point.im + rng.get(),
                                };
                                if let Some(i) = iterate_sample(point, cupe.max_iterations) {
                                    samples.push((c, i));
                                }
                            }
                            None => {
                                break;
                            }
                        }
                    }
                    samples
                })
            })
            .collect();

        let min_iterations = cupe.min_iterations;
        samples = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .flatten()
            .filter(|s| s.1 >= min_iterations)
            .collect()
    })
    .unwrap();
    samples
}

/// Given the final collection of points, map them to the plane.
pub fn map_samples(
    cupe: &CupeRenderer,
    points: &[(Complex<f64>, usize)],
) -> Result<Vec<u16>, String> {
    let mut plane = vec![0 as u16; cupe.planes.len()];
    for point in points {
        let mut z = Complex {
            re: 0.0_f64,
            im: 0.0_f64,
        };
        for i in 0..point.1 {
            z = z * z + point.0;
            if z.norm_sqr() >= 4.0_f64 {
                break;
            }
            if i < cupe.min_plot_count {
                continue;
            }
            if let Some(offset) = cupe.planes.point_to_offset(&z) {
                if plane[offset] < 65535 {
                    plane[offset] += 1;
                }
            }
        }
    }
    Ok(plane)
}

/// The main function, and primary entry point.
pub fn cupe_buddhabrot(cupe: &CupeRenderer, threads: usize) -> Result<Vec<u16>, String> {
    let interesting_points = find_interesting_points(cupe, threads);
    let valid_points = collect_samples(cupe, &interesting_points, threads);
    map_samples(cupe, &valid_points)
}
