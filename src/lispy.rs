//! The CupeRenderer is based on an algorithm by Johann Korndoerfer,
//! which was the easiest to read of all the publicly available
//! algorithms I could find despite its being written in Common Lisp.
//! It has a lot more knobs and dials than the NaiveRenderer.

use num::complex::Complex;
use rand::prelude::*;
use naive::PlaneMapper;

/// The CupeRenderer is a primitive at this point.  TODO
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
    planes: PlaneMapper,
    // Our rng seed
    rng: ThreadRng,
        
}


impl CupeRenderer {
    /// TODO: Placeholder new.  We need something smarter than this.
    pub fn new(width: usize, height: usize, leftlower: Complex<f64>, rightupper: Complex<f64>) -> Result<Self, String> {
        let rng = rand::thread_rng();
        match PlaneMapper::new(width, height, leftlower, rightupper) {
            Ok(planes) => Ok(CupeRenderer {
                min_iterations: 10_000,
                max_iterations: 50_000,
                min_plot_count: 500,
                samples_per_cell: 200,
                max_scan: 1000,
                cell_size: 0.005,
                planes,
                rng
           }),
            Err(u) => Err(u)
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
    pub fn find_interesting_points(&mut self) -> Vec<Complex<f64>> {
        let mut ll = self.planes.complex_plane.0.re;
        let mut ret: Vec<Complex<f64>> = vec![];
        loop {
            let mut lb = self.planes.complex_plane.0.im;
            loop {
                let (mut seen_inside, mut seen_outside) = (false, false);
                for _ in 0..self.samples_per_cell {
                    let point = Complex{ re: ll, im: lb };
                    let count = self.iterate_random_sample(point);
                    match count {
                        None => { seen_inside = true; }
                        Some(_) => { seen_outside = true; }
                    };
                    if seen_inside && seen_outside {
                        ret.push(point);
                        break;
                    }
                }
                lb = lb + self.cell_size;
                if lb >= self.planes.complex_plane.1.im {
                    break;
                }
            }
            ll = ll + self.cell_size;
            if lb >= self.planes.complex_plane.1.re {
                break;
            }
        }
        ret
    }

    /// A helper function to the above, finds a random point within the
    /// zone about the sample point, and then if there's a good chance
    /// the cell is "interesting," actually run the expensive version
    /// of the function to make sure.
    fn iterate_random_sample(&mut self, point: Complex<f64>) -> Option<usize> {
        let c = Complex{ re: point.re + self.rng.gen_range(0.0_f64, self.cell_size),
                         im: point.im + self.rng.gen_range(0.0_f64, self.cell_size) };
        if Self::maybe_outside(c) {
            self.iterate_sample(&c, self.max_scan)
        } else {
            None
        }
    }
    
    /// As I understand it, the two halves of the `and` expression
    /// represent false if the point is guaranteed to be inside the
    /// mandelbrot set.  Is does *not* guarantee that a point will be
    /// outside, however; it is at best a crude estimation, and there
    /// are points inside the mandelbrot set for which this will
    /// return true.  Those still have to be checked, but it
    /// constrains our problem set a little bit, and that's what we
    /// want.
    pub fn maybe_outside(point: Complex<f64>) -> bool {
        let y = point.im.powi(2);
        let q = y + (point.re - 0.25_f64).powi(2);
        q * (q + point.re - 0.25_f64) > (y * 0.25_f64) && (point.re + 1.0_f64).powi(2) + y > 0.0625_f64
    }

    /// This is our classic iterator function, which either returns the
    /// number of iterations it took to escape the Mandelbrot set, or
    /// it returns nothing at all.
    pub fn iterate_sample(&self, point: &Complex<f64>, max_iterations: usize) -> Option<usize> {
        let mut z = Complex{ re: 0.0_f64, im: 0.0_f64 };
        for i in 0..max_iterations {
            z = z * z + point;
            if z.norm_sqr() >= 4.0_f64 {
                return Some(i);
            }
        }
        None
    }

    /// Given a collection of possibly interesting points, now revisit
    /// them to find out which ones really are in the Buddhabrot and
    /// when those leave the Mandelbrot set.
    pub fn collect_samples(&mut self, points: &[Complex<f64>]) -> Vec<(Complex<f64>, usize)> {
        let mut samples: Vec<(Complex<f64>, usize)> = vec![];
        for point in points {
            let c = Complex{ re: point.re + self.rng.gen_range(0.0_f64, self.cell_size),
                             im: point.im + self.rng.gen_range(0.0_f64, self.cell_size) };
            if Self::maybe_outside(c) {
                if let Some(i) = self.iterate_sample(point, self.max_iterations) {
                    if i > self.min_iterations {
                        samples.push((c, i));
                    }
                }
            }
        }
        samples
    }

    /// Given the final collection of points, map them to the plane.
    pub fn map_samples(&self, points: &[(Complex<f64>, usize)]) -> Result<Vec<u32>, String> {
        let mut plane = vec![0 as u32; self.planes.len()];
        for point in points {
            let mut z = Complex{ re: 0.0_f64, im: 0.0_f64 };
            for i in 0..point.1 {
                z = z * z + point.0;
                if i > self.min_plot_count {
                    if let Some(offset) = self.planes.point_to_offset(&z) {
                        plane[offset] += 1;
                    }
                }
            }
        }
        Ok(plane)
    }

    /// The main function, and primary entry point.
    pub fn buddhabrot(&mut self, _threads: usize) -> Result<Vec<u32>, String> {
        let points = self.find_interesting_points();
        let valid_points = self.collect_samples(&points);
        self.map_samples(&valid_points)
    }
}
    
    
        
