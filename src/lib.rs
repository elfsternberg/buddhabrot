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
extern crate num;
extern crate num_cpus;
extern crate itertools;

use num::{clamp, Complex};
use std::ops::{Index, IndexMut};

/// A plane is an array with methods that map a coordinate pair to a
/// location on the array.  The array is left exposed so that we can
/// do min/max analysis on it later.

// In all cases, the order is Width (x-axis), Height (y-axis).

#[derive(Debug)]
struct Plane<T>(pub usize, pub usize, pub Vec<T>);

impl<T> Plane<T> {
    // A linearized plane shall be indexed by (y * width) + x; the
    // width dictates the size of the row, so we skip rows by indexing
    // `y` many widths, and then index into the row by `x`.  #academic
    #[inline]
    fn at(&self, index: (usize, usize)) -> usize {
        self.0 * index.1 + index.0
    }
}

impl<T> Index<(usize, usize)> for Plane<T> {
    type Output = T;
    fn index(&self, index: (usize, usize)) -> &Self::Output {
        &self.2[self.at(index)]
    }
}

impl<T> IndexMut<(usize, usize)> for Plane<T> {
    fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
        let loc = self.at(index);
        &mut self.2[loc]
    }
}

/// Describes the width and height of a region.
#[derive(Copy, Clone, Debug)]
pub struct Region<T>(pub T, pub T);

/// Describes the x, y of a point in a region.  Yes, it's the exact
/// same. Names are important.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pixel(pub usize, pub usize);

/// Contains the definitions of two planes: an integral cartesian plane,
/// and a complex, real cartesian plane.  Maps points from one to the
/// other.  'leftupper' may seem ungrammatical, but it fits with our
/// x,y schema.
#[derive(Debug)]
pub struct PlaneMapper {
    plane: Region<usize>,
    leftlower: Complex<f64>,
    grid_factors: (f64, f64),
}

impl PlaneMapper where {
    /// Constructor.  Takes a region describing the integral plane, and
    /// two points describing the complex plane.  Has function to map
    /// points inside one to points inside the other.
    pub fn new(
        plane: Region<usize>,
        leftlower: Complex<f64>,
        rightupper: Complex<f64>,
    ) -> Result<PlaneMapper, String> {
        if rightupper.re < leftlower.re {
            return Err(
                "The left lower corner is not to the left of the right upper corner.".to_string(),
            );
        }

        if rightupper.im < leftlower.im {
            return Err(
                "The left lower corner is not lower than the right upper corner".to_string(),
            );
        }

        // The total size of the region.
        let region_width = rightupper.re - leftlower.re;
        let region_height = rightupper.im - leftlower.im;

        // The relationship of a given point in the real plane to the
        // cartesian plane.  Multiply the elements of a complex number
        // by these, and floor(), to get the coordinates of the grid.
        // let xscale = ((plane.0 as f64) - 1.0) / region_width;
        // let yscale = ((plane.1 as f64) - 1.0) / region_height;

        // these are the multipliers of the complex plane to the real plane.
        let grid_factors = (
            (plane.0 as f64) / region_width,
            (plane.1 as f64) / region_height,
        );

        Ok(PlaneMapper {
            plane,
            leftlower,
            grid_factors,
        })
    }

    /// The total number of points in the cartesian grid.  Used to
    /// calculate a lot of different memory needs.
    pub fn len(&self) -> usize {
        self.plane.0 * self.plane.1
    }

    /// The total number of points in the cartesian grid.  Used to
    /// calculate a lot of different memory needs.
    pub fn is_empty(&self) -> bool {
        self.plane.0 == 0 || self.plane.1 == 0
    }

    /// Given a complex number corresponding to a location on the
    /// complex cartesian plane, map that as closely as possible to a
    /// point on the integral cartesian plane.
    pub fn point_to_pixel(&self, point: &Complex<f64>) -> Pixel {
        let left = (point.re - self.leftlower.re) * self.grid_factors.0;
        let top = (point.im - self.leftlower.im) * self.grid_factors.1;
        Pixel(left as usize, top as usize)
    }

    /// Given a pixel on the integral cartesian plane, map that as
    /// closely as possible to a point on the complex cartesian plane.
    pub fn pixel_to_point(&self, pixel: &Pixel) -> Complex<f64> {
        Complex::new(
            ((pixel.0 as f64) / self.grid_factors.0) + self.leftlower.re,
            ((pixel.1 as f64) / self.grid_factors.1) + self.leftlower.im,
        )
    }

    /// Since the Buddhabrot actually tracks the progress of a complex
    /// number as it orbits the Mandelbrot set's interior, we have to
    /// map those complex numbers back to the pixel plane, and then
    /// increment those points on the pixel plane as the orbit passes
    /// through them.  This function takes a point, maps it to pixel
    /// coordinates, then returns the linear offset from the root of
    /// the image buffer in memory.
    pub fn point_to_offset(&self, point: &Complex<f64>) -> Option<usize> {
        let left = (point.re - self.leftlower.re) * self.grid_factors.0;
        let top = (point.im - self.leftlower.im) * self.grid_factors.1;
        if left < 0.0 || left > (self.plane.0 as f64) || top < 0.0 || top > (self.plane.1 as f64) {
            return None;
        }
        Some((top as usize) * self.plane.0 + (left as usize))
    }
}

/// This is the 'primary' helper function, in that its purpose is to
/// take a point, a plane, and a buffer, and plot the orbit of that
/// point up to some limit, either some max number, or the depth until
/// transition outside the black heart.
fn plot(start: Complex<f64>, buffer: &mut [u32], plane: &PlaneMapper, limit: usize) {
    let mut z: Complex<f64> = Complex { re: 0.0, im: 0.0 };
    for _ in 0..limit {
        z = z * z + start;
        if let Some(offset) = plane.point_to_offset(&z) {
            buffer[offset] += 1;
        }
    }
}

// The main function: Given a buffer and a plane, map the buddhabrot set.
fn render(
    buffer: &mut [u32],
    plane: &PlaneMapper,
    zone: (usize, usize),
    limit: usize,
) -> Result<(), String> {
    if buffer.len() < plane.plane.0 * plane.plane.1 {
        return Err("Buffer size not large enough to hold requested plane.".to_string());
    }
    for column in zone.0..zone.1 {
        for row in 0..plane.plane.1 {
            let mut z: Complex<f64> = Complex { re: 0.0, im: 0.0 };
            let p = Pixel(column, row);
            let c = plane.pixel_to_point(&p);
            for i in 0..limit {
                z = z * z + c;
                if z.norm_sqr() >= 4.0 {
                    plot(c, buffer, plane, i);
                    break;
                }
            }
        }
    }
    Ok(())
}

fn render_merge(regions: &[u32], plane_size: usize) -> Vec<u32> {
    let mut ret = vec![0 as u32; plane_size];
    let regions: Vec<&[u32]> = regions.chunks(plane_size).collect();
    for i in 0..ret.len() {
        for region in &regions {
            ret[i] += region[i];
        }
    }
    ret
}

/// A single-threaded version of the buddhabrot renderer
pub fn buddhabrot(plane: &PlaneMapper, limit: usize) -> Result<Vec<u32>, String> {
    let mut buffer = vec![0 as u32; plane.len()];
    render(&mut buffer, plane, (0, plane.plane.0), limit).unwrap();
    Ok(buffer)
}

/// A multi-threaded version of the render function that takes a thread count
/// as an option.
pub fn buddhabrot_threaded(
    plane: &PlaneMapper,
    limit: usize,
    threads: usize,
) -> Result<Vec<u32>, String> {
    let mut regions = vec![0 as u32; plane.len() * threads];

    let zones: Vec<(usize, usize)> = {
        let zone_interval = plane.plane.0 / threads + (plane.plane.0 % threads);
        (0..threads).map(|i| ((zone_interval * i), clamp(zone_interval * (i + 1), 0, plane.plane.0))).collect()
    };

    {
        let regions: Vec<&mut [u32]> = regions.chunks_mut(plane.len()).collect();
        crossbeam::scope(|spawner| {
            for (zone, region) in itertools::zip(zones, regions) {
                spawner.spawn(move |_| {
                    render(region, plane, zone, limit).unwrap();
                });
            }
        })
        .unwrap();
    }

    Ok(render_merge(&regions, plane.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planemapper_fails_on_bad_shape() {
        let pm = PlaneMapper::new(
            Region(4, 4),
            Complex::new(-1.0, 1.0),
            Complex::new(1.0, -1.0),
        );
        assert!(pm.is_err());
    }

    #[test]
    fn planemapper_passes_on_good_shape() {
        let pm = PlaneMapper::new(
            Region(4, 4),
            Complex::new(-1.0, -1.0),
            Complex::new(1.0, 1.0),
        );
        assert!(pm.is_ok());
    }

    #[test]
    fn point_to_pixel_on_positive_planes() {
        let pm =
            PlaneMapper::new(Region(5, 5), Complex::new(0.0, 0.0), Complex::new(5.0, 5.0)).unwrap();
        println!("{:?}", pm);
        assert_eq!(pm.point_to_pixel(&Complex::new(0.0, 0.0)), Pixel(0, 0));
        assert_eq!(pm.point_to_pixel(&Complex::new(2.0, 2.0)), Pixel(2, 2));
        assert_eq!(pm.point_to_pixel(&Complex::new(4.0, 4.0)), Pixel(4, 4));
    }

    #[test]
    fn point_to_pixel_on_mixed_planes() {
        let pm = PlaneMapper::new(
            Region(4, 4),
            Complex::new(-2.0, -2.0),
            Complex::new(2.0, 2.0),
        )
        .unwrap();
        assert_eq!(pm.point_to_pixel(&Complex::new(0.0, 0.0)), Pixel(2, 2));
        assert_eq!(pm.point_to_pixel(&Complex::new(-2.0, -2.0)), Pixel(0, 0));
        assert_eq!(pm.point_to_pixel(&Complex::new(2.0, 2.0)), Pixel(4, 4));
    }

    #[test]
    fn point_to_pixel_maps_on_large_mixed_planes() {
        let pm = PlaneMapper::new(
            Region(640, 640),
            Complex::new(-2.0, -2.0),
            Complex::new(2.0, 2.0),
        )
        .unwrap();
        assert_eq!(pm.point_to_pixel(&Complex::new(0.0, 0.0)), Pixel(320, 320));
        assert_eq!(pm.point_to_pixel(&Complex::new(-2.0, -2.0)), Pixel(0, 0));
        assert_eq!(pm.point_to_pixel(&Complex::new(2.0, 2.0)), Pixel(640, 640));
        assert_eq!(pm.point_to_pixel(&Complex::new(1.0, 2.0)), Pixel(480, 640));
    }

    #[test]
    fn pixel_to_point_on_positive_planes() {
        let pm =
            PlaneMapper::new(Region(5, 5), Complex::new(0.0, 0.0), Complex::new(5.0, 5.0)).unwrap();
        assert_eq!(pm.pixel_to_point(&Pixel(0, 0)), Complex::new(0.0, 0.0));
        assert_eq!(pm.pixel_to_point(&Pixel(2, 2)), Complex::new(2.0, 2.0));
        assert_eq!(pm.pixel_to_point(&Pixel(4, 4)), Complex::new(4.0, 4.0));
    }

    #[test]
    fn pixel_to_points_on_mixed_planes() {
        let pm = PlaneMapper::new(
            Region(4, 4),
            Complex::new(-2.0, -2.0),
            Complex::new(2.0, 2.0),
        )
        .unwrap();
        assert_eq!(pm.pixel_to_point(&Pixel(2, 2)), Complex::new(0.0, 0.0));
        assert_eq!(pm.pixel_to_point(&Pixel(0, 0)), Complex::new(-2.0, -2.0));
        assert_eq!(pm.pixel_to_point(&Pixel(4, 4)), Complex::new(2.0, 2.0));
    }

}
