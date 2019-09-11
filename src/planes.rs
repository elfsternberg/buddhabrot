//! Contains the PlaneMapper struct, which describes a relationship
//! between a rectangle on the integral plane with an origin at 0,0,
//! and a rectangle on the real plane with an arbitrary pair of
//! corners defining the leftlower and rightupper corners of the real
//! plane.
use num::Complex;

/// Describes the width and height of an integral plane that is assumed to start at
/// 0,0 and all values are assumed to be non-negative integers.  For that reason,
/// the lower-left-hand corner is not included.  
#[derive(Copy, Clone, Debug)]
pub struct IntegralPlane(pub usize, pub usize);

/// Describes the lower-left corner and upper-right corner of the
/// Complex plane, treating the real part of each value as the
/// x-component and the imaginary part of each value as the
/// y-component.
#[derive(Copy, Clone, Debug)]
pub struct ComplexPlane(pub Complex<f64>, pub Complex<f64>);

/// Describes the x, y of a point in a region.  Yes, it's the exact
/// same. Names are important.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Pixel(pub usize, pub usize);

/// We don't need a Point, as a single Complex number is a Point.

/// Contains the definitions of two planes: an integral cartesian plane,
/// and a complex, real cartesian plane.  Maps points from one to the
/// other.  'leftupper' may seem ungrammatical, but it fits with our
/// x,y schema.

#[derive(Debug)]
pub struct PlaneMapper {
    /// The right-upper hand corner of the integral cartesian plane.
    /// The left-lower is assumed to be at 0,0
    pub integral_plane: IntegralPlane,
    /// The two coordinates defining the complex cartesian plane,
    /// left-lower and right-upper
    pub complex_plane: ComplexPlane,
    // The ratio mapping the width and height, respectively, of the two
    // different planes.
    grid_factors: (f64, f64),
}

impl PlaneMapper where {
    /// Constructor.  Takes a region describing the integral plane, and
    /// two points describing the complex plane.  Has function to map
    /// points inside one to points inside the other.
    pub fn new(
        width: usize,
        height: usize,
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
            (width as f64) / region_width,
            (height as f64) / region_height,
        );

        Ok(PlaneMapper {
            integral_plane: IntegralPlane(width, height),
            complex_plane: ComplexPlane(leftlower, rightupper),
            grid_factors,
        })
    }

    /// The total number of points in the integral grid.  Used to
    /// calculate memory needs.
    pub fn len(&self) -> usize {
        self.integral_plane.0 * self.integral_plane.1
    }

    /// Describes that the integral plane is of a size.
    pub fn is_empty(&self) -> bool {
        self.integral_plane.0 == 0 || self.integral_plane.1 == 0
    }

    /// Given a complex number corresponding to a location on the
    /// complex cartesian plane, map that as closely as possible to a
    /// point on the integral cartesian plane.
    pub fn point_to_pixel(&self, point: &Complex<f64>) -> Pixel {
        let left = (point.re - self.complex_plane.0.re) * self.grid_factors.0;
        let top = (point.im - self.complex_plane.0.im) * self.grid_factors.1;
        Pixel(left as usize, top as usize)
    }

    /// Given a pixel on the integral cartesian plane, map that as
    /// closely as possible to a point on the complex cartesian plane.
    pub fn pixel_to_point(&self, pixel: &Pixel) -> Complex<f64> {
        Complex::new(
            ((pixel.0 as f64) / self.grid_factors.0) + self.complex_plane.0.re,
            ((pixel.1 as f64) / self.grid_factors.1) + self.complex_plane.0.im,
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
        let left = (point.re - self.complex_plane.0.re) * self.grid_factors.0;
        let top = (point.im - self.complex_plane.0.im) * self.grid_factors.1;
        if left < 0.0
            || left > (self.integral_plane.0 as f64)
            || top < 0.0
            || top > (self.integral_plane.1 as f64)
        {
            return None;
        }
        Some((top as usize) * self.integral_plane.0 + (left as usize))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planemapper_fails_on_bad_shape() {
        let pm = PlaneMapper::new(4, 4, Complex::new(-1.0, 1.0), Complex::new(1.0, -1.0));
        assert!(pm.is_err());
    }

    #[test]
    fn planemapper_passes_on_good_shape() {
        let pm = PlaneMapper::new(4, 4, Complex::new(-1.0, -1.0), Complex::new(1.0, 1.0));
        assert!(pm.is_ok());
    }

    #[test]
    fn point_to_pixel_on_positive_planes() {
        let pm = PlaneMapper::new(5, 5, Complex::new(0.0, 0.0), Complex::new(5.0, 5.0)).unwrap();
        println!("{:?}", pm);
        assert_eq!(pm.point_to_pixel(&Complex::new(0.0, 0.0)), Pixel(0, 0));
        assert_eq!(pm.point_to_pixel(&Complex::new(2.0, 2.0)), Pixel(2, 2));
        assert_eq!(pm.point_to_pixel(&Complex::new(4.0, 4.0)), Pixel(4, 4));
    }

    #[test]
    fn point_to_pixel_on_mixed_planes() {
        let pm = PlaneMapper::new(4, 4, Complex::new(-2.0, -2.0), Complex::new(2.0, 2.0)).unwrap();
        assert_eq!(pm.point_to_pixel(&Complex::new(0.0, 0.0)), Pixel(2, 2));
        assert_eq!(pm.point_to_pixel(&Complex::new(-2.0, -2.0)), Pixel(0, 0));
        assert_eq!(pm.point_to_pixel(&Complex::new(2.0, 2.0)), Pixel(4, 4));
    }

    #[test]
    fn point_to_pixel_maps_on_large_mixed_planes() {
        let pm =
            PlaneMapper::new(640, 640, Complex::new(-2.0, -2.0), Complex::new(2.0, 2.0)).unwrap();
        assert_eq!(pm.point_to_pixel(&Complex::new(0.0, 0.0)), Pixel(320, 320));
        assert_eq!(pm.point_to_pixel(&Complex::new(-2.0, -2.0)), Pixel(0, 0));
        assert_eq!(pm.point_to_pixel(&Complex::new(2.0, 2.0)), Pixel(640, 640));
        assert_eq!(pm.point_to_pixel(&Complex::new(1.0, 2.0)), Pixel(480, 640));
    }

    #[test]
    fn pixel_to_point_on_positive_planes() {
        let pm = PlaneMapper::new(5, 5, Complex::new(0.0, 0.0), Complex::new(5.0, 5.0)).unwrap();
        assert_eq!(pm.pixel_to_point(&Pixel(0, 0)), Complex::new(0.0, 0.0));
        assert_eq!(pm.pixel_to_point(&Pixel(2, 2)), Complex::new(2.0, 2.0));
        assert_eq!(pm.pixel_to_point(&Pixel(4, 4)), Complex::new(4.0, 4.0));
    }

    #[test]
    fn pixel_to_points_on_mixed_planes() {
        let pm = PlaneMapper::new(4, 4, Complex::new(-2.0, -2.0), Complex::new(2.0, 2.0)).unwrap();
        assert_eq!(pm.pixel_to_point(&Pixel(2, 2)), Complex::new(0.0, 0.0));
        assert_eq!(pm.pixel_to_point(&Pixel(0, 0)), Complex::new(-2.0, -2.0));
        assert_eq!(pm.pixel_to_point(&Pixel(4, 4)), Complex::new(2.0, 2.0));
    }

}
