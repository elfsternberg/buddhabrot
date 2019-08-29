extern crate crossbeam;
extern crate image;
extern crate num;
extern crate num_cpus;

use image::pnm::PNMEncoder;
use image::pnm::{PNMSubtype, SampleEncoding};
use image::ColorType;
use num::Complex;
use std::fs::File;
use std::str::FromStr;

/// Given a string and a separator, returns the two values
/// separated by the separator.
fn parse_pair<T: FromStr>(s: &str, separator: char) -> Option<(T, T)> {
    match s.find(separator) {
        None => None,
        Some(index) => match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
            (Ok(l), Ok(r)) => Some((l, r)),
            _ => None,
        },
    }
}

/// A specific implementation of parse_pair using a comma and expecting
/// floating point numbers.
fn parse_complex(s: &str) -> Option<Complex<f64>> {
    match parse_pair(s, ',') {
        Some((re, im)) => Some(Complex { re, im }),
        None => None,
    }
}

/// Calculate the distance between two complex numbers on a Cartesian
/// plane.
#[inline]
pub fn dist(i: Complex<f64>, j: Complex<f64>) -> f64 {
    ((i.im - j.im) * (i.im - j.im) + (i.re - j.re) * (i.re - j.re))
}

#[derive(Copy, Clone, Debug)]
struct Region<T> {
    width: T,
    height: T,
}

#[derive(Copy, Clone)]
struct Pixel {
    left: usize,
    top: usize,
}

#[derive(Debug)]
struct PlaneMapper {
    bounds: Region<usize>,
    complex_bounds: Region<f64>,
    origin: Complex<f64>,
}

/// Contains the definitions of two planes: an integral cartesian plane,
/// and a complex cartesian plane.  Maps points from one to the other.
impl PlaneMapper where {
    pub fn new(bounds: Region<usize>, ul: Complex<f64>, lr: Complex<f64>) -> PlaneMapper {
        PlaneMapper {
            bounds,
            complex_bounds: Region {
                width: (lr.re - ul.re) / (bounds.width as f64),
                height: (lr.im - ul.im) / (bounds.height as f64),
            },
            origin: ul,
        }
    }

    /// Given the row and column of a pixel on the integral cartesian plane,
    /// return an complex number that corresponds to the equivalent location
    /// mapped to the complex cartesian plane.
    pub fn pixel_to_point(&self, pixel: Pixel) -> Complex<f64> {
        Complex {
            re: self.origin.re + (pixel.left as f64) * self.complex_bounds.width,
            im: self.origin.im + (pixel.top as f64) * self.complex_bounds.height,
        }
    }

    /// Given a complex number corresponding to a location on the
    /// complex cartesian plane, map that as closely as possible to a
    /// point on the integral cartesian plane.
    pub fn point_to_pixel(&self, point: &Complex<f64>) -> Pixel {
        let left = (point.re - self.origin.re) / self.complex_bounds.width;
        let top = (point.im - self.origin.im) / self.complex_bounds.height;
        Pixel {
            left: (left as usize),
            top: (top as usize),
        }
    }

    /// Since the Buddhabrot actually tracks the progress of a complex
    /// number as it orbits the Mandelbrot set's interior, we have to
    /// map those complex numbers back to the pixel plane, and then
    /// increment those points on the pixel plane as the orbit passes
    /// through them.  This function takes a point, maps it to pixel
    /// coordinates, then returns the linear offset from the root of
    /// the image buffer in memory.
    pub fn point_to_offset(&self, point: &Complex<f64>) -> usize {
        let p = self.point_to_pixel(point);
        p.top * self.bounds.width + p.left
    }

    /// Given a buffer and a point, plot the orbits of that point,
    /// incrementing the pixel points the orbit passes through as we
    /// do so.
    pub fn plot(&self, pixels: &mut [u16], c: Complex<f64>, limit: usize) {
        let mut z: Complex<f64> = Complex { re: 0.0, im: 0.0 };
        let max_offset = pixels.len();
        for _ in 0..limit {
            z = z * z + c;
            if z.norm_sqr() >= 4.0 {
                break;
            }
            let offset = self.point_to_offset(&z);
            if offset < max_offset {
                pixels[self.point_to_offset(&z)] += 1;
            }
        }
    }

    /// Given a buffer, plot the orbits of every point in the plane
    pub fn render(&self, pixels: &mut [u16], limit: usize) {
        assert!(pixels.len() == self.bounds.width * self.bounds.height);
        for row in 0..self.bounds.height {
            for column in 0..self.bounds.width {
                let mut z: Complex<f64> = Complex { re: 0.0, im: 0.0 };
                let p = Pixel {
                    left: column,
                    top: row,
                };
                let c = self.pixel_to_point(p);
                for _i in 0..limit {
                    z = z * z + c;
                    if z.norm_sqr() >= 4.0 {
                        self.plot(pixels, c, limit);
                        break;
                    }
                }
            }
        }
    }
}

fn write_image(
    filename: &str,
    pixels: &[u8],
    bounds: (usize, usize),
) -> Result<(), std::io::Error> {
    let output = File::create(filename)?;
    let mut encoder =
        PNMEncoder::new(output).with_subtype(PNMSubtype::Graymap(SampleEncoding::Binary));
    encoder.encode(pixels, bounds.0 as u32, bounds.1 as u32, ColorType::Gray(8))?;
    Ok(())
}

fn pixelate(region: &mut Vec<u16>, pixels: &mut Vec<u8>) {
    for (i, ref mut p) in pixels.into_iter().enumerate() {
        let v = (u16::from(**p) + region[i]) % 255;
        **p = v as u8;
    }
}

pub fn main() {
    let args: Vec<String> = std::env::args().collect();

    let threads = num_cpus::get();
    let limit = 120000;

    if args.len() != 5 {
        eprintln!("Usage: buddhabrot FILE PIXELS UPPERLEFT LOWERRIGHT");
        eprintln!("Exaple: {} mandel.png 1000x750 -1.20,0.35 -1,02.0", args[0]);
        std::process::exit(1);
    }

    let bounds = parse_pair(&args[2], 'x').expect("Error parsing image dimensions");
    let upper_left = parse_complex(&args[3]).expect("Error parsing upper left hand point");
    let lower_right = parse_complex(&args[4]).expect("Error parsing lower right hand point");

    let plane_size = bounds.0 * bounds.1;

    let mut regions = vec![0 as u16; plane_size * threads];
    let plane = PlaneMapper::new(
        Region {
            width: bounds.0,
            height: bounds.1,
        },
        upper_left,
        lower_right,
    );

    // Creates multiple pixel planes, one for each possible thread, and
    // renders multiple copies of the buddhabrot.
    {
        let zonesize = (bounds.1 / threads) + 1;
        let mut start = 0;
        let regions: Vec<&mut [u16]> = regions.chunks_mut(plane_size).collect();
        let plane = &plane;
        crossbeam::scope(|spawner| {
            for region in regions {
                spawner.spawn(move || {
                    plane.render(region, limit);
                });
                start += zonesize;
            }
        });
    }

    // Assemble all of the pixel planes into a single plane, and render.
    let mut pixels = vec![0 as u8; plane_size];
    {
        let mut assem = vec![0 as u16; plane_size];
        let regions: Vec<&mut [u16]> = regions.chunks_mut(plane_size).collect();
        for region in regions {
            for (i, pixel) in region.iter().enumerate() {
                assem[i] += pixel;
            }
        }
        pixelate(&mut assem, &mut pixels);
    }

    write_image("buddha.pnm", &pixels, bounds).unwrap();
}
