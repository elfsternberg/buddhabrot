extern crate clap;
extern crate image;
extern crate num;
extern crate num_cpus;

extern crate buddhabrot;
use buddhabrot::NaiveRenderer;
use buddhabrot::{cupe_buddhabrot, CupeRenderer};
use buddhabrot::lispy::find_interesting_points;
use clap::{App, Arg, ArgMatches};
use image::pnm::PNMEncoder;
use image::pnm::{PNMSubtype, SampleEncoding};
use image::png::PNGEncoder;
use image::ColorType;
use num::{clamp, Complex};
use std::fs::File;
use std::path::Path;
use std::str::FromStr;

fn parse_pair<T>(s: &str, separator: char) -> Option<(T, T)>
where
    T: FromStr,
{
    match s.find(separator) {
        None => None,
        Some(index) => match (T::from_str(&s[..index]), T::from_str(&s[index + 1..])) {
            (Ok(l), Ok(r)) => Some((l, r)),
            _ => None,
        },
    }
}

fn parse_complex(s: &str) -> Option<Complex<f64>> {
    match parse_pair(s, ',') {
        Some((re, im)) => Some(Complex { re, im }),
        None => None,
    }
}

fn validate_pair<T: FromStr>(s: &str, separator: char, err: &str) -> Result<(), String> {
    match parse_pair::<T>(s, separator) {
        Some(_) => Ok(()),
        None => Err(err.to_string()),
    }
}

fn validate_range<T: FromStr + Ord>(
    s: &str,
    low: T,
    high: T,
    isnotanumber_err: &str,
    isnotinrange_err: &str,
) -> Result<(), String> {
    match T::from_str(s) {
        Ok(i) => {
            if i >= low && i <= high {
                Ok(())
            } else {
                Err(isnotinrange_err.to_string())
            }
        }
        Err(_) => Err(isnotanumber_err.to_string()),
    }
}

const OUTPUT: &str = "output";
const SIZE: &str = "size";
const LEFTLOWER: &str = "leftlower";
const RIGHTUPPER: &str = "rightupper";
const THREADS: &str = "threads";
const ITERATIONS: &str = "iterations";
const MAXTHREADS: &str = "maxthreads";

fn args<'a>() -> ArgMatches<'a> {
    let max_threads = num_cpus::get();

    App::new("buddha")
        .version("0.2.0")
        .author("Elf M. Sternberg <elf.sternberg@gmail.com>")
        .about("Buddhabrot renderer")
        .arg(
            Arg::with_name(OUTPUT)
                .required(true)
                .long(OUTPUT)
                .short("o")
                .takes_value(true)
                .help("Output file"),
        )
        .arg(
            Arg::with_name(SIZE)
                .required(false)
                .long(SIZE)
                .short("s")
                .takes_value(true)
                .default_value("800x600")
                .validator(|s| validate_pair::<u16>(&s, 'x', "Could not parse output image size"))
                .help("Size of output image"),
        )
        .arg(
            Arg::with_name(LEFTLOWER)
                .required(false)
                .long(LEFTLOWER)
                .short("l")
                .takes_value(true)
                .default_value("-2.103,-1.238")
                .validator(|s| validate_pair::<f64>(&s, ',', "Could not parse left upper corner"))
                .help("Left upper corner of the mandelbrot space"),
        )
        .arg(
            Arg::with_name(RIGHTUPPER)
                .required(false)
                .long(RIGHTUPPER)
                .short("r")
                .takes_value(true)
                .default_value("1.201,1.240")
                .validator(|s| validate_pair::<f64>(&s, ',', "Could not parse right lower corner"))
                .help("Right lower corner of the mandelbrot space"),
        )
        .arg(
            Arg::with_name(THREADS)
                .required(false)
                .long(THREADS)
                .short("t")
                .takes_value(true)
                .default_value("1")
                .validator(move |s| {
                    validate_range(
                        &s,
                        1,
                        max_threads,
                        "Could not parse thread count",
                        &format!("Thread count must be between 1 and {}", max_threads),
                    )
                })
                .help("Number of threads to use in solver"),
        )
        .arg(
            Arg::with_name(MAXTHREADS)
                .required(false)
                .long(MAXTHREADS)
                .short("m")
                .takes_value(false)
                .required(false)
                .help("Maxthreads: use all available cores."),
        )
        .arg(
            Arg::with_name(ITERATIONS)
                .required(false)
                .long(ITERATIONS)
                .short("i")
                .takes_value(true)
                .default_value("2000")
                .validator(move |s| {
                    validate_range(
                        &s,
                        250,
                        200_000,
                        "Could not parse iteration count",
                        "Iteration count must be between 250 and 200000",
                    )
                })
                .help("Number of threads to use in solver"),
        )
        .get_matches()
}

fn write_image(outfile: &str, pixels: &[u8], bounds: (usize, usize)) -> Result<(), std::io::Error> {
    let path = Path::new(outfile);
    let output = File::create(&path).unwrap();
    let mut encoder =
        PNMEncoder::new(output).with_subtype(PNMSubtype::Graymap(SampleEncoding::Binary));
    encoder.encode(pixels, bounds.0 as u32, bounds.1 as u32, ColorType::Gray(8))?;
    Ok(())
}

fn main() {
    let matches = args();
    let image_size =
        parse_pair(matches.value_of(SIZE).unwrap(), 'x').expect("Error parsing image dimensions");
    let leftlower = parse_complex(matches.value_of(LEFTLOWER).unwrap())
        .expect("Error parsing left upper point");
    let rightupper = parse_complex(matches.value_of(RIGHTUPPER).unwrap())
        .expect("Error parsing right lower point");
    let iterations = usize::from_str(matches.value_of(ITERATIONS).unwrap())
        .expect("Could not parse iteration_count.");
    let threads = if matches.is_present(MAXTHREADS) {
        num_cpus::get()
    } else {
        usize::from_str(matches.value_of(THREADS).unwrap()).expect("Could not parse thread count.")
    };

    let buddha = CupeRenderer::new(image_size.0, image_size.1, leftlower, rightupper)
        .expect("Initialization error");

    let mut plane = vec![0 as u8; buddha.planes.len()];
    let points = find_interesting_points(&buddha, 8);
    for point in points {
        if let Some(offset) = buddha.planes.point_to_offset(&point) {
            plane[offset] = 127;
        }
    }

    let path = Path::new(matches.value_of(OUTPUT).unwrap());
    let output = File::create(&path).unwrap();
    let mut encoder = PNGEncoder::new(output);
    encoder.encode(&plane, image_size.0 as u32, image_size.1 as u32, ColorType::Gray(8)).unwrap();
    
/*
    match cupe_buddhabrot(&buddha, threads) {
        Err(e) => {
            eprintln!("Render failure: {}", e);
            std::process::exit(1);
        }
        Ok(raw) => {
            println!("{:?}", raw);
            let maxi = *raw.iter().max().unwrap();
            let bbias = (0.045 * (maxi as f32)) as f64;
            let tbias = (maxi as f64) - bbias;
            let m = 256.0 / ((tbias - bbias) as f64);
            println!("M: {}", m);
            
            let nraw: Vec<u8> = raw
                .iter()
                .map(|s| {
                    let c = ((*s as f64) * m - bbias) as u32;
                    clamp(c, 0, 255) as u8
                })
                .collect();
            write_image(
                matches.value_of(OUTPUT).unwrap(),
                &nraw,
                (image_size.0, image_size.1),
            )
            .unwrap();
        }
    }
*/
}
