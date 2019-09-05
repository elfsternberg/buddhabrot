extern crate clap;
extern crate image;
extern crate num;
extern crate num_cpus;

use clap::{App, Arg, ArgMatches};
use image::pnm::PNMEncoder;
use image::pnm::{PNMSubtype, SampleEncoding};
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
    let image_size = buddhabrot::Region(image_size.0, image_size.1);
    let plane = buddhabrot::PlaneMapper::new(image_size, leftlower, rightupper).unwrap();
    let threads =
        usize::from_str(matches.value_of(THREADS).unwrap()).expect("Could not parse thread count.");
    let iterations = usize::from_str(matches.value_of(ITERATIONS).unwrap())
        .expect("Could not parse thread count.");

    match buddhabrot::buddhabrot_threaded(&plane, iterations, threads) {
        Err(e) => {
            eprintln!("Render failure: {}", e);
            std::process::exit(1);
        }
        Ok(raw) => {
            let maxi = *raw.iter().max().unwrap();
            let bias = (0.045 * (maxi as f32)) as u32;
            let nraw: Vec<u8> = raw
                .iter()
                .map(|mut s| {
                    s = if *s < bias { &0 } else { s };
                    clamp((s * 256) / maxi, 0, 255) as u8
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
}
