# Buddhabrot

![Language: Rust](https://img.shields.io/badge/language-Rust-green)
![Topic: Study](https://img.shields.io/badge/topic-Study-red)
![Topic: Graphics](https://img.shields.io/badge/topic-Graphics-red)

![Buddhabrot](https://github.com/elfsternberg/buddhabrot/raw/master/buddha.jpg)

The Buddhabrot is a relative of the Mandelbrot image.  In O'Reilly's
[*Programming Rust*](http://shop.oreilly.com/product/0636920040385.do),
the exercise calls for writing a Mandelbrot generator but as with
everything it's not enough to just type in the text from the book, to
understand it I have to make something new out of it.

## Buddhabrot Math

The Buddhabrot answers the question, "What's really inside that massive
black heart in the center of the traditional Mandelbrot?"  

### Mandelbrot Math

The math of the Mandelbrot is straightforward, if a bit wibbly.  Start
with a complex number of the form <code>a + b<i>i</i></code>, where
<i>i</i> is the square root of -1.  Just as natural and real numbers
belong to a linear space (negative infinity to 0 to positive infinity),
complex numbers take up a planar space: `a` and `b` are coordinates on a
two-dimensional grid.

Given a function <code>f(z) = z<sup>2</sup> + c</code>, where `z` and
`c` are complex numbers, and starting with a location `c`, repeatedly
call `f()` with the results of the previous iteration.  The result may
go to infinity or it may not.  If it does not go to infinity, color it
black.  If it does go to infinity, color it with the velocity with which
it went there; how *soon* does it go to infinity?  The classic
Mandelbrot is grey-scale, but using a well-chosen color palette can
result in some interesting images.

### Buddhabrot Math

In the center of the Mandlebrot is a large, vaguely heart-shaped zone of
black.  Melinda Green [wondered if the iterative process could be used
elsewise](
https://groups.google.com/forum/?hl=en#!msg/sci.fractals/PNOBmN_zpPg/TXorwQukkbgJ).
In her formulation, each iteration produces a new complex number, which
can *also be treated as a new coordinate pair*.  By steadily tracing the
"path" any given point takes from its starting point, and incrementing
each of the points it lands on by one, the Buddhabrot algorithm slowly
builds an image inside the Mandelbrot's black heart.  If you squint hard
enough, it looks like a sitting Buddha, hence its name.

## Implementation

This implementation is much heavier than the one for the Mandelbrot, at
least in terms of memory.  The Buddhabrot cannot be sliced the way the
Mandelbrot can; while we can choose different regions of memory from
which to start, the Buddhabrot "traces" through the entire region we
mean to illustrate, so each thread must have a Vec big enough to contain
the entire board.

That said, it's a pretty good implementation.  It does what I want it to
do, and it's not *too* slow.

## TODO

- Look into the [`faster`](https://github.com/AdamNiederer/faster) crate
- Use [`clap`](https://github.com/clap-rs/clap/) crate for command line
  parsing
  
## LICENSE 

The Rust code here is a slam between the Buddhabrot algorithm and what's
in the O'Reilly book.  Both are covered under the [MIT
license](./docs/LICENSE-MIT.md), a copy of which is included in the docs
folder. 

Contributing to this code requires that you also read and honor a [Code
of Conduct](./docs/CODE_OF_CONDUCT.md), the summary of which is "Don't
be a jerk."
  
