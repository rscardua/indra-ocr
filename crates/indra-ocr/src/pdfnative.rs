//! PDFs in pure Rust (hayro), behind the same calls as ``pdfium``: the
//! pipeline code is shared. Pages render through hayro; the text layer comes
//! from a device that sees glyphs in content order and reports them the way
//! pdfium's text page does: tight glyph boxes, a space where words sit apart,
//! a line break where a line ends. Boxes are already in the rendered frame
//! (points, y down, rotation applied), so ``device_affine`` is the identity.
//! A document is self-contained, so parallel workers need no library copies.

use std::path::Path;
use std::sync::Arc;

use hayro::hayro_interpret::font::{Glyph, GlyphRun};
use hayro::hayro_interpret::{
    BlendMode, ClipPath, Context, Device, DrawMode, DrawProps, Image, ImageDrawProps, InterpreterCache, InterpreterSettings, SoftMask,
    TransformExt, interpret_page,
};
use hayro::hayro_syntax::{LoadPdfError, Pdf};
use hayro::kurbo::{Affine, BezPath, Point, Rect, Shape, Vec2};
use hayro::vello_cpu::color::palette::css::WHITE;
use hayro::{PixmapSettings, RenderCache, RenderSettings};

use crate::imgproc::Gray;
use crate::io::InputError;

/// Stands in for pdfium's function table; nothing to load.
pub struct Api;

/// Always zero: no lock to wait for (kept for the CLI's profile line).
pub static LOCK_WAIT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// No per-worker library instances to pick.
pub fn use_instance(_k: usize) -> Result<(), InputError> {
    Ok(())
}

/// An open PDF document.
pub struct Document {
    pdf: Pdf,
}

impl Document {
    /// Open a PDF; a locked one without the right password is an InputError
    /// saying so.
    pub fn open(path: &Path, password: Option<&str>) -> Result<Document, InputError> {
        let bytes = std::fs::read(path).map_err(|e| InputError(format!("{}: cannot open the PDF ({e})", path.display())))?;
        let pdf = Pdf::new_with_password(Arc::new(bytes), password.unwrap_or("")).map_err(|e| {
            InputError(match e {
                LoadPdfError::Decryption(hayro::hayro_syntax::DecryptionError::PasswordProtected) => {
                    format!("{}: password-protected PDF (pass password=, or --password)", path.display())
                }
                e => format!("{}: cannot open the PDF ({e:?})", path.display()),
            })
        })?;
        Ok(Document { pdf })
    }

    pub fn page_count(&self) -> usize {
        self.pdf.pages().len()
    }

    /// Run ``f`` on page ``index``.
    pub fn with_page<T>(&self, index: usize, f: impl FnOnce(&Api, PageRef<'_>) -> T) -> Result<T, InputError> {
        let page = self.pdf.pages().get(index).ok_or_else(|| InputError(format!("Failed to load page {index}.")))?;
        Ok(f(&Api, PageRef { page }))
    }
}

/// A page of a ``Document`` (valid inside ``Document::with_page``).
#[derive(Clone, Copy)]
pub struct PageRef<'a> {
    page: &'a hayro::hayro_syntax::page::Page<'a>,
}

impl PageRef<'_> {
    /// Width and height in points, as rendered (page rotation applied).
    pub fn size(&self, _api: &Api) -> (f64, f64) {
        let (w, h) = self.page.render_dimensions();
        (w as f64, h as f64)
    }

    /// The page in grayscale at ``scale`` pixels per point, on white,
    /// annotations drawn.
    pub fn render(&self, api: &Api, scale: f64) -> Gray {
        let (w, h) = self.size(api);
        let (bw, bh) = ((w * scale).ceil().max(1.0) as usize, (h * scale).ceil().max(1.0) as usize);
        let px = PixmapSettings { x_scale: scale as f32, y_scale: scale as f32, bg_color: WHITE };
        let pix = hayro::render(self.page, &RenderCache::new(), &InterpreterSettings::default(), &RenderSettings::default(), &px);
        // ponytail: hayro truncates the size, pdfium rounds up; the last partial row/column stays white
        let mut out = Gray::filled(bw, bh, 255);
        let (pw, ph) = (pix.width() as usize, pix.height() as usize);
        let data = pix.data_as_u8_slice();
        for y in 0..ph.min(bh) {
            let src = &data[y * pw * 4..][..pw.min(bw) * 4];
            for (d, c) in out.row_mut(y).iter_mut().zip(src.chunks_exact(4)) {
                // opaque on white: pdfium's grey, (30 R + 59 G + 11 B) / 100
                *d = ((30 * c[0] as u32 + 59 * c[1] as u32 + 11 * c[2] as u32) / 100) as u8;
            }
        }
        out
    }

    /// The map from ``chars`` boxes to the rendered frame: the identity.
    pub fn device_affine(&self, _api: &Api) -> [[f64; 3]; 2] {
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    }

    /// Text on the page: (total, invisible), counted in text-showing runs
    /// (pdfium counts text objects; ``max_depth`` does not apply). The
    /// pipeline uses the ratio.
    pub fn text_objects(&self, _api: &Api, _max_depth: usize) -> (usize, usize) {
        let dev = self.text_device();
        (dev.runs, dev.invisible_runs)
    }

    /// The page's characters as pdfium gives them: (code point, angle, box
    /// l r b t) in content order, with generated spaces and line breaks.
    pub fn chars(&self, _api: &Api) -> Result<Vec<(u32, f32, [f64; 4])>, InputError> {
        Ok(self.text_device().out)
    }

    fn text_device(&self) -> TextDevice {
        let (w, h) = self.page.render_dimensions();
        // pdfium's text page holds the page content only, not annotation appearances
        let settings = InterpreterSettings { render_annotations: false, ..InterpreterSettings::default() };
        let cache = InterpreterCache::new();
        let mut ctx = Context::new(
            self.page.initial_transform(true).to_kurbo(),
            Rect::new(0.0, 0.0, w as f64, h as f64),
            &cache,
            self.page.xref(),
            settings,
        );
        let mut dev = TextDevice::default();
        interpret_page(self.page, &mut ctx, &mut dev);
        dev
    }
}

/// Glyphs to pdfium-like characters.
#[derive(Default)]
struct TextDevice {
    out: Vec<(u32, f32, [f64; 4])>,
    /// The last glyph: its pen origin, writing direction, em size and box.
    last: Option<(Point, Vec2, f64, Rect)>,
    runs: usize,
    invisible_runs: usize,
}

impl TextDevice {
    fn glyph(&mut self, text: &str, t: Affine, ink: Rect) {
        let origin = t * Point::ORIGIN;
        let dir = t * Point::new(1000.0, 0.0) - origin;
        let em = (t * Point::new(0.0, 1000.0) - origin).hypot().max(1e-6);
        let unit = dir / dir.hypot().max(1e-9);
        // a glyph without ink (a space) is a point on the baseline
        let ink = if ink.area() > 0.0 { ink } else { Rect::from_points(origin, origin) };
        let space = text.chars().all(|c| c.is_whitespace());
        if let Some((po, pdir, pem, pink)) = self.last {
            let d = origin - po;
            let (along, across) = (d.dot(unit), d.cross(unit).abs());
            let pend = (Point::new(pink.x1, pink.y1) - po).dot(unit).max(0.0);
            let turned = (pdir / pdir.hypot().max(1e-9)).dot(unit) < 0.99;
            let size = em.max(pem);
            // A new line: off the baseline, turned, or back behind the last glyph.
            if turned || across > 0.5 * size || along < -size {
                self.out.push(('\n' as u32, 0.0, [f64::NAN; 4]));
            } else if !space && !matches!(self.out.last(), Some((32, ..))) && along - pend > 0.15 * size {
                // Words apart: pdfium's generated space (its box spans the gap).
                self.out.push((' ' as u32, 0.0, [pink.x1, ink.x0.max(pink.x1), ink.y0.min(pink.y0), ink.y1.max(pink.y1)]));
            }
        }
        let angle = unit.y.atan2(unit.x) as f32;
        // a ligature's characters share its box, split along the line
        let n = text.chars().count().max(1) as f64;
        for (k, c) in text.chars().enumerate() {
            let (a, b) = (ink.x0 + ink.width() * k as f64 / n, ink.x0 + ink.width() * (k + 1) as f64 / n);
            self.out.push((c as u32, angle, [a, b, ink.y0, ink.y1]));
        }
        self.last = Some((origin, dir, em, ink));
    }
}

impl<'a> Device<'a> for TextDevice {
    fn draw_path(&mut self, _: &BezPath, _: DrawProps<'a>, _: &DrawMode) {}
    fn push_clip_path(&mut self, _: &ClipPath) {}
    fn push_transparency_group(&mut self, _: f32, _: Option<SoftMask<'a>>, _: BlendMode) {}
    fn draw_image(&mut self, _: Image<'a, '_>, _: ImageDrawProps<'a>) {}
    fn pop_clip(&mut self) {}
    fn pop_transparency_group(&mut self) {}

    fn draw_glyph_run(&mut self, run: &GlyphRun<'_, 'a>, props: DrawProps<'a>, mode: &DrawMode) {
        self.runs += 1;
        self.invisible_runs += matches!(mode, DrawMode::Invisible) as usize;
        for g in run.glyphs() {
            let t = props.transform * g.transform();
            let ink = match &**g {
                Glyph::Outline(o) => t.transform_rect_bbox(o.outline().bounding_box()),
                // ponytail: Type3 glyphs get their em square, not their ink
                Glyph::Type3(_) => t.transform_rect_bbox(Rect::new(0.0, 0.0, 1000.0, 1000.0)),
            };
            // pdfium reports an unmapped code as U+FFFD-like garbage; the pipeline counts those as bad
            let text = match g.as_unicode() {
                Some(hayro::hayro_interpret::hayro_cmap::BfString::Char(c)) => c.to_string(),
                Some(hayro::hayro_interpret::hayro_cmap::BfString::String(s)) => s,
                None => '\u{FFFD}'.to_string(),
            };
            self.glyph(&text, t, ink);
        }
    }
}
