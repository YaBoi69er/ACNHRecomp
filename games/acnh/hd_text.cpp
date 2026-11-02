// Text at the screen's resolution (video.hd_text, default on; NX_HD_TEXT=0 off, NX_HD_TEXT=<factor> forced). The
// game's text is TrueType (nn::font::ScalableFont over nn::fontll), rasterised by the CPU into a glyph cache texture
// (nn::font::TextureCache) at each font's own pixel size and drawn from there: on a screen scaled past the game's 1080
// rows every glyph is that bitmap magnified, soft at 1440 rows and more so at 2160. Here the cache rasterises each
// glyph at its size times the screen scale (video.resolution, with supersampling) into a cache that many times larger
// (to the next power of two, which it needs), and the font reports every size the layouts see (advances, widths, the
// quad of each glyph) at the font's own size again: the text takes exactly the place it had, drawn from more texels.
// (Making the fonts themselves larger made the text larger: the message window sizes its text from the game's own
// font sizes, not from the font's height.) Fonts and the cache are made at boot: a later resolution change keeps the
// boot's scale until a restart.
//
// 3.0.3 (games/acnh/hooks-3.0.3.txt); a glyph node of the cache holds, in cache pixels: +0x40/+0x42 its cell's x/y,
// +0x44/+0x46 the texels it covers, +0x48 glyph width, +0x4a height, +0x4c advance, +0x4e left bearing, +0x50 baseline.
//   main+253c00 TextureCache::Initialize(this, device, const InitializeArg& arg, ...): arg+0/+4 the cache's width and
//     height (powers of two), read at 253ed0/253ee0 and kept at this+0x28/0x2c; with no memory pool (the game's case)
//     it allocates the texture's memory itself from them.
//   main+2552d0 the cache's glyph rasteriser: ScalableFontEngine::SetScale(size << 16, 0, 0, size << 16) at 2553c8,
//     then w27 = the ascent at that size (the glyph's line and baseline) before AcquireGlyphmap at 255400.
//   main+252760 ScalableFont::GetGlyph(this, Glyph* out, code): out+8 left, +0xa glyph width, +0xc advance, +0x10
//     height (the quad, in font pixels), +0xe/+0x12 texels covered, +0x14/+0x16 the cache's size, +0x18/+0x1a the cell
//     (the texture rectangle, in cache pixels), +0x28 baseline difference; filled by 252938.
//   main+2525e0 ScalableFont::GetCharWidth: a cached glyph's advance in w0 at 252684.
//   main+252690 ScalableFont::GetCharWidths: a cached glyph's left, width and advance (bytes) loaded by 252740, packed
//     from 252744 (also reached with a not yet cached glyph's widths, at the font's size already).
// The hooks change registers or the glyph being returned; the game's argument structs are left as they were.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <map>
#include <mutex>

#include <nx/runtime.h>

#include "config.h"
#include "nvn/nvn_ultrawide.h"

namespace nx {

namespace {

constexpr uint32_t kMaxCache = 8192;  // the cache texture's side at most

// The text's scale: the screen's rows over the game's (1 when off or not past them).
float text_scale() {
    static const float s = [] {
        const char* e = std::getenv("NX_HD_TEXT");
        if (e ? e[0] == '0' : !config_get_bool("video.hd_text", true)) return 1.0f;
        float f = e && std::atof(e) > 1.0 ? float(std::atof(e)) : nvn::uw_resolution_at_boot();
        // 2160 rows at most: past the screen's own rows (supersampling) the glyphs only get scaled down again.
        f = std::clamp(f, 1.0f, 2.0f);
        log("hd text: glyphs rasterised at x%.3f", double(f));
        return f;
    }();
    return s;
}

template <typename T>
T& at(uint64_t p, uint32_t off) { return *reinterpret_cast<T*>(p + off); }

int32_t down(double v) { return int32_t(std::lround(v / text_scale())); }

// Each cache grows by doublings of its shorter side until it has the glyphs' area (scale squared), to kMaxCacheBytes.
constexpr uint64_t kMaxCacheBytes = 32ull << 20;
thread_local uint32_t t_height_factor = 1;
thread_local bool t_enlarged = false;  // the cache being made is larger than the game asked for

// The cache's texture memory: TextureCache::Initialize allocates it from the game's font heap (arg+8, called at
// main+254914), which has no room for more (an enlarged cache got null there and was cleared through it at boot, even
// at 20 MB). An enlarged cache gets runtime memory instead: the game's allocator is asked for one page, which is kept
// and handed back to it by TextureCache::Finalize's free (main+254f04) in place of ours; ours is kept for the next
// cache (kernel_alloc memory is never freed).
struct CacheMemory {
    uint64_t size = 0, game = 0;
};
std::mutex g_mem_lock;
std::map<uint64_t, CacheMemory> g_in_use;          // ours -> its size and the game's page
std::multimap<uint64_t, uint64_t> g_spare;          // size -> ours, no longer used
thread_local uint64_t t_size = 0, t_align = 0;

// main+254914 (blr x8, the allocator): x0 = size, x1 = alignment.
void cache_alloc(CPU& c) {
    if (!t_enlarged) return;
    t_size = c.x[0], t_align = std::max<uint64_t>(c.x[1], 0x1000);
    c.x[0] = 0x1000;
}
// main+254918 (after it): x0 = the game's page.
void cache_allocated(CPU& c) {
    if (!t_enlarged) return;
    t_enlarged = false;
    std::lock_guard lk(g_mem_lock);
    uint64_t ours = 0;
    if (auto it = g_spare.lower_bound(t_size); it != g_spare.end() && it->second % t_align == 0) {
        ours = it->second;
        g_spare.erase(it);
    } else {
        ours = kernel_alloc(t_size, t_align);
    }
    g_in_use[ours] = {t_size, c.x[0]};
    c.x[0] = ours;
}
// main+254f00 (mov x1, x21; the free at 254f04): x0 = the cache's memory.
void cache_free(CPU& c) {
    std::lock_guard lk(g_mem_lock);
    auto it = g_in_use.find(c.x[0]);
    if (it == g_in_use.end()) return;
    g_spare.emplace(it->second.size, it->first);
    c.x[0] = it->second.game;
    g_in_use.erase(it);
}

// main+253ed8 (mov w1, #4): w8 = arg.width (stored at 253edc), x22 = arg (+4 the height). Both factors are taken here.
void cache_width(CPU& c) {
    const uint32_t w = uint32_t(c.x[8]), h = *reinterpret_cast<const uint32_t*>(c.x[22] + 4);
    uint32_t fw = 1, fh = 1;
    const double area = double(text_scale()) * text_scale();
    while (fw * fh < area) {
        uint32_t& f = w * fw <= h * fh ? fw : fh;
        const uint32_t side = &f == &fw ? w : h;
        if (side * f * 2 > kMaxCache || uint64_t(w * fw) * (h * fh) * 2 > kMaxCacheBytes) break;
        f *= 2;
    }
    c.x[8] = w * fw;
    t_height_factor = fh;
    t_enlarged = fw * fh > 1;
    if (fw * fh > 1) log("hd text: glyph cache %ux%u -> %ux%u", w, h, w * fw, h * fh);
}
// main+253ee8 (str wzr, [x19, #0x1bcc]): w8 = arg.height (stored at 253ef0).
void cache_height(CPU& c) {
    c.x[8] = uint32_t(c.x[8]) * t_height_factor;
    t_height_factor = 1;
}

// main+2553c8 (bl SetScale): w1 = w4 = the glyph's size << 16 (16.16 fixed point: any scale).
void raster_scale(CPU& c) {
    if (text_scale() <= 1.0f) return;
    const uint32_t s = uint32_t(std::lround(double(uint32_t(c.x[1])) * text_scale()));
    c.x[1] = c.x[4] = s;
}
// main+255400 (bl AcquireGlyphmap): w27 = the ascent at the glyph's size.
void raster_ascent(CPU& c) {
    if (text_scale() <= 1.0f) return;
    c.x[27] = uint32_t(int32_t(std::lround(double(int32_t(c.x[27])) * text_scale())));
}

// main+252938 (ldp x20, x19, [sp, #0x20]): x19 = the Glyph out, x0 = the glyph node, x20 = the font.
void glyph_metrics(CPU& c) {
    if (text_scale() <= 1.0f) return;
    const uint64_t g = c.x[19], n = c.x[0], font = c.x[20];
    at<int16_t>(g, 0x8) = int16_t(down(at<int16_t>(n, 0x4e)));
    // The quad is glyph width + 1 by height + 1 font pixels over the texels covered + 1 (half a texel each side).
    at<uint16_t>(g, 0xa) = uint16_t(std::max(0, down(at<uint16_t>(n, 0x48) + 1) - 1));
    at<uint16_t>(g, 0xc) = uint16_t(down(at<uint16_t>(n, 0x4c)));
    at<uint16_t>(g, 0x10) = uint16_t(std::max(1, down(at<uint16_t>(n, 0x4a) + 2) - 1));
    at<int16_t>(g, 0x28) = int16_t(down(at<uint16_t>(n, 0x50)) - at<uint16_t>(font, 0x24));
}

// main+252684 (ldp x20, x19, [sp, #0x10]): w0 = a cached glyph's advance.
void char_width(CPU& c) {
    if (text_scale() > 1.0f) c.x[0] = uint32_t(down(uint32_t(c.x[0])));
}

// main+252740 (ldrb w10, [x0, #0x4c]): a cached glyph's widths, packed at 252744 (where the other path joins).
thread_local bool t_cached_widths = false;
void char_widths_cached(CPU&) { t_cached_widths = text_scale() > 1.0f; }
void char_widths(CPU& c) {
    if (!t_cached_widths) return;
    t_cached_widths = false;
    const uint64_t n = c.x[0];
    c.x[9] = uint8_t(int8_t(std::clamp(down(at<int16_t>(n, 0x4e)), -128, 127)));
    c.x[8] = uint32_t(std::clamp(down(at<uint16_t>(n, 0x48) + 1) - 1, 0, 255));
    c.x[10] = uint32_t(std::clamp(down(at<uint16_t>(n, 0x4c)), 0, 255));
}

}  // namespace

NX_HOOK(0x253ed8, cache_width);
NX_HOOK(0x253ee8, cache_height);
NX_HOOK(0x254914, cache_alloc);
NX_HOOK(0x254918, cache_allocated);
NX_HOOK(0x254f00, cache_free);
NX_HOOK(0x2553c8, raster_scale);
NX_HOOK(0x255400, raster_ascent);
NX_HOOK(0x252938, glyph_metrics);
NX_HOOK(0x252684, char_width);
NX_HOOK(0x252740, char_widths_cached);
NX_HOOK(0x252744, char_widths);

}  // namespace nx
