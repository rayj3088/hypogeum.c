/*
 * holopress.c - convert one safetensors shard into HoloMoE plate pieces.
 *
 *   holopress SHARD.safetensors PLATE_DIR HIDDEN MOE_INTERMEDIATE [N_EXPERTS]
 *
 * Experts become ternary bitplanes with one scale per group of 128 weights
 * (~2.1 bits/weight). Two layouts are understood:
 *   fused   (Qwen3.5/3.6): "mlp.experts.gate_up_proj" / "mlp.experts.down_proj"
 *   unfused (Qwen3 MoE):   "mlp.experts.N.gate_proj|up_proj|down_proj.weight",
 *            possibly spread over several shards. Each expert is pressed into
 *            experts/parts/, and a layer is assembled into the same plate files
 *            as the fused layout once all N_EXPERTS of its experts are present.
 * Everything else is copied exactly. Vision and MTP weights are skipped.
 *
 * Safety: every piece is written to NAME.part, flushed to disk, then renamed.
 * A piece either exists completely (with a checksum footer) or not at all.
 * Re-running skips pieces whose checksum verifies, so it can be killed at any
 * moment and resumed. Prints one line per piece for the status page.
 */
#define _DEFAULT_SOURCE
#define _DARWIN_C_SOURCE
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define GROUP 128
static void die(const char *m) { fprintf(stderr, "holopress: %s\n", m); exit(2); }

/* ---------------------------------------------------------------- checksum */
typedef struct { uint64_t h; } Fnv;
static void fnv_init(Fnv *f) { f->h = 1469598103934665603ULL; }
static void fnv_add(Fnv *f, const void *p, size_t n) {
    const uint8_t *b = p; for (size_t i = 0; i < n; i++) { f->h ^= b[i]; f->h *= 1099511628211ULL; }
}

/* writer that hashes everything it writes, finishes with footer + atomic rename */
typedef struct { FILE *f; Fnv h; char tmp[4096], dst[4096]; } Out;
static void out_open(Out *o, const char *dst) {
    snprintf(o->dst, sizeof o->dst, "%s", dst);
    snprintf(o->tmp, sizeof o->tmp, "%s.part", dst);
    o->f = fopen(o->tmp, "wb"); if (!o->f) { perror(o->tmp); exit(2); }
    fnv_init(&o->h);
}
static void out_write(Out *o, const void *p, size_t n) {
    if (fwrite(p, 1, n, o->f) != n) die("write failed (disk full?)");
    fnv_add(&o->h, p, n);
}
static void out_close(Out *o) {
    uint32_t pad = 0; uint64_t h = o->h.h;
    if (fwrite("HMOK", 1, 4, o->f) != 4 || fwrite(&pad, 4, 1, o->f) != 1 || fwrite(&h, 8, 1, o->f) != 1) die("write failed");
    if (fflush(o->f) || fsync(fileno(o->f))) die("flush failed");
    fclose(o->f);
    if (rename(o->tmp, o->dst)) { perror("rename"); exit(2); }
}
/* does a finished, checksum-valid piece already exist? */
static int piece_ok(const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) return 0;
    if (fseek(f, 0, SEEK_END)) { fclose(f); return 0; }
    long n = ftell(f); if (n < 16) { fclose(f); return 0; }
    char mg[4]; uint32_t pad; uint64_t want;
    fseek(f, n - 16, SEEK_SET);
    if (fread(mg, 1, 4, f) != 4 || fread(&pad, 4, 1, f) != 1 || fread(&want, 8, 1, f) != 1 || memcmp(mg, "HMOK", 4)) { fclose(f); return 0; }
    fseek(f, 0, SEEK_SET);
    Fnv h; fnv_init(&h); static uint8_t buf[1 << 20]; long left = n - 16;
    while (left > 0) { size_t k = left > (long)sizeof buf ? sizeof buf : (size_t)left; if (fread(buf, 1, k, f) != k) { fclose(f); return 0; } fnv_add(&h, buf, k); left -= (long)k; }
    fclose(f);
    return h.h == want;
}

/* ---------------------------------------------------------------- safetensors header */
typedef struct { char name[256]; char dtype[16]; int ndim; uint64_t shape[8]; uint64_t a, b; } Ten;

static const char *skip_ws(const char *p) { while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') p++; return p; }
static const char *read_str(const char *p, char *out, size_t cap) {
    p = skip_ws(p); if (*p != '"') die("bad header (string)"); p++;
    size_t n = 0;
    while (*p && *p != '"') { if (*p == '\\' && p[1]) p++; if (n + 1 < cap) out[n++] = *p; p++; }
    out[n] = 0; return *p ? p + 1 : p;
}
static const char *skip_value(const char *p) {
    p = skip_ws(p);
    if (*p == '"') { char tmp[4096]; return read_str(p, tmp, sizeof tmp); }
    if (*p == '{' || *p == '[') {
        int depth = 0; int instr = 0;
        for (; *p; p++) {
            if (instr) { if (*p == '\\') p++; else if (*p == '"') instr = 0; continue; }
            if (*p == '"') instr = 1; else if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') { if (--depth == 0) return p + 1; }
        }
        return p;
    }
    while (*p && *p != ',' && *p != '}') p++;
    return p;
}
static int parse_header(const char *js, Ten **out) {
    int cap = 4096, n = 0; Ten *t = calloc((size_t)cap, sizeof *t);
    const char *p = skip_ws(js); if (*p != '{') die("bad header");
    p++;
    for (;;) {
        p = skip_ws(p); if (*p == '}') break; if (*p == ',') { p++; continue; }
        char key[256]; p = read_str(p, key, sizeof key); p = skip_ws(p); if (*p != ':') die("bad header (:)"); p++;
        if (!strcmp(key, "__metadata__")) { p = skip_value(p); continue; }
        if (n == cap) { cap *= 2; t = realloc(t, (size_t)cap * sizeof *t); }
        Ten *x = &t[n]; memset(x, 0, sizeof *x); snprintf(x->name, sizeof x->name, "%s", key);
        p = skip_ws(p); if (*p != '{') die("bad header (tensor)"); p++;
        for (;;) {
            p = skip_ws(p); if (*p == '}') { p++; break; } if (*p == ',') { p++; continue; }
            char f[64]; p = read_str(p, f, sizeof f); p = skip_ws(p); p++;  /* ':' */
            p = skip_ws(p);
            if (!strcmp(f, "dtype")) p = read_str(p, x->dtype, sizeof x->dtype);
            else if (!strcmp(f, "shape") || !strcmp(f, "data_offsets")) {
                if (*p != '[') die("bad header (array)");
                p++;
                uint64_t v[8]; int k = 0;
                for (;;) { p = skip_ws(p); if (*p == ']') { p++; break; } if (*p == ',') { p++; continue; }
                    char *e; uint64_t num = strtoull(p, &e, 10); if (e == p) die("bad number"); if (k < 8) v[k++] = num; p = e; }
                if (f[0] == 's') { x->ndim = k; memcpy(x->shape, v, sizeof(uint64_t) * (size_t)k); }
                else { x->a = v[0]; x->b = k > 1 ? v[1] : v[0]; }
            } else p = skip_value(p);
        }
        n++;
    }
    *out = t; return n;
}

/* ---------------------------------------------------------------- conversion */
static float bf2f(uint16_t h) { uint32_t u = (uint32_t)h << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t f2bf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7FFF + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

static void mkdirs(const char *path) {
    char tmp[4096]; snprintf(tmp, sizeof tmp, "%s", path);
    for (char *p = tmp + 1; *p; p++) if (*p == '/') { *p = 0; mkdir(tmp, 0755); *p = '/'; }
    mkdir(tmp, 0755);
}

/* one matrix (rows x cols) read through strides from a bf16 slab -> group ternary */
static void ternary_matrix(Out *o, const uint16_t *slab, size_t off, size_t rs, size_t cs, uint32_t rows, uint32_t cols) {
    uint32_t gpr = cols / GROUP;
    uint64_t *masks = calloc((size_t)rows * gpr * 4, 8); uint16_t *scales = calloc((size_t)rows * gpr, 2);
    float w[GROUP];
    for (uint32_t r = 0; r < rows; r++)
        for (uint32_t g = 0; g < gpr; g++) {
            /* TWN optimal ternary: threshold = 0.7 * mean|w|, scale = mean |w| of the survivors */
            double s = 0;
            for (int j = 0; j < GROUP; j++) { w[j] = bf2f(slab[off + (size_t)r * rs + (size_t)(g * GROUP + j) * cs]); s += fabsf(w[j]); }
            float thr = 0.7f * (float)(s / GROUP);
            uint64_t *m = masks + ((size_t)r * gpr + g) * 4;
            double ks = 0; int kn = 0;
            for (int j = 0; j < GROUP; j++) {
                if (w[j] > thr) { m[j / 64] |= 1ULL << (j % 64); ks += w[j]; kn++; }
                else if (w[j] < -thr) { m[2 + j / 64] |= 1ULL << (j % 64); ks -= w[j]; kn++; }
            }
            float a = kn ? (float)(ks / kn) : 0.0f;
            scales[(size_t)r * gpr + g] = f2bf(a);
        }
    out_write(o, masks, (size_t)rows * gpr * 32); out_write(o, scales, (size_t)rows * gpr * 2);
    free(masks); free(scales);
}

/* ---------------------------------------------------------------- unfused experts: parts -> layer files */
static void part_path(char *p, size_t n, const char *outdir, int L, int e, char kind) { snprintf(p, n, "%s/experts/parts/L%02d/E%04d.%c", outdir, L, e, kind); }
static void append_part(Out *o, const char *path) {
    FILE *f = fopen(path, "rb"); if (!f) die("missing expert part");
    fseek(f, 0, SEEK_END); long n = ftell(f) - 16; fseek(f, 0, SEEK_SET);
    static uint8_t buf[1 << 20];
    while (n > 0) { size_t k = n > (long)sizeof buf ? sizeof buf : (size_t)n; if (fread(buf, 1, k, f) != k) die("read failed"); out_write(o, buf, k); n -= (long)k; }
    fclose(f);
}
static void assemble_layer(const char *outdir, int L, int E, uint32_t H, uint32_t I) {
    char gu[4096], dn[4096], p[4096];
    snprintf(gu, sizeof gu, "%s/experts/L%02d.gu", outdir, L); snprintf(dn, sizeof dn, "%s/experts/L%02d.dn", outdir, L);
    if (piece_ok(gu) && piece_ok(dn)) return;
    for (int e = 0; e < E; e++) { const char *k = "gud"; for (int j = 0; j < 3; j++) { part_path(p, sizeof p, outdir, L, e, k[j]); if (!piece_ok(p)) return; } }
    for (int m = 0; m < 2; m++) {
        Out o; out_open(&o, m ? dn : gu);
        uint32_t hdr[16] = {0}; memcpy(hdr, "HMX1", 4);
        hdr[1] = 2; hdr[2] = (uint32_t)E; hdr[3] = m ? 1 : 2; hdr[4] = m ? H : I; hdr[5] = m ? I : H; hdr[6] = GROUP; hdr[7] = hdr[5] / GROUP;
        out_write(&o, hdr, sizeof hdr);
        for (int e = 0; e < E; e++) {
            if (m) { part_path(p, sizeof p, outdir, L, e, 'd'); append_part(&o, p); }
            else { part_path(p, sizeof p, outdir, L, e, 'g'); append_part(&o, p); part_path(p, sizeof p, outdir, L, e, 'u'); append_part(&o, p); }
        }
        out_close(&o);
    }
    for (int e = 0; e < E; e++) { const char *k = "gud"; for (int j = 0; j < 3; j++) { part_path(p, sizeof p, outdir, L, e, k[j]); unlink(p); } }
    snprintf(p, sizeof p, "%s/experts/parts/L%02d", outdir, L); rmdir(p);
    printf("LAYER L%02d experts assembled (%d)\n", L, E); fflush(stdout);
}

int main(int argc, char **argv) {
    if (argc != 5 && argc != 6) { fprintf(stderr, "usage: holopress SHARD PLATE_DIR HIDDEN MOE_INTERMEDIATE [N_EXPERTS]\n"); return 2; }
    int NEXP = argc == 6 ? atoi(argv[5]) : 0;
    const char *shard = argv[1], *outdir = argv[2]; uint64_t H = strtoull(argv[3], 0, 10), I = strtoull(argv[4], 0, 10);
    if (!H || !I || H % GROUP || I % GROUP) die("hidden and moe_intermediate must be multiples of 128");
    FILE *f = fopen(shard, "rb"); if (!f) { perror(shard); return 2; }
    uint64_t hl; if (fread(&hl, 8, 1, f) != 1 || hl > (64u << 20)) die("not a safetensors file");
    char *js = calloc(hl + 1, 1); if (fread(js, 1, hl, f) != hl) die("truncated header");
    Ten *t; int n = parse_header(js, &t);
    uint64_t base = 8 + hl, maxb = 0; for (int i = 0; i < n; i++) if (t[i].b > maxb) maxb = t[i].b;
    struct stat sb; fstat(fileno(f), &sb);
    if ((uint64_t)sb.st_size != base + maxb) { fprintf(stderr, "holopress: shard size %lld != expected %llu (incomplete download)\n", (long long)sb.st_size, (unsigned long long)(base + maxb)); return 3; }
    if (strlen(outdir) > 1800) die("plate path too long");
    char d1[2048], d2[2048]; snprintf(d1, sizeof d1, "%s/experts", outdir); snprintf(d2, sizeof d2, "%s/dense", outdir);
    mkdirs(d1); mkdirs(d2);
    printf("SHARD %s tensors %d\n", shard, n); fflush(stdout);

    for (int i = 0; i < n; i++) {
        Ten *x = &t[i]; char path[4096];
        int is_gu = strstr(x->name, "mlp.experts.gate_up_proj") != NULL, is_dn = strstr(x->name, "mlp.experts.down_proj") != NULL;
        const char *ue = strstr(x->name, "mlp.experts."); int uexp = -1; char ukind = 0;
        if (ue && !is_gu && !is_dn && ue[12] >= '0' && ue[12] <= '9') {          /* unfused: mlp.experts.N.gate_proj.weight */
            char *e2; uexp = (int)strtol(ue + 12, &e2, 10);
            ukind = !strncmp(e2, ".gate_proj", 10) ? 'g' : !strncmp(e2, ".up_proj", 8) ? 'u' : !strncmp(e2, ".down_proj", 10) ? 'd' : 0;
            if (!ukind) die("unknown expert tensor");
        }
        if (!strncmp(x->name, "model.visual.", 13) || !strncmp(x->name, "mtp.", 4) || strstr(x->name, ".mtp.")) {
            printf("PIECE %d/%d SKIP %s\n", i + 1, n, x->name); fflush(stdout); continue;
        }
        if (ukind) {
            const char *lp = strstr(x->name, "layers."); if (!lp) die("expert tensor without layer number");
            int L = atoi(lp + 7); char dir[4096];
            snprintf(dir, sizeof dir, "%s/parts/L%02d", d1, L); mkdirs(dir);
            part_path(path, sizeof path, outdir, L, uexp, ukind);
            snprintf(dir, sizeof dir, "%s/L%02d.gu", d1, L);
            if (piece_ok(path) || piece_ok(dir)) { printf("PIECE %d/%d HAVE %s\n", i + 1, n, x->name); fflush(stdout); continue; }
            uint64_t want_r = ukind == 'd' ? H : I, want_c = ukind == 'd' ? I : H;
            if (strcmp(x->dtype, "BF16") || x->ndim != 2 || x->shape[0] != want_r || x->shape[1] != want_c) die("unexpected expert tensor shape");
            size_t cnt = (size_t)(want_r * want_c); uint16_t *slab = malloc(cnt * 2);
            if (fseeko(f, (off_t)(base + x->a), SEEK_SET) || fread(slab, 2, cnt, f) != cnt) die("read failed");
            Out o; out_open(&o, path); ternary_matrix(&o, slab, 0, (size_t)want_c, 1, (uint32_t)want_r, (uint32_t)want_c); out_close(&o); free(slab);
            if (uexp % 32 == 31) { printf("PROGRESS %s\n", x->name); fflush(stdout); }
            printf("PIECE %d/%d DONE %s\n", i + 1, n, x->name); fflush(stdout);
        } else if (is_gu || is_dn) {
            const char *lp = strstr(x->name, "layers."); if (!lp) die("expert tensor without layer number");
            int L = atoi(lp + 7);
            snprintf(path, sizeof path, "%s/L%02d.%s", d1, L, is_gu ? "gu" : "dn");
            if (piece_ok(path)) { printf("PIECE %d/%d HAVE %s\n", i + 1, n, x->name); fflush(stdout); continue; }
            if (strcmp(x->dtype, "BF16") || x->ndim != 3) die("expert tensor must be 3-D BF16");
            uint64_t E = x->shape[0], d1n = x->shape[1], d2n = x->shape[2];
            /* figure out orientation from the shape */
            uint32_t rows, cols, nmat; size_t rs, cs, up_off = 0;
            if (is_gu) {
                rows = (uint32_t)I; cols = (uint32_t)H; nmat = 2;
                if (d1n == 2 * I && d2n == H) { rs = H; cs = 1; up_off = I * H; }          /* [E, 2I, H] */
                else if (d1n == H && d2n == 2 * I) { rs = 1; cs = 2 * I; up_off = I; }     /* [E, H, 2I] */
                else die("unexpected gate_up_proj shape");
            } else {
                rows = (uint32_t)H; cols = (uint32_t)I; nmat = 1;
                if (d1n == H && d2n == I) { rs = I; cs = 1; }                              /* [E, H, I] */
                else if (d1n == I && d2n == H) { rs = 1; cs = H; }                         /* [E, I, H] */
                else die("unexpected down_proj shape");
            }
            Out o; out_open(&o, path);
            uint32_t hdr[16] = {0}; memcpy(hdr, "HMX1", 4);
            hdr[1] = 2; hdr[2] = (uint32_t)E;   /* v2: TWN ternary */ hdr[3] = nmat; hdr[4] = rows; hdr[5] = cols; hdr[6] = GROUP; hdr[7] = cols / GROUP;
            out_write(&o, hdr, sizeof hdr);
            size_t slab_n = (size_t)(d1n * d2n); uint16_t *slab = malloc(slab_n * 2);
            for (uint64_t e = 0; e < E; e++) {
                if (fseeko(f, (off_t)(base + x->a + e * slab_n * 2), SEEK_SET) || fread(slab, 2, slab_n, f) != slab_n) die("read failed");
                ternary_matrix(&o, slab, 0, rs, cs, rows, cols);
                if (nmat == 2) ternary_matrix(&o, slab, up_off, rs, cs, rows, cols);
                if ((e + 1) % 32 == 0) { printf("PROGRESS %s %llu/%llu\n", x->name, (unsigned long long)(e + 1), (unsigned long long)E); fflush(stdout); }
            }
            free(slab); out_close(&o);
            printf("PIECE %d/%d DONE %s\n", i + 1, n, x->name); fflush(stdout);
        } else {
            if (strlen(d2) + strlen(x->name) + 8 > sizeof path) die("path too long");
            snprintf(path, sizeof path, "%s/%.255s.t", d2, x->name);
            if (piece_ok(path)) { printf("PIECE %d/%d HAVE %s\n", i + 1, n, x->name); fflush(stdout); continue; }
            uint32_t code = !strcmp(x->dtype, "F32") ? 0 : !strcmp(x->dtype, "BF16") ? 1 : !strcmp(x->dtype, "F16") ? 2 : 3;
            Out o; out_open(&o, path);
            uint32_t h4[4] = {0}; memcpy(h4, "HMD1", 4); h4[1] = code; h4[2] = (uint32_t)x->ndim;
            uint64_t shp[8] = {0}; memcpy(shp, x->shape, sizeof(uint64_t) * (size_t)x->ndim);
            uint64_t nb = x->b - x->a;
            out_write(&o, h4, sizeof h4); out_write(&o, shp, sizeof shp); out_write(&o, &nb, 8);
            static uint8_t buf[8 << 20]; uint64_t left = nb;
            if (fseeko(f, (off_t)(base + x->a), SEEK_SET)) die("seek failed");
            while (left) { size_t k = left > sizeof buf ? sizeof buf : (size_t)left; if (fread(buf, 1, k, f) != k) die("read failed"); out_write(&o, buf, k); left -= k; }
            out_close(&o);
            printf("PIECE %d/%d DONE %s\n", i + 1, n, x->name); fflush(stdout);
        }
    }
    fclose(f);
    if (NEXP > 0) {                                  /* finish any layer whose experts are now all present */
        char pd[4096]; snprintf(pd, sizeof pd, "%s/parts", d1);
        for (int L = 0; L < 1000; L++) { char ld[4200]; struct stat st; snprintf(ld, sizeof ld, "%s/L%02d", pd, L); if (!stat(ld, &st)) assemble_layer(outdir, L, NEXP, (uint32_t)H, (uint32_t)I); }
        rmdir(pd);                                   /* only succeeds once every layer is assembled */
    }
    printf("DONE %s\n", shard); fflush(stdout);
    return 0;
}
